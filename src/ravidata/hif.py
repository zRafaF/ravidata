"""HIF binary protocol (data channel): packet parsing, frame reassembly and packet building.

Packet layout (HIF Communication Format manual §2, §4.6.1; legacy ``rd_6x_parser.c``)::

    0xA5 | check8 | header u32 LE | [ext u32 if ext bit] | payload (<= 4095 B) | [check32 u32 if check bit]

    check8  = ~(0xA5 + the 4 header bytes) & 0xFF
    header  = flags[7:0] (type:2, req:1, enc:1, check:1, more:1, ext:1, mac:1)
              | msg_id[15:8] | payload_len[27:16] (12 bit) | seq[30:28] | frag[31]
    check32 = ~sum(dwords of [header u32 | ext | payload], last partial dword zero padded)  (mod 2^32)
              i.e. the 4 header bytes ARE included (vendor HostSample hif_protocol.md / collect_Tracking.py,
              confirmed on hardware 2026-09-25: excluding them made every packet fail).
              Vendor note: some firmware (r3_databox) shows a constant +5 offset; the parser learns a
              constant offset if it repeats (see ``HifParser.learned_c32_offset``).

Datacube messages (msg_id 0xC1) start with ``{frame_index, frame_length_dwords,
data_offset_dwords}`` (3 x u32) followed by FFT-bin DWORDs placed at ``data_offset``.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .logsetup import get_logger

log = get_logger("hif")

MAGIC = 0xA5
MSG_CUBE = 0xC1
MSG_MULTI_FFT = 0xC2
MSG_POINTS = 0xC3
MSG_DEBUG = 0xC6
MAX_PAYLOAD = 4095
FIXED_HDR = 6
MSG_NAMES = {MSG_CUBE: "cube", MSG_MULTI_FFT: "multi_fft", MSG_POINTS: "points", MSG_DEBUG: "debug"}


def check8(header_bytes: bytes) -> int:
    return (~(MAGIC + sum(header_bytes))) & 0xFF


def check32(data: bytes) -> int:
    if len(data) % 4:
        data = data + b"\x00" * (4 - len(data) % 4)
    s = int(np.frombuffer(data, dtype="<u4").sum(dtype=np.uint64)) & 0xFFFFFFFF
    return (0xFFFFFFFF - s) & 0xFFFFFFFF


@dataclass
class Header:
    type: int
    req: int
    enc: int
    check: int
    more: int
    ext: int
    mac: int
    msg_id: int
    length: int
    seq: int
    frag: int

    @classmethod
    def unpack(cls, raw: int) -> "Header":
        f = raw & 0xFF
        return cls(type=f & 3, req=(f >> 2) & 1, enc=(f >> 3) & 1, check=(f >> 4) & 1, more=(f >> 5) & 1,
                   ext=(f >> 6) & 1, mac=(f >> 7) & 1, msg_id=(raw >> 8) & 0xFF, length=(raw >> 16) & 0xFFF,
                   seq=(raw >> 28) & 7, frag=(raw >> 31) & 1)

    def pack(self) -> int:
        f = (self.type & 3) | (self.req << 2) | (self.enc << 3) | (self.check << 4) | (self.more << 5) \
            | (self.ext << 6) | (self.mac << 7)
        return f | (self.msg_id << 8) | ((self.length & 0xFFF) << 16) | ((self.seq & 7) << 28) | ((self.frag & 1) << 31)


@dataclass
class AssembledFrame:
    frame_index: int
    words: np.ndarray            # uint32 DWORDs of the whole cube, exactly as received
    n_fragments: int
    complete: bool = True


@dataclass
class ParserStats:
    bytes_in: int = 0
    packets_ok: int = 0
    frames_ok: int = 0
    frames_incomplete: int = 0      # replaced by a newer frame before completion
    resyncs: int = 0                # bytes skipped looking for magic
    skipped_bytes: int = 0
    errors: Counter = field(default_factory=Counter)
    msg_ids: Counter = field(default_factory=Counter)
    frame_len_dwords_seen: Counter = field(default_factory=Counter)
    c32_offset_accepted: int = 0
    c32_ignored: int = 0

    def to_dict(self) -> dict:
        return {
            "bytes_in": self.bytes_in, "packets_ok": self.packets_ok, "frames_ok": self.frames_ok,
            "frames_incomplete": self.frames_incomplete, "resyncs": self.resyncs,
            "skipped_bytes": self.skipped_bytes, "errors": dict(self.errors),
            "msg_ids": {hex(k): v for k, v in self.msg_ids.items()},
            "frame_len_dwords_seen": {str(k): v for k, v in self.frame_len_dwords_seen.items()},
            "c32_offset_accepted": self.c32_offset_accepted, "c32_ignored": self.c32_ignored,
        }


class HifParser:
    """Streaming parser: feed arbitrary byte chunks, get completed cube frames back."""

    def __init__(self):
        self.buf = bytearray()
        self.stats = ParserStats()
        self._cur_idx: int | None = None
        self._cur_len = 0
        self._cur_words: np.ndarray | None = None
        self._cur_filled = 0
        self._cur_frags = 0
        self.learned_c32_offset: int | None = None
        self._c32_offsets: Counter = Counter()
        self._c32_consecutive = 0
        self.check32_mode = "strict"          # "strict" | "ignore" (check8 only)
        self.auto_ignore_check32 = True

    def reset(self):
        self.buf.clear()
        self._cur_idx = None

    def feed(self, data: bytes) -> list[AssembledFrame]:
        out: list[AssembledFrame] = []
        self.stats.bytes_in += len(data)
        self.buf.extend(data)
        buf = self.buf
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                if buf:
                    self.stats.skipped_bytes += len(buf)
                    self.stats.resyncs += 1
                buf.clear()
                break
            if i > 0:
                self.stats.skipped_bytes += i
                self.stats.resyncs += 1
                del buf[:i]
            if len(buf) < FIXED_HDR:
                break
            hdr_bytes = bytes(buf[2:6])
            if buf[1] != check8(hdr_bytes):
                self._err("check8")
                del buf[:1]
                continue
            h = Header.unpack(struct.unpack("<I", hdr_bytes)[0])
            ext_len = 4 if h.ext else 0
            total = FIXED_HDR + ext_len + h.length + (4 if h.check else 0)
            if len(buf) < total:
                break
            payload = bytes(buf[FIXED_HDR + ext_len: FIXED_HDR + ext_len + h.length])
            if h.check:
                (rx_c32,) = struct.unpack_from("<I", buf, FIXED_HDR + ext_len + h.length)
                covered = bytes(buf[2: FIXED_HDR + ext_len + h.length])     # header + ext + payload
                calc = check32(covered)
                if rx_c32 != calc and self.check32_mode == "ignore":
                    self.stats.c32_ignored += 1
                elif rx_c32 != calc:
                    self._c32_consecutive += 1
                    if self._c32_consecutive >= 20 and self.learned_c32_offset is None and self.auto_ignore_check32:
                        self.check32_mode = "ignore"
                        log.warning("HIF check32 failed on %d consecutive packets with valid check8: switching to "
                                    "check8-only validation (vendor: 'Check8 is 100%% reliable'). Offsets seen: %s",
                                    self._c32_consecutive, dict(self._c32_offsets.most_common(5)))
                    off = (calc - rx_c32) & 0xFFFFFFFF
                    if self.learned_c32_offset is not None and off == self.learned_c32_offset:
                        self.stats.c32_offset_accepted += 1
                    else:
                        self._c32_offsets[off] += 1
                        if self._c32_offsets[off] >= 5 and off < 64 and self.learned_c32_offset is None:
                            self.learned_c32_offset = off
                            log.warning("HIF check32 differs by a constant %d on %d packets: accepting that offset "
                                        "(known vendor firmware quirk)", off, self._c32_offsets[off])
                        self._err("check32")
                        del buf[:1]
                        continue
                else:
                    self._c32_consecutive = 0
            del buf[:total]
            self.stats.packets_ok += 1
            self.stats.msg_ids[h.msg_id] += 1
            if h.msg_id == MSG_CUBE:
                fr = self._on_cube_payload(payload)
                if fr is not None:
                    out.append(fr)
            else:
                log.debug("non-cube HIF message 0x%02X len=%d", h.msg_id, h.length)
        return out

    def _err(self, kind: str):
        self.stats.errors[kind] += 1
        log.debug("HIF %s error (total %d)", kind, self.stats.errors[kind])

    def _on_cube_payload(self, payload: bytes) -> AssembledFrame | None:
        if len(payload) < 12:
            self._err("payload_short")
            return None
        fidx, flen, off = struct.unpack_from("<III", payload, 0)
        data = payload[12:]
        if len(data) % 4:
            self._err("payload_not_dword")
            data = data[: len(data) - len(data) % 4]
        if flen == 0 or flen > 1 << 20:
            self._err("bad_frame_len")
            return None
        if fidx != self._cur_idx or flen != self._cur_len:
            if self._cur_idx is not None and self._cur_filled < self._cur_len:
                self.stats.frames_incomplete += 1
                log.debug("frame %s dropped incomplete (%d/%d dwords)", self._cur_idx, self._cur_filled, self._cur_len)
            self._cur_idx, self._cur_len = fidx, flen
            self._cur_words = np.zeros(flen, dtype=np.uint32)
            self._cur_filled = 0
            self._cur_frags = 0
            self.stats.frame_len_dwords_seen[flen] += 1
        nd = len(data) // 4
        if off + nd > self._cur_len:
            self._err("fragment_overrun")
            return None
        self._cur_words[off: off + nd] = np.frombuffer(data, dtype="<u4")
        self._cur_filled += nd
        self._cur_frags += 1
        if self._cur_filled >= self._cur_len:
            fr = AssembledFrame(frame_index=fidx, words=self._cur_words, n_fragments=self._cur_frags)
            self.stats.frames_ok += 1
            self._cur_idx = None
            self._cur_filled = 0
            return fr
        return None


# ----------------------------------------------------------------- building (sim + tests)
def build_packet(msg_id: int, payload: bytes, seq: int = 0, with_check32: bool = True) -> bytes:
    h = Header(type=2, req=0, enc=0, check=1 if with_check32 else 0, more=0, ext=0, mac=0,
               msg_id=msg_id, length=len(payload), seq=seq, frag=0)
    hb = struct.pack("<I", h.pack())
    out = bytes([MAGIC, check8(hb)]) + hb + payload
    if with_check32:
        out += struct.pack("<I", check32(hb + payload))
    return out


def device_piece_bytes(cube_bytes: int) -> int:
    """Cube data bytes per 0xC1 message as the RS6130 firmware splits them (mmw_report.c: 1 KB if the cube
    is <= 1 KB, 2 KB if <= 8 KB, else 3 KB). Matches `getreportcfg` split_size on the bench."""
    return 1024 if cube_bytes <= 1024 else 2048 if cube_bytes <= 8192 else 3072


def build_cube_packets(frame_index: int, words: np.ndarray, max_payload: int = MAX_PAYLOAD) -> bytes:
    words = np.asarray(words, dtype="<u4")
    per = (max_payload - 12) // 4
    out = bytearray()
    for k, off in enumerate(range(0, len(words), per)):
        chunk = words[off: off + per]
        payload = struct.pack("<III", frame_index, len(words), off) + chunk.tobytes()
        out += build_packet(MSG_CUBE, payload, seq=k & 7)
    return bytes(out)
