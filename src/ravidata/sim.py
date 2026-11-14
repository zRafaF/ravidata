"""A deliberately simple RS6130 simulator, so the whole stack runs without hardware.

It is **not** a physics-grade model. It exists to exercise the parser, decoder, recorder,
processing and GUI end to end, and to create scenes with known ground truth for smoke tests.

What it models: FMCW beat tones per scatterer (range, 3 RX phases from the L-array geometry,
a rough elevation/azimuth gain pattern), TX leakage in the first bins, ground clutter, a
user walking forward with a side-to-side sweep, optional Doppler intervals, int16 quantisation and
HIF packetisation. What it does not model: multipath, HPF, saturation, real antenna patterns,
chip DSP details.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .config import ADC_FS_HZ, C, RX_SPACING_M, RadarConfig
from .cube import SCALE, iq_to_words
from .hif import build_cube_packets, device_piece_bytes


@dataclass
class Scatterer:
    x: float          # m ahead of the walk start
    y: float          # m lateral (+ = right)
    z: float          # m above ground
    amp: float        # amplitude in LSB at 1 m on boresight (sqrt(RCS)-like)
    label: str = ""


@dataclass
class Scene:
    name: str = "approach"
    sensor_height_m: float = 0.9
    pitch_deg: float = 0.0            # radar boresight pitch (+ = up)
    walk_speed_mps: float = 1.0
    walk_start_x: float = 0.0
    walk_length_m: float = 3.0        # loops back after this distance
    sweep_amp_deg: float = 20.0
    sweep_hz: float = 1.0
    noise_lsb: float = 1.5
    leakage_lsb: float = 3000.0
    scatterers: list[Scatterer] = field(default_factory=list)


def scene_presets() -> dict[str, Scene]:
    ground = [Scatterer(x=x, y=y, z=0.0, amp=6.0, label="ground") for x in np.arange(0.5, 6.0, 0.35)
              for y in (-0.6, 0.0, 0.6)]
    S = {}
    S["branch+pole"] = Scene(name="branch+pole", scatterers=ground + [
        Scatterer(3.3, 0.0, 1.70, 60.0, "branch@1.7m"),
        *[Scatterer(4.5, 0.3, z, 80.0, "pole") for z in np.arange(0.1, 2.4, 0.25)],
    ])
    S["head-branch-only"] = Scene(name="head-branch-only", sweep_amp_deg=0.0, scatterers=ground + [
        Scatterer(3.3, 0.0, 1.70, 40.0, "branch@1.7m")])
    S["empty"] = Scene(name="empty", scatterers=ground)
    S["static-target-1m"] = Scene(name="static-target-1m", walk_speed_mps=0.0, sweep_amp_deg=0.0,
                                  scatterers=[Scatterer(1.0, 0.0, 0.9, 200.0, "reflector@1m boresight")])
    return S


def _pattern_amp(el: float, az: float) -> float:
    """Rough 1-way amplitude pattern (dB ~ -12 at +-60 deg in each plane)."""
    db = -12.0 * ((el / math.radians(60)) ** 2 + (az / math.radians(60)) ** 2)
    return 10 ** (db / 20)


class Simulator:
    def __init__(self, cfg: RadarConfig, scene: Scene, seed: int = 0):
        self.cfg = cfg
        self.scene = scene
        self.d = cfg.derive()
        self.rng = np.random.default_rng(seed)
        self.frame_index = 0
        self.t = 0.0

    def truth(self, t: float) -> dict:
        sc = self.scene
        dist = (sc.walk_speed_mps * t) % max(1e-6, sc.walk_length_m) if sc.walk_speed_mps > 0 else 0.0
        return {"t": t, "sensor_x": sc.walk_start_x + dist, "yaw_deg": sc.sweep_amp_deg * math.sin(2 * math.pi * sc.sweep_hz * t)}

    def cube_at(self, t: float) -> tuple[np.ndarray, dict]:
        cfg, d, sc = self.cfg, self.d, self.scene
        N, nfft = d.n_samples, d.range_fft_len
        K = d.n_intervals
        lam = d.wavelength_mm / 1000
        tr = self.truth(t)
        sx, yaw = tr["sensor_x"], math.radians(tr["yaw_deg"])
        pitch = math.radians(sc.pitch_deg)
        rmax = cfg.range_max_mm / 1000
        n = np.arange(N)
        win = np.hanning(N).astype(np.float32)
        tc = (d.interval_spacing_us or 0) * 1e-6
        # time-domain beat signal per interval and RX
        sig = np.zeros((K, 3, N), dtype=np.complex64)
        visible = []
        for s in sc.scatterers:
            dx, dy, dz = s.x - sx, s.y, s.z - sc.sensor_height_m
            if dx <= 0.05:
                continue
            # rotate world->radar: yaw (sweep) then pitch
            xr = dx * math.cos(yaw) + dy * math.sin(yaw)
            yr = -dx * math.sin(yaw) + dy * math.cos(yaw)
            xr2 = xr * math.cos(pitch) + dz * math.sin(pitch)
            zr = -xr * math.sin(pitch) + dz * math.cos(pitch)
            r = math.sqrt(xr2 ** 2 + yr ** 2 + zr ** 2)
            if r >= rmax or xr2 <= 0:
                continue
            el = math.asin(zr / r)
            az = math.asin(yr / r)
            a = s.amp * _pattern_amp(el, az) ** 2 / (r ** 2)
            vr = -sc.walk_speed_mps * dx / r
            fb = r / rmax * ADC_FS_HZ
            base = np.exp(1j * (2 * math.pi * fb * n / ADC_FS_HZ + 4 * math.pi * r / lam))
            k2 = 2 * math.pi * RX_SPACING_M / lam
            ph = np.array([np.exp(-1j * k2 * math.sin(az)),      # RX0: offset along -X of the chip
                           1.0 + 0j,                            # RX1: corner
                           np.exp(1j * k2 * math.sin(el))])     # RX2: offset along +Z
            for k in range(K):
                dop = np.exp(1j * 4 * math.pi * vr * k * tc / lam)
                sig[k] += (a * dop) * ph[:, None] * base[None, :]
            visible.append({"label": s.label, "r": round(r, 3), "el_deg": round(math.degrees(el), 1),
                            "z": s.z, "amp": round(a, 2)})
        spec = np.fft.fft(sig * win[None, None, :], n=nfft, axis=2) / (N * 0.5)
        # TX leakage at the first bins (fixed complex template)
        leak = sc.leakage_lsb * np.array([1.0, 0.45, 0.08]) * np.exp(1j * np.array([0.3, 1.1, 2.0]))
        spec[:, :, :3] += leak[None, None, :]
        if K > 1 and cfg.dop_fft:
            spec = np.fft.fftshift(np.fft.fft(spec * np.hanning(K)[:, None, None], axis=0) / (K * 0.5), axes=0)
        spec = spec.astype(np.complex64)
        noise = (self.rng.standard_normal(spec.shape) + 1j * self.rng.standard_normal(spec.shape)) * sc.noise_lsb / math.sqrt(2)
        cube_lsb = spec + noise.astype(np.complex64)
        cube = (cube_lsb * SCALE).astype(np.complex64)
        truth = {**tr, "visible": visible}
        return cube, truth

    def next_packets(self) -> tuple[bytes, dict]:
        cube, truth = self.cube_at(self.t)
        words = iq_to_words(cube, self.cfg)
        pk = build_cube_packets(self.frame_index, words, max_payload=12 + device_piece_bytes(words.size * 4))
        truth["frame_index"] = self.frame_index
        self.frame_index += 1
        self.t += self.cfg.frame_period_ms / 1000.0
        return pk, truth


class SimSerial:
    """Fake pair of serial ports (control + data) backed by :class:`Simulator`.

    The control side answers every known `mmwc` command with ``Command Success`` (and a few
    with canned read-back values), unknown commands with ``Command Failed``. The data side
    streams packets in real time once ``mmwc start`` was received.
    """

    KNOWN = {"", "stop", "start", "open", "close", "ps", "mode", "range", "veloc", "freq", "chirp_num", "intv",
             "frame", "rx_gain", "hpf", "tx_rf_att", "clutter_rm", "dc_suppression", "win", "dop_fft", "report",
             "baudrate", "uart", "version", "state", "chirp_info", "fft_len", "getreportcfg", "noise", "ant",
             "cfar", "mount", "clutter_halt", "comtype"}

    def __init__(self, sim: Simulator, fail_commands: set[str] | None = None, route: str = "data"):
        """``route``: where the fake chip sends cube data: "data" (normal), "ctrl" (wrong UART) or "none"."""
        self.route = route
        self.sim = sim
        self.fail_commands = fail_commands or set()
        self.ctrl = _FakePort(self._on_ctrl_line, name="SIM-CTRL")
        self.data = _FakePort(None, name="SIM-DATA", baudrate=sim.cfg.baudrate)
        self.truth_log: dict[int, dict] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _on_ctrl_line(self, line: str) -> bytes:
        parts = line.strip().split()
        cmd = parts[1] if len(parts) > 1 else ""
        body = "mmwc " + " ".join(parts[1:]) + "\r\n\r\n"
        if cmd not in self.KNOWN or line.strip() in self.fail_commands or cmd in self.fail_commands:
            body += "Command Failed\r\n"
        elif cmd == "" :
            body = "mmwc\r\ncommand group help of mmwc (SIMULATOR)\r\n"
        elif cmd == "version":
            body += "SIM RS6130 fw 0.0\r\nCommand Success\r\n"
        elif cmd == "report" and (len(parts) == 2 or parts[2] == "get"):
            body += "gui report=cube, num=4294967295\r\nCommand Success\r\n"
        elif cmd == "dop_fft" and len(parts) == 2:
            body += f"dop fft={int(bool(self.sim.cfg.dop_fft) and self.sim.cfg.fft_mode == 2)}\r\nCommand Success\r\n"
        elif cmd == "fft_len":
            d = self.sim.d
            body += f"range fft {d.range_fft_len} doppler fft {d.doppler_len}\r\nCommand Success\r\n"
        else:
            body += "Command Success\r\n"
        if cmd == "start":
            self._start_stream()
        elif cmd == "stop":
            self._stop_stream()
        return (body + "\r\npsic:/$").encode()

    def _start_stream(self):
        if self._running:
            return
        self._running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._stream, name="sim-stream", daemon=True)
        self._thread.start()

    def _stop_stream(self):
        self._running = False
        self._stop.set()

    def _stream(self):
        period = self.sim.cfg.frame_period_ms / 1000.0
        nxt = time.monotonic()
        while not self._stop.is_set():
            pk, truth = self.sim.next_packets()
            self.truth_log[truth["frame_index"]] = truth
            if len(self.truth_log) > 2000:
                self.truth_log.pop(next(iter(self.truth_log)))
            port = {"data": self.data, "ctrl": self.ctrl}.get(self.route)
            if port is not None:
                for i in range(0, len(pk), 512):          # emulate UART chunking
                    port.push(pk[i:i + 512])
            nxt += period
            dt = nxt - time.monotonic()
            if dt > 0:
                self._stop.wait(dt)
            else:
                nxt = time.monotonic()

    def close(self):
        self._stop_stream()


class _FakePort:
    def __init__(self, on_line, name: str, baudrate: int = 115200):
        self._on_line = on_line
        self.name = name
        self.port = name
        self.baudrate = baudrate
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._line = b""
        self.is_open = True

    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._buf)

    def push(self, data: bytes):
        with self._lock:
            self._buf.extend(data)

    def read(self, n: int = 1) -> bytes:
        with self._lock:
            out = bytes(self._buf[:n])
            del self._buf[:n]
        if not out:
            time.sleep(0.002)
        return out

    def write(self, data: bytes) -> int:
        if self._on_line is None:
            return len(data)
        self._line += data
        while b"\n" in self._line:
            line, self._line = self._line.split(b"\n", 1)
            self.push(self._on_line(line.decode(errors="replace")))
        return len(data)

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        with self._lock:
            self._buf.clear()
