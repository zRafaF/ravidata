import logging
import time

import numpy as np
import pytest

from ravidata import hif
from ravidata.config import RadarConfig, has_errors, presets
from ravidata.cube import CubeShape, decode, expected_shape, infer_shape, iq_to_words, words_to_iq
from ravidata.logsetup import CollapsingHandler


def test_hif_roundtrip_fragmented_and_noisy():
    rng = np.random.default_rng(1)
    words = rng.integers(0, 2**32, 3 * 16 * 64, dtype=np.uint64).astype(np.uint32)
    pk = hif.build_cube_packets(7, words)
    assert pk.count(bytes([hif.MAGIC])) >= 4
    stream = b"\x00garbage\xa5\x01" + pk + hif.build_cube_packets(8, words)
    p = hif.HifParser()
    frames = []
    for i in range(0, len(stream), 333):          # arbitrary chunking
        frames += p.feed(stream[i:i + 333])
    assert [f.frame_index for f in frames] == [7, 8]
    assert np.array_equal(frames[0].words, words)
    assert p.stats.resyncs >= 1


def test_hif_detects_corruption():
    words = np.arange(96, dtype=np.uint32)
    pk = bytearray(hif.build_cube_packets(1, words))
    pk[30] ^= 0xFF                                # corrupt payload -> check32 fails
    p = hif.HifParser()
    assert p.feed(bytes(pk)) == []
    assert p.stats.errors["check32"] == 1


@pytest.mark.parametrize("iq", ["imag_low", "real_low"])
@pytest.mark.parametrize("order", ["tx_rx_dop_range", "dop_tx_rx_range"])
def test_cube_encode_decode(iq, order):
    cfg = RadarConfig(iq_order=iq, cube_order=order)
    rng = np.random.default_rng(0)
    cube = ((rng.standard_normal((4, 3, 32)) + 1j * rng.standard_normal((4, 3, 32))) * 0.01).astype(np.complex64)
    w = iq_to_words(cube, cfg)
    back = decode(w, CubeShape(4, 3, 32), cfg)
    assert np.allclose(back, cube, atol=2 / 32768)


def test_iq_order_convention():
    w = np.array([(100 << 16) | (0xFFFF & -5)], dtype=np.uint32)   # high=100, low=-5
    assert words_to_iq(w, "imag_low")[0] == pytest.approx((100 - 5j) / 32768)
    assert words_to_iq(w, "real_low")[0] == pytest.approx((-5 + 100j) / 32768)


def test_legacy_presets_fail_validation_and_new_ones_pass():
    P = presets()
    if "legacy-v3" in P:                     # private presets (not in the public repo)
        assert has_errors(P["legacy-v3"].validate())
        assert has_errors(P["legacy-perception2"].validate())
    for name in ("coarse-1d-20cm", "coarse-1d-30cm", "legal-1d-40mm", "legal-1d-30mm", "doppler-2d-coarse",
                 "doppler-2d-80mm", "slowtime-2d-coarse", "doppler-2d-coarse-2mbaud"):
        assert not has_errors(P[name].validate()), name


def test_link_budget_error():
    cfg = presets()["doppler-2d-80mm"].copy(frame_period_ms=40)
    codes = {i.code for i in cfg.validate()}
    assert "link" in codes


def test_derived_numbers():
    d = presets()["legal-1d-40mm"].derive()
    assert d.n_samples == 128 and d.range_fft_len == 128
    assert d.slope_mhz_per_us == pytest.approx(292.7, rel=1e-2)
    assert d.cube_bytes == 128 * 3 * 4


def test_shape_inference():
    cfg = presets()["legal-1d-40mm"]
    exp = expected_shape(cfg)
    s = infer_shape(3 * 256, cfg, exp)
    assert s.inferred and s.n_range == 256
    cfg2 = presets()["doppler-2d-80mm"]
    s = infer_shape(3 * 64 * 8, cfg2, expected_shape(cfg2))
    assert s.inferred and s.n_dop == 8 and s.n_range == 64


def test_collapsing_handler():
    recs = []

    class Sink(logging.Handler):
        def emit(self, r):
            recs.append(r.getMessage())

    lg = logging.getLogger("ravidata.test_collapse")
    lg.propagate = False
    h = CollapsingHandler(Sink(), burst=2, window_s=100)
    lg.addHandler(h)
    for i in range(50):
        lg.warning("crc error %d", i)
    lg.warning("different")
    h.flush_suppressed(force=True)
    lg.removeHandler(h)
    assert recs[:2] == ["crc error 0", "crc error 1"]
    assert "different" in recs
    assert any("48 more" in r for r in recs)


def test_sim_engine_end_to_end(tmp_path):
    from ravidata.engine import Engine
    from ravidata.sources import open_recording
    eng = Engine(runs_dir=tmp_path)
    eng.start_sim(presets()["legal-1d-40mm"].copy(frame_period_ms=20), "static-target-1m", record=True,
                  fail_commands={"hpf"})
    t0 = time.time()
    while eng.frame_counter < 40 and time.time() - t0 < 20:
        time.sleep(0.05)
    res = eng.latest
    d = eng.session.dir
    eng.stop()
    assert res is not None
    assert res.outputs["cfar"].nearest is not None
    assert abs(res.outputs["cfar"].nearest.range_m - 1.0) < 0.1
    rep = (d / "SESSION_REPORT.md").read_text()
    assert "Verdict" in rep and "summary.log" in rep
    rec = open_recording(d)
    assert rec.n >= 40 and rec.get_words(0) is not None
    assert rec.get_cube(0).shape == (1, 3, 128)


def test_sim_reports_rejections_and_shape_mismatch(tmp_path):
    from ravidata.engine import Engine
    eng = Engine(runs_dir=tmp_path)
    cfg = presets()["legal-1d-40mm"].copy(hpf_code=16, frame_period_ms=20)
    eng.start_sim(cfg, "static-target-1m", fail_commands={"hpf"}, chip_overrides={"range_res_mm": 20})
    t0 = time.time()
    while eng.frame_counter < 20 and time.time() - t0 < 20:
        time.sleep(0.05)
    d = eng.session.dir
    eng.stop()
    rep = (d / "SESSION_REPORT.md").read_text()
    assert "rejected by the chip" in rep and "hpf 255 16" in rep
    assert "Cube size mismatch" in rep
    summ = (d / "summary.log").read_text()
    assert "cube size mismatch" in summ
    led = (tmp_path / "config_ledger.jsonl").read_text()
    assert "hpf 255 16" in led


def test_readback_parse_and_compare():
    from ravidata.readback import compare, parse_all
    q = {
        "fft_len": "Range FFT: 32, Doppler FFT: 1   Command Success",
        "range": "range=6400mm, resol=200mm   Command Success",
        "freq": "FreqStart=58000MHz, Max=70000MHz   Command Success",
        "frame": "period=25ms, frame_num=0   Command Success",
        "chirp_info": "Chirp: start_freq=58000MHz end_freq=59605MHz slope=234MHz/us sweep_BW=1605MHz  valid_start=58738MHz "
                      "valid_BW=749MHz sample_time=3us  ADC num=32 rame_time=6us period=14248ns   Command Success",
        "getreportcfg": "total_size=96 split_size=3072 split_num=1   Command Success",
    }
    rb = parse_all(q)
    assert rb["chirp_info"]["adc_num"] == 32 and rb["fft_len"]["range_fft"] == 32
    checks = compare(presets()["coarse-1d-20cm"], rb)
    assert checks and all(c.ok for c in checks), [c for c in checks if not c.ok]
    bad = compare(presets()["legal-1d-40mm"], rb)
    assert any(not c.ok and c.item == "range FFT length" for c in bad)


@pytest.mark.parametrize("route,expect", [("ctrl", "CONTROL port"), ("none", "no bytes on data port")])
def test_link_diagnostics(tmp_path, route, expect):
    from ravidata.engine import Engine
    eng = Engine(runs_dir=tmp_path)
    eng.start_sim(presets()["coarse-1d-20cm"], "static-target-1m", route=route)
    time.sleep(5.5)
    d = eng.session.dir
    eng.stop()
    summ = (d / "summary.log").read_text(encoding="utf-8")
    assert expect in summ, summ[-2000:]



def _vendor_check32_ok(data: bytes, ck: bytes) -> bool:
    """Independent re-implementation of the vendor HostSample hif_check32_ok (header[2:6] + payload)."""
    import struct as _s
    total = 0
    for i in range(0, len(data) - 3, 4):
        total = (total + _s.unpack_from("<I", data, i)[0]) & 0xFFFFFFFF
    rem = len(data) % 4
    if rem:
        total = (total + _s.unpack("<I", data[-rem:] + b"\x00" * (4 - rem))[0]) & 0xFFFFFFFF
    return ((~total) & 0xFFFFFFFF) == _s.unpack("<I", ck)[0]


def test_check32_matches_vendor_definition():
    words = np.arange(96, dtype=np.uint32) * 12345
    pk = hif.build_cube_packets(3, words)
    L = len(pk)
    assert _vendor_check32_ok(pk[2:L - 4], pk[L - 4:])
    # vendor POLL frame from HostSample build_poll(): A5 ?? 15 0C 04 00 | 01 00 00 01 | ck32
    hdr = bytes([0x15, 0x0C, 4, 0])
    pl = bytes([1, 0, 0, 1])
    ck = hif.check32(hdr + pl)
    import struct as _s
    assert _vendor_check32_ok(hdr + pl, _s.pack("<I", ck))


def test_parser_learns_constant_check32_offset():
    import struct as _s
    words = np.arange(96, dtype=np.uint32)
    out = b""
    for k in range(8):
        pk = bytearray(hif.build_cube_packets(k, words))
        c = _s.unpack_from("<I", pk, len(pk) - 4)[0]
        _s.pack_into("<I", pk, len(pk) - 4, (c - 5) & 0xFFFFFFFF)   # firmware quirk: off by 5
        out += bytes(pk)
    p = hif.HifParser()
    fr = p.feed(out)
    assert p.learned_c32_offset == 5 and len(fr) >= 2


def test_dop_fft_command_is_numeric_and_read_back():
    """EXP-003: the shell parses atoi(argv[1]), so 'on' would mean OFF."""
    from ravidata.readback import compare, parse_all
    c2 = presets()["doppler-2d-coarse"]
    assert "dop_fft 1" in c2.to_commands() and not any(c.startswith("dop_fft on") for c in c2.to_commands())
    assert "dop_fft 0" in presets()["coarse-1d-20cm"].to_commands()
    assert "dop_fft" in c2.QUERY_COMMANDS and "dop_fft get" not in c2.QUERY_COMMANDS
    ok = compare(c2, parse_all({"dop_fft": "dop fft=1\r\nCommand Success"}))
    assert ok and all(c.ok for c in ok)
    bad = compare(c2, parse_all({"dop_fft": "dop fft=0\r\nCommand Success"}))
    assert any(not c.ok and c.level == "error" for c in bad)


def test_slow_time_dc_and_mti_reduction():
    """With the chip Doppler FFT off, 'dc' keeps the static part and 'no_dc' removes it."""
    from ravidata.frames import Frame
    from ravidata.processing.pipeline import Preprocess
    cfg = presets()["doppler-2d-coarse"].copy(dop_fft=False)
    nd, R = 16, 32
    cube = np.zeros((nd, 3, R), np.complex64)
    cube[:, :, 5] = 0.1                                               # static at bin 5
    cube[:, :, 9] = 0.1 * np.exp(1j * 2 * np.pi * 0.25 * np.arange(nd))[:, None]  # moving at bin 9
    fr = Frame(seq=0, frame_index=0, t_host=0.0, t_mono=0.0, cube=cube, shape=CubeShape(nd, 3, R))
    for mode, strong, weak in (("dc", 5, 9), ("no_dc", 9, 5)):
        pre = Preprocess(cfg, leak_mode="off", combine="rx1", dop_reduce=mode)
        f = pre.process(fr)
        assert f.p_db[strong] > f.p_db[weak] + 30, (mode, f.p_db[[5, 9]])
