"""Parse the chip's read-back answers and compare them with what the config predicts.

Answer formats observed on firmware ``MMW version:V4.3.1.0`` (2026-09-25, first bench run)::

    fft_len      Range FFT: 32, Doppler FFT: 1
    range        range=6400mm, resol=200mm
    freq         FreqStart=58000MHz, Max=70000MHz
    frame        period=25ms, frame_num=0
    intv         period=396us, intv_num=0
    chirp_info   Chirp: start_freq=58000MHz end_freq=59605MHz slope=234MHz/us sweep_BW=1605MHz
                 valid_start=58738MHz valid_BW=749MHz sample_time=3us ADC num=32 rame_time=6us period=14248ns
    getreportcfg total_size=96 split_size=3072 split_num=1
    state        Active, frame_sent=636
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .config import F_MAX_MHZ, RadarConfig
from .logsetup import KEEP, get_logger

log = get_logger("readback")

_KV = re.compile(r"([A-Za-z][A-Za-z_ ]*?)\s*[=:]\s*(-?\d+(?:\.\d+)?)([A-Za-z/%]*)")


def parse_answer(text: str) -> dict[str, float]:
    out = {}
    for k, v, unit in _KV.findall(text or ""):
        key = re.sub(r"\s+", "_", k.strip().lower())
        out[key] = float(v)
        if unit:
            out[key + "__unit"] = unit  # type: ignore[assignment]
    return out


def parse_all(queries: dict[str, str]) -> dict[str, dict]:
    return {q: parse_answer(a) for q, a in queries.items()}


@dataclass
class Check:
    item: str
    expected: object
    chip: object
    ok: bool
    level: str = "info"     # info | warning | error
    note: str = ""

    def to_dict(self):
        return asdict(self)


def compare(cfg: RadarConfig, rb: dict[str, dict]) -> list[Check]:
    d = cfg.derive()
    C: list[Check] = []

    def chk(item, exp, got, tol=0.0, level="warning", note=""):
        if got is None:
            return
        ok = abs(float(got) - float(exp)) <= tol * max(1.0, abs(float(exp))) if tol else float(got) == float(exp)
        C.append(Check(item, exp, got, ok, "info" if ok else level, note))

    f = rb.get("fft_len", {})
    chk("range FFT length", d.range_fft_len, f.get("range_fft"), level="error",
        note="cube shape depends on it")
    chk("Doppler FFT length", d.doppler_len, f.get("doppler_fft"), level="error")
    dq = rb.get("dop_fft", {})
    chk("Doppler FFT on chip (1=on)", int(cfg.fft_mode == 2 and bool(cfg.dop_fft)), dq.get("dop_fft"), level="error",
        note="0 = the cube's 2nd axis is slow time (intervals), 1 = Doppler bins")
    r = rb.get("range", {})
    chk("range_max_mm", cfg.range_max_mm, r.get("range"))
    chk("range_res_mm", cfg.range_res_mm, r.get("resol"))
    fq = rb.get("freq", {})
    chk("freq start MHz", cfg.freq_start_mhz, fq.get("freqstart"))
    fr = rb.get("frame", {})
    chk("frame period ms", cfg.frame_period_ms, fr.get("period"))
    ci = rb.get("chirp_info", {})
    chk("ADC samples", d.n_samples, ci.get("adc_num"), tol=1.01 / max(1, d.n_samples), level="warning",
        note="the chip reports N-1 now and then (127 for 128: 2026-09-25, 2026-11-13), accepted; the cube size "
             "(report total_size) is what matters")
    chk("slope MHz/us", round(d.slope_mhz_per_us), ci.get("slope"), tol=0.03)
    chk("valid bandwidth MHz", round(d.bandwidth_hz / 1e6), ci.get("valid_bw"), tol=0.03)
    if ci.get("end_freq") is not None:
        end = ci["end_freq"]
        C.append(Check("sweep end MHz <= 64000", f"<= {F_MAX_MHZ}", end, end <= F_MAX_MHZ,
                       "info" if end <= F_MAX_MHZ else "error", "whole sweep incl. settling, from chirp_info"))
        chk("sweep end MHz (model)", round(d.sweep_end_mhz), end, tol=0.01, level="info",
            note="config.py chirp model vs chip")
    rc = rb.get("getreportcfg", {})
    chk("report total_size (dwords)", d.cube_dwords, rc.get("total_size"), level="error",
        note="what the chip will send per frame")
    return C


def log_checks(checks: list[Check]):
    bad = [c for c in checks if not c.ok]
    for c in checks:
        fn = {"info": log.info, "warning": log.warning, "error": log.error}[c.level]
        fn("read-back check %-28s expected %-10s chip %-10s %s %s", c.item, c.expected, c.chip,
           "OK" if c.ok else "MISMATCH", c.note, extra=KEEP)
    if not checks:
        log.warning("no read-back values could be parsed (unknown answer format?)", extra=KEEP)
    elif not bad:
        log.info("read-back: all %d checks match the config", len(checks), extra=KEEP)
    return bad
