"""Command line entry point: ``uv run ravidata <command>``.

Commands::

    gui                      open the lab GUI (default)
    ports                    list serial ports
    check-config [NAME|FILE] print derived values and validation for presets or a JSON config
    sim                      headless simulator run (writes a run dir + report)
    live --ctrl COMx --data COMy   headless live run with the radar module
    replay PATH              re-run all detectors over a recording (as fast as possible)
    report [RUN_DIR]         print SESSION_REPORT.md (default: latest run)
    ledger [-n N]            table of the last N runs: config, rejected commands, fps, shape
    bringup --ctrl COMx --data COMy   try a matrix of link configurations back to back (which one streams?)
    live --link spi --ctrl COMx       live run with the datacube over SPI (CH347 bridge, P1812; EXP-002)
    spi-probe --ctrl COMx             step-by-step SPI bring-up report (bridge, GPIO/INT, POLL, frames)
    linkbench --link uart|spi ...     sweep cube size and frame rate; measure delivered fps / chip frame loss
    compare [RUN ...] [--detectors a,b]  replay recordings through detectors; proxy metrics table (+ markers)
    score [RUN ...]          GT-001 ground-truth takes: real detection / false-alarm leaderboard of all detectors
    ab [RUN ...] [--a X --b Y]        blind A/B review GUI of two detectors on the same recording
    ab-report                         tally of the A/B votes (runs/AB_VOTES.jsonl)
    inject-eval [RUN ...] [--detectors a,b]  score detectors on synthetic targets injected into real recordings
    tune --detector NAME [RUN ...] [--trials N]  Optuna parameter search on injected targets (train/test split)
    phase-check RUN                   interferometry calibration from a run with "phase: ..." markers
    collect --ctrl COMx --env closed|open   TCC-001 scripted recorder (text only, no processing; docs/protocols/TCC-001-collection.md)
    rec [--sim]              lean recorder GUI: config, stream, record, sanity plots, dropped frames (no detectors)
    align                    viewer: phone video + IMU (3D attitude) + radar on one clock; manual fix; export pack
    pack [--video V] [--imu Z] [--folder F] [--video-ms MS --imu-ms MS]  headless auto-align + export one .tccpack.h5
    print-targets [--out DIR]  AprilTag + checkerboard A4 PDFs and tags.json (docs/protocols/targets)
    calib VIDEO [--step N] [--model k1k2]  camera intrinsics from a checkerboard video (<video>.calib.json)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def _cfg_from(arg: str | None):
    from .config import RadarConfig, presets
    if arg is None:
        return presets()["coarse-1d-20cm"]
    if arg in presets():
        return presets()[arg]
    p = Path(arg)
    if p.exists():
        return RadarConfig.load(p)
    raise SystemExit(f"unknown preset or file: {arg}. Presets: {', '.join(presets())}")


def _wait(engine, seconds: float, until_finished: bool = False):
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            time.sleep(0.2)
            if until_finished and engine.state in ("finished", "error", "stopped"):
                break
            if engine.state == "error":
                break
    except KeyboardInterrupt:
        print("interrupted")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ravidata", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("gui")
    rc = sub.add_parser("rec", help="lean recorder GUI (config, record, sanity plots, link health)")
    rc.add_argument("--sim", action="store_true", help="start with the simulator selected")
    sub.add_parser("align", help="video + IMU + radar alignment viewer and pack export")
    pk = sub.add_parser("pack", help="auto-align a phone recording with the radar takes and write one .tccpack.h5")
    pk.add_argument("--folder", help="folder with the phone video and the Sensor Logger zip")
    pk.add_argument("--video", help="phone video file")
    pk.add_argument("--imu", help="Sensor Logger zip")
    pk.add_argument("--out", help="output file (default runs/packs/<first take>.tccpack.h5)")
    pk.add_argument("--no-video", action="store_true", help="do not embed the video in the pack")
    pk.add_argument("--video-ms", type=float, default=0.0,
                    help="manual video offset, ms (positive = video later on the radar clock), as in `align`")
    pk.add_argument("--imu-ms", type=float, default=0.0, help="manual IMU offset, ms (same convention)")
    pt = sub.add_parser("print-targets", help="AprilTag + checkerboard A4 PDFs to print")
    pt.add_argument("--out", default=None, help="output folder (default docs/protocols/targets)")
    cb = sub.add_parser("calib", help="camera intrinsics from a checkerboard video -> <video>.calib.json")
    cb.add_argument("video", help="calibration video, e.g. runs/recs/calib/<phone>_<date>.mp4")
    cb.add_argument("--step", type=int, default=None, help="use every N-th frame (default: ~6 per second)")
    cb.add_argument("--model", default="k1k2", choices=["k1k2", "k1k2p", "k1k2k3p"], help="distortion model")
    cb.add_argument("--settings", default="", help="camera settings to store with the result (free text)")
    cb.add_argument("--out", default=None, help="output JSON (default: beside the video)")
    sub.add_parser("ports")
    c = sub.add_parser("check-config")
    c.add_argument("config", nargs="?", help="preset name or JSON file (default: all presets)")
    s = sub.add_parser("sim")
    s.add_argument("--preset", default="coarse-1d-20cm")
    s.add_argument("--scene", default="branch+pole")
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--record", action="store_true")
    lv = sub.add_parser("live")
    lv.add_argument("--ctrl", required=True, help="control port (mmwc), e.g. COM8")
    lv.add_argument("--data", help="data port (UART link), e.g. COM7")
    lv.add_argument("--link", choices=["uart", "spi"], default="uart")
    lv.add_argument("--spi-dev", type=int, default=0)
    lv.add_argument("--spi-clock", type=float, default=30e6)
    lv.add_argument("--int-bit", default="auto")
    lv.add_argument("--ch347-dll", default=None)
    lv.add_argument("--preset", default="coarse-1d-20cm", help="preset name or JSON file")
    lv.add_argument("--seconds", type=float, default=30)
    lv.add_argument("--no-record", action="store_true")
    lv.add_argument("--no-process", action="store_true", help="record only: no detectors (slow laptops)")
    co = sub.add_parser("collect", help="TCC-001 scripted recorder (docs/protocols/TCC-001-collection.md)")
    co.add_argument("--env", choices=["closed", "open"], required=True, help="closed = indoor room, open = outdoors")
    co.add_argument("--ctrl", help="control port, e.g. COM10")
    co.add_argument("--data", help="data port (UART link only)")
    co.add_argument("--link", choices=["uart", "spi"], default="spi")
    co.add_argument("--spi-dev", type=int, default=0)
    co.add_argument("--int-bit", default="auto")
    co.add_argument("--configs", default="", help="comma list of presets (default: tcc-1d-40mm,tcc-2d-40mm)")
    co.add_argument("--only", default="", help="comma list of tests to run, e.g. T1,T2")
    co.add_argument("--start", default="", help="resume at this step id (see --list)")
    co.add_argument("--seconds", type=float, default=0, help="override the seconds per segment")
    co.add_argument("--t3", action="store_true", help="include T3 (angular separation) indoors too")
    co.add_argument("--bonus", action="store_true", help="include T4 (head-height board)")
    co.add_argument("--walk", action="store_true", help="include T5 (rig carried: walk toward wall/panel/can)")
    co.add_argument("--list", action="store_true", help="print the steps and exit")
    co.add_argument("--sim", action="store_true", help="dry run with the simulator")
    co.add_argument("--countdown", type=int, default=10, help="seconds to walk to the start position (moving steps)")
    rp = sub.add_parser("replay")
    rp.add_argument("path")
    rp.add_argument("--speed", type=float, default=1000.0, help="1 = real time")
    bu = sub.add_parser("bringup")
    bu.add_argument("--ctrl", required=True)
    bu.add_argument("--data", required=True)
    bu.add_argument("--seconds", type=float, default=5.0, help="per variant")
    bu.add_argument("--quick", action="store_true", help="only the 4 most informative variants")
    sp = sub.add_parser("spi-probe")
    from .cli_spi import add_spi_args
    add_spi_args(sp, need_ctrl=False)
    sp.add_argument("--preset", default="coarse-1d-20cm")
    sp.add_argument("--seconds", type=float, default=10.0, help="read time per mode")
    sp.add_argument("--watch", type=float, default=2.0, help="GPIO watch time while streaming")
    sp.add_argument("--sim", action="store_true", help="run against the simulated SPI device")
    lb = sub.add_parser("linkbench")
    add_spi_args(lb, need_ctrl=False)
    lb.add_argument("--link", choices=["uart", "spi"], required=True)
    lb.add_argument("--data", help="data port (UART link)")
    lb.add_argument("--baud", type=int, default=1_000_000)
    lb.add_argument("--seconds", type=float, default=6.0)
    lb.add_argument("--quick", action="store_true")
    lb.add_argument("--sim", action="store_true")
    lg = sub.add_parser("ledger")
    lg.add_argument("-n", type=int, default=20)
    r = sub.add_parser("report")
    r.add_argument("run_dir", nargs="?")
    cp = sub.add_parser("compare")
    cp.add_argument("runs", nargs="*", help="run dirs or names under runs/ (default: the 6 newest with cubes.h5)")
    cp.add_argument("--detectors", default="", help="comma list of registered detectors (default: all)")
    cp.add_argument("--min-r", type=float, default=0.25)
    cp.add_argument("--max-r", type=float, default=3.0)
    ab = sub.add_parser("ab")
    ab.add_argument("runs", nargs="*")
    ab.add_argument("--a", default="track")
    ab.add_argument("--b", default="cfar")
    ab.add_argument("--segment", type=float, default=10.0)
    ab.add_argument("--no-blind", action="store_true")
    sub.add_parser("ab-report")
    scp = sub.add_parser("score")
    scp.add_argument("runs", nargs="*", help="run dirs/names (default: every run under runs/ with GT-001 takes)")
    scp.add_argument("--detectors", default="", help="comma list of registered detectors (default: all)")
    scp.add_argument("--tol-walk", type=float, default=0.25)
    scp.add_argument("--warn-min", type=float, default=1.0)
    scp.add_argument("--alarm-max", type=float, default=2.5)
    pc = sub.add_parser("phase-check")
    pc.add_argument("run")
    ie = sub.add_parser("inject-eval")
    ie.add_argument("runs", nargs="*")
    ie.add_argument("--detectors", default="")
    ie.add_argument("--scenarios", type=int, default=8)
    ie.add_argument("--snr", default="25,18", help="per-cell SNR at 1 m, dB (comma list, cycled)")
    ie.add_argument("--seed", type=int, default=0)
    ie.add_argument("--mix", action="store_true", help="EXP-009 scenario mix (point / small near / extended / static)")
    ie.add_argument("--wall-ref", action="store_true", help="EXP-011: SNRs relative to each run's wall")
    ie.add_argument("--emulate", default="", help="EXP-013: comma list of emulated chip configs, e.g. "
                    "'orig,56mm,100mm,K16,fps/2,150mm+K16' (coarser than the recording only)")
    ie.add_argument("--tol-m", type=float, default=None, help="fixed hit tolerance (use with --emulate)")
    cg = sub.add_parser("config-grid", help="EXP-013: list chip configs that validate and fit the SPI link")
    cg.add_argument("--mode", type=int, default=2)
    cg.add_argument("--max-util", type=float, default=0.6)
    tu = sub.add_parser("tune")
    tu.add_argument("runs", nargs="*")
    tu.add_argument("--detector", required=True)
    tu.add_argument("--trials", type=int, default=60)
    tu.add_argument("--scenarios", type=int, default=8, help="per recording, for train and for test")
    tu.add_argument("--clutter-weight", type=float, default=0.25)
    tu.add_argument("--seed", type=int, default=0)
    tu.add_argument("--wall-ref", action="store_true",
                    help="EXP-011: injected SNRs relative to each run's strongest return at 1.4-2.1 m (compare configs)")
    tu.add_argument("--pre", default="{}", help="JSON Preprocess overrides applied to every trial")
    a = ap.parse_args(argv)

    if a.cmd == "ab":
        from .gui.ab import run_ab
        return run_ab(a.runs or None, a.a, a.b, a.segment, not a.no_blind)

    if a.cmd == "rec":
        from .gui.recorder import run_recorder
        return run_recorder(sim=a.sim)

    if a.cmd == "align":
        from .gui.aligner import run_aligner
        return run_aligner()

    if a.cmd in (None, "gui"):
        from .gui.app import run_gui
        return run_gui()

    from .logsetup import setup_base
    from .session import default_runs_dir
    # collect: keep the console for the operator's prompts (full logs still go to each run's files)
    setup_base(logging.WARNING if a.cmd == "collect" else logging.INFO, app_log_dir=default_runs_dir())

    if a.cmd == "print-targets":
        from .session import repo_root
        from .targets import write_all
        for p_ in write_all(Path(a.out) if a.out else repo_root() / "docs" / "protocols" / "targets"):
            print(p_)
        return 0

    if a.cmd == "calib":
        from .calib import fmt, run
        res = run(a.video, a.out, a.step, a.model, a.settings,
                  progress=lambda f: print(f"\r  decoding {100 * f:3.0f} %", end="", flush=True))
        print()
        print(fmt(res))
        print(res["_json"])
        return 0

    if a.cmd == "pack":
        from .align import find_phone_files, fmt_alignment, headless_pack
        video, imu = (Path(a.video) if a.video else None), (Path(a.imu) if a.imu else None)
        if a.folder:
            fv, fz = find_phone_files(a.folder)
            video, imu = video or fv, imu or fz
        if not video and not imu:
            raise SystemExit("give --folder, or --video and/or --imu")
        out, al, takes = headless_pack(video, imu, default_runs_dir(), Path(a.out) if a.out else None,
                                       embed_video=not a.no_video, manual_video_ms=a.video_ms,
                                       manual_imu_ms=a.imu_ms)
        print(fmt_alignment(al))
        print(f"takes: {', '.join(t.run_id for t in takes) or 'none'}")
        print(f"pack: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    if a.cmd == "ab-report":
        from .gui.ab import ab_report
        print(ab_report())
        return 0

    if a.cmd == "phase-check":
        from .phasecheck import phase_check
        from .session import default_runs_dir as _rd
        pth = Path(a.run) if Path(a.run).exists() else _rd() / a.run
        print(phase_check(pth))
        return 0

    if a.cmd == "config-grid":
        from .emulate import config_grid
        rows = config_grid(a.mode, max_util=a.max_util)
        print("| res mm | range max m | K | v_res m/s | fps | cube KB | link util | IF high-pass m | B GHz | warnings |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['res_mm']} | {r['rmax_m']:.2f} | {r['K']} | {r['v_res']:.3f} | {r['fps']:.1f} | {r['cube_kb']:.0f} | "
                  f"{r['util']:.2f} | {r['hpf_m']:.2f} | {r['B_ghz']:.2f} | {' '.join(r['warn'])} |")
        return 0

    if a.cmd in ("inject-eval", "tune"):
        from .session import default_runs_dir as _rd
        rd = _rd()
        runs = [Path(x) if Path(x).exists() else rd / x for x in a.runs] if a.runs else \
            [d for d in sorted(rd.iterdir(), reverse=True) if (d / "cubes.h5").exists()
             and (d / "cubes.h5").stat().st_size > 200_000][:3]
        if a.cmd == "tune":
            try:
                from .tune import tune
            except ImportError:
                raise SystemExit("tune is not part of this repository")
            tune(a.detector, runs, a.trials, a.scenarios, a.seed, a.clutter_weight, out_dir=rd,
                 wall_ref=a.wall_ref, base_pre=__import__("json").loads(a.pre))
            return 0
        from datetime import datetime
        from .inject import default_scenarios, run_injected, summarise
        from .processing.detectors import DETECTORS
        from .sources import open_recording
        from .emulate import parse as _parse_emu
        from .inject import scenario_mix, summarise_by_emulation, wall_relative
        names = [x.strip() for x in a.detectors.split(",") if x.strip()] or sorted(DETECTORS)
        snrs = tuple(float(x) for x in a.snr.split(","))
        emus = [_parse_emu(x) for x in a.emulate.split(",")] if a.emulate else [None]
        rows = []
        for r in runs:
            rec = open_recording(r)
            dur = float(rec.times[-1] - rec.times[0])
            sc = scenario_mix(dur, a.scenarios, a.seed) if a.mix else default_scenarios(dur, a.scenarios, a.seed, snrs)
            if a.wall_ref:
                sc = wall_relative(rec, sc)
            for e in emus:
                print(f"injecting {len(sc)} targets into {r.name} ({e.name() if e else 'orig'}) …", flush=True)
                rows += run_injected(r, sc, {n: {"kind": n} for n in names}, emulation=e, tol_m=a.tol_m)
        txt = "# Injected-target evaluation\n\n" + (summarise_by_emulation(rows) if (a.emulate or a.mix)
                                                     else summarise(rows))
        out = rd / f"INJECT_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(txt, encoding="utf-8")
        print(txt + f"\n\nwritten: {out}")
        return 0

    if a.cmd == "score":
        from datetime import datetime
        from .evaluate import load_markers
        from .groundtruth import ScoreCfg, report_md, score_runs
        from .processing.detectors import DETECTORS
        from .session import default_runs_dir as _rd
        rd = _rd()
        if a.runs:
            runs = [Path(x) if Path(x).exists() else rd / x for x in a.runs]
        else:
            runs = [d for d in sorted(rd.iterdir()) if d.is_dir() and (d / "cubes.h5").exists()
                    and any(str(m["text"]).lower().startswith("take start") for m in load_markers(d))]
        if not runs:
            raise SystemExit("no runs with GT-001 takes found (record with the GUI: Markers tab, F7/F8)")
        names = [x.strip() for x in a.detectors.split(",") if x.strip()] or None
        if names and [n for n in names if n not in DETECTORS]:
            raise SystemExit(f"unknown detectors; known: {sorted(DETECTORS)}")
        specs = {n: {"kind": n} for n in names} if names else None
        sc = ScoreCfg(tol_walk_m=a.tol_walk, warn_min_m=a.warn_min, alarm_max_m=a.alarm_max)
        print(f"scoring {len(runs)} runs: {', '.join(r.name for r in runs)}", flush=True)
        rows, takes = score_runs(runs, specs, sc)
        txt = report_md(rows, takes, sc)
        out = rd / f"SCORE_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(txt, encoding="utf-8")
        print(txt.split("## Per scene")[0])
        print(f"\nwritten: {out}")
        return 0

    if a.cmd == "compare":
        from datetime import datetime
        from .evaluate import run_offline, table
        from .processing.detectors import DETECTORS
        from .session import default_runs_dir as _rd
        rd = _rd()
        runs = [Path(x) if Path(x).exists() else rd / x for x in a.runs] if a.runs else \
            [d for d in sorted(rd.iterdir(), reverse=True) if (d / "cubes.h5").exists()
             and (d / "cubes.h5").stat().st_size > 50_000][:6]
        names = [x.strip() for x in a.detectors.split(",") if x.strip()] or None
        specs = {n: {"kind": n} for n in names} if names else None
        if names:
            bad = [n for n in names if n not in DETECTORS]
            if bad:
                raise SystemExit(f"unknown detectors {bad}; known: {sorted(DETECTORS)}")
        res = []
        for r in runs:
            print(f"replaying {r.name} …", flush=True)
            res.append(run_offline(r, specs, min_r=a.min_r, max_r=a.max_r))
        txt = ("# Detector comparison (proxy metrics, see ravidata/evaluate.py)\n" + table(res))
        out = rd / f"COMPARE_{datetime.now():%Y%m%d_%H%M%S}.md"
        out.write_text(txt, encoding="utf-8")
        print(txt)
        print(f"\nwritten: {out}")
        return 0

    if a.cmd == "ports":
        from serial.tools import list_ports
        for p in list_ports.comports():
            print(f"{p.device:10s} {p.description}  [{p.hwid}]")
        return 0

    if a.cmd == "check-config":
        from .config import presets
        cfgs = [_cfg_from(a.config)] if a.config else list(presets().values())
        for cfg in cfgs:
            print(f"== {cfg.name}: {cfg.notes}")
            for line in cfg.derive().summary_lines():
                print("   " + line)
            for i in cfg.validate():
                print("   * " + str(i))
            print("   commands: " + " | ".join(cfg.to_commands()))
        return 0

    if a.cmd == "report":
        runs = default_runs_dir()
        d = Path(a.run_dir) if a.run_dir else Path((runs / "LATEST.txt").read_text().strip())
        print((d / "SESSION_REPORT.md").read_text(encoding="utf-8"))
        return 0

    if a.cmd == "ledger":
        import json as _json
        p = default_runs_dir() / "config_ledger.jsonl"
        if not p.exists():
            print("no runs yet")
            return 0
        rows = [_json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()][-a.n:]
        for r_ in rows:
            c = r_["config"]
            print(f"{r_['run_id'][:15]} {r_['source']:6s} {r_['preset'][:18]:18s} mode{c['fft_mode']} "
                  f"range {c['range_max_mm']}/{c['range_res_mm']} chirps {c['chirp_num']} frame {c['frame_period_ms']}ms | "
                  f"boot {r_['boot_ok']} rejected {r_['rejected_cmds']} | {r_['frames']} fr "
                  f"{r_['fps_measured']}/{r_['fps_expected']} fps | shape {r_['shape_used']} | gaps {r_['index_gaps']} "
                  f"| err {r_['parse_errors']} | {','.join(v for v in r_['validation'] if v.startswith('error'))}")
        return 0

    if a.cmd == "bringup":
        return _bringup(a)
    if a.cmd == "spi-probe":
        from .cli_spi import spi_probe
        if not a.sim and not a.ctrl:
            raise SystemExit("--ctrl is required (or --sim)")
        return spi_probe(a)
    if a.cmd == "linkbench":
        from .cli_spi import linkbench
        if not a.sim and (not a.ctrl or (a.link == "uart" and not a.data)):
            raise SystemExit("--ctrl (and --data for the UART link) are required, or --sim")
        return linkbench(a)

    if a.cmd == "collect":
        from .collect import main_collect
        return main_collect(a)

    from .engine import Engine
    eng = Engine(process=not getattr(a, "no_process", False))
    if a.cmd == "sim":
        eng.start_sim(_cfg_from(a.preset), a.scene, record=a.record)
        _wait(eng, a.seconds)
    elif a.cmd == "live":
        cfg = _cfg_from(a.preset)
        if a.link == "spi":
            from .cli_spi import int_bit_arg
            eng.start_spi(cfg.copy(spi_clock_hz=int(a.spi_clock)), a.ctrl, a.spi_dev, int_bit_arg(a.int_bit),
                          record=not a.no_record, ch347_dll=a.ch347_dll)
        else:
            if not a.data:
                raise SystemExit("--data is required for the UART link")
            eng.start_serial(cfg.copy(link="uart"), a.ctrl, a.data, record=not a.no_record)
        _wait(eng, a.seconds)
    elif a.cmd == "replay":
        eng.start_replay(a.path, speed=a.speed, loop=False)
        _wait(eng, 3600, until_finished=True)
    d = eng.session.dir if eng.session else None
    eng.stop("cli finished")
    if d:
        print(f"\nrun dir: {d}\nreport:  {d / 'SESSION_REPORT.md'}")
    return 0


def bringup_variants(quick: bool = False):
    """(label, config) pairs, most informative first."""
    from .config import presets
    P = presets()
    base = P["coarse-1d-20cm"]
    V = [("legacy-perception2-exact", P.get("legacy-perception2")),   # known to have streamed in July 2026
         ("coarse1d-p2-1M", base.copy(cmd_order="p2", baudrate=1_000_000)),
         ("coarse1d-p2-2M", base.copy(cmd_order="p2", baudrate=2_000_000)),
         ("doppler2d-p2-2M", P["doppler-2d-coarse"].copy(cmd_order="p2"))]
    if not quick:
        V += [("coarse1d-v3-1M", base.copy(cmd_order="v3", baudrate=1_000_000)),
              ("coarse1d-bblast-1M", base.copy(cmd_order="baseband_last", baudrate=1_000_000)),
              ("coarse1d-p2-1M-noopen", base.copy(cmd_order="p2", baudrate=1_000_000, send_open=False)),
              ("legal40-p2-1M", P["legal-1d-40mm"].copy(cmd_order="p2", baudrate=1_000_000)),
              ("legacy-v3-exact", P.get("legacy-v3"))]
    return [(label, c) for label, c in V if c is not None]   # the legacy presets are private


def _bringup(a):
    """Run each variant for a few seconds, back to back, and summarise which ones deliver frames."""
    import json as _json
    from datetime import datetime
    from .engine import Engine
    from .session import default_runs_dir
    eng = Engine()
    rows = []
    for label, cfg in bringup_variants(a.quick):
        cfg = cfg.copy(name=cfg.name)
        print(f"--- {label}: {' | '.join(cfg.to_commands())}")
        eng.start_serial(cfg, a.ctrl, a.data, diagnose=False, label=f"bringup_{label}")
        t0 = time.monotonic()
        while time.monotonic() - t0 < a.seconds and eng.state != "error":
            time.sleep(0.2)
        src, ses = eng.source, eng.session
        rep = (ses.config_report or {}) if ses else {}
        ps = src.parser.stats if src is not None else None
        row = {"variant": label, "run_dir": str(ses.dir) if ses else None, "state": eng.state,
               "frames": src.stats.frames if src else 0, "fps": round(src.stats.fps(), 1) if src else 0,
               "bytes": ps.bytes_in if ps else 0, "packets": ps.packets_ok if ps else 0,
               "parse_errors": dict(ps.errors) if ps else {}, "rejected": rep.get("failed", []),
               "readback_mismatch": rep.get("check_mismatches", []),
               "comtype": (rep.get("queries") or {}).get("comtype", "")[:80]}
        eng.stop(f"bringup variant {label} done")
        rows.append(row)
        print(f"    -> frames {row['frames']} ({row['fps']} fps), bytes {row['bytes']}, packets {row['packets']}, "
              f"rejected {row['rejected']}, mismatches {row['readback_mismatch']}")
        time.sleep(0.5)
    out = default_runs_dir() / f"BRINGUP_{datetime.now():%Y%m%d_%H%M%S}.md"
    L = ["# Link bring-up matrix", "", f"ctrl {a.ctrl}, data {a.data}, {a.seconds:.0f} s per variant", "",
         "| variant | frames | fps | bytes on data port | HIF packets | rejected cmds | read-back mismatches | run |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['variant']} | {r['frames']} | {r['fps']} | {r['bytes']} | {r['packets']} | "
                 f"{', '.join(r['rejected']) or '-'} | {', '.join(r['readback_mismatch']) or '-'} | "
                 f"`{Path(r['run_dir']).name if r['run_dir'] else '-'}` |")
    ok = [r["variant"] for r in rows if r["frames"] > 0]
    L += ["", f"**Streaming variants:** {', '.join(ok) if ok else 'NONE'}", "", "```json",
          _json.dumps(rows, indent=1), "```"]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nstreaming variants: {ok or 'NONE'}\nsummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
