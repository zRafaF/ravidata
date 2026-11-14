"""`ravidata spi-probe` and `ravidata linkbench` (EXP-002: is the data rate bounded by the link or the chip?)."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

from .logsetup import get_logger

log = get_logger("cli")


def int_bit_arg(v: str):
    """--int-bit: 'auto' (detect), 'none' (blind polling) or a GPIO bit number 0..7."""
    return "auto" if v == "auto" else None if v == "none" else int(v)


def add_spi_args(p, need_ctrl: bool = True):
    p.add_argument("--ctrl", required=need_ctrl, help="control port (mmwc shell, UART0; single COM port with MODE SEL ON)")
    p.add_argument("--spi-dev", type=int, default=0, help="CH347 device index (default 0)")
    p.add_argument("--spi-clock", type=float, default=30e6, help="SPI clock in Hz (CH347: 60/30/15/7.5/3.75 MHz)")
    p.add_argument("--int-bit", default="auto", help="CH347 GPIO bit of the radar INT: auto | none | 0..7")
    p.add_argument("--ch347-dll", default=None, help="path to WCH CH347DLLA64.DLL if not on the DLL search path")
    p.add_argument("--gpio-input", default=None, help="probe only: set these CH347 GPIO bits to input first, e.g. 0xF8")


# ---------------------------------------------------------------------------------------- probe
def _open_bus(a, fake=None):
    if fake is not None:
        from .spi.simbus import SimSpiBus
        return SimSpiBus(fake, xfer_latency_s=0.0003)
    from .spi.ch347 import Ch347Bus
    return Ch347Bus(a.spi_dev, a.spi_clock, a.ch347_dll)


def spi_probe(a) -> int:
    """Step-by-step SPI bring-up with a readable report. Each step is independent so one run shows
    how far the link gets: bridge -> GPIO -> radar config -> INT line -> POLL/read -> frames."""
    from . import chip
    from .config import presets
    from .hif import HifParser
    from .session import default_runs_dir
    from .spi.link import HifSpiLink, build_poll, pick_int_bit, watch_gpio

    cfg = presets()[a.preset].copy(link="spi", spi_clock_hz=int(a.spi_clock))
    R: dict = {"started": datetime.now().isoformat(timespec="seconds"), "preset": cfg.name,
               "spi_clock_req_hz": a.spi_clock, "steps": []}
    step = lambda _step, **kw: (R["steps"].append({"step": _step, **kw}), print(f"[{_step}] {str(kw)[:300]}"))  # noqa: E731
    fake = None
    if a.sim:
        from .sim import SimSerial, Simulator, scene_presets
        fake = SimSerial(Simulator(cfg, scene_presets()["static-target-1m"]))
    bus = ctrl = None
    try:
        bus = _open_bus(a, fake)
        step("bridge", ok=True, name=bus.name)
        if a.gpio_input and hasattr(bus, "gpio_make_input"):
            step("gpio_make_input", mask=a.gpio_input, ok=bus.gpio_make_input(int(a.gpio_input, 0)))
        step("gpio_idle", **watch_gpio(bus, 0.5))
        if fake is not None:
            ctrl = fake.ctrl
        else:
            import serial
            ctrl = serial.Serial(a.ctrl, cfg.ctrl_baudrate, timeout=0.05)
        cc = chip.ControlChannel(ctrl)
        rep = chip.configure(cc, cfg)
        step("radar_config", boot_ok=rep.boot_ok, failed=[r.cmd for r in rep.failed],
             commands=[f"{r.cmd} -> {r.status}" for r in rep.results if r.phase == "config"],
             report_get=rep.queries.get("report get", "")[:80], version=rep.queries.get("version", "")[:60])
        st = cc.send("start", phase="start")
        step("start", status=st.status, response=st.response[:80])
        w = watch_gpio(bus, a.watch)
        ib = pick_int_bit(w) if a.int_bit == "auto" else (None if a.int_bit == "none" else int(a.int_bit))
        step("gpio_streaming", int_bit=ib, **w)
        if ib is None and a.int_bit == "auto" and hasattr(bus, "int_scan"):
            counts = bus.int_scan()
            best = max(counts, key=lambda k: counts[k]) if counts else None
            step("int_scan", note="CH347 interrupt callbacks per GPIO pin (both edges), radar streaming",
                 counts=counts, pick=best if best is not None and counts[best] >= 3 else None)
            if best is not None and counts[best] >= 3:
                ib = best
        tries = [ib] if ib is not None else []
        tries.append(None)
        for mode in tries:
            link = HifSpiLink(bus, int_bit=mode, burst=64)
            parser = HifParser()
            link.sync()
            frames = 0
            per_s = [0] * int(math.ceil(a.seconds))
            gaps = 0
            last_idx = None
            t0 = time.monotonic()
            t_end = t0 + a.seconds
            while time.monotonic() < t_end:
                pks = link.read_burst()
                if not pks:
                    time.sleep(0.0003 if mode is not None else 0.002)
                for pk in pks:
                    for f in parser.feed(pk):
                        frames += 1
                        per_s[min(len(per_s) - 1, int(time.monotonic() - t0))] += 1
                        if last_idx is not None and f.frame_index > last_idx + 1:
                            gaps += f.frame_index - last_idx - 1
                        last_idx = f.frame_index
            dur = a.seconds
            step("read_" + ("int" if mode is not None else "blind"), int_bit=mode, frames=frames,
                 fps=round(frames / dur, 1), chip_frames_lost=gaps, frames_per_second=per_s,
                 kbytes_per_s=round(parser.stats.bytes_in / dur / 1e3, 1),
                 parser=parser.stats.to_dict(), link=link.stats.to_dict(),
                 first_bytes=bytes(link.first_bytes[:64]).hex(" "), last_bytes=link.last_rx[-64:].hex(" "))
            if frames:
                break
        else:
            raw = [bus.xfer(b"\x00" * 32).hex(" ") for _ in range(3)]
            bus.xfer(build_poll(8))
            time.sleep(0.002)
            raw += [bus.xfer(b"\x00" * 32).hex(" ") for _ in range(3)]
            step("raw_dump", note="3 reads, POLL, 3 reads (32 B each)", reads=raw)
        cc.send("stop", phase="stop", timeout_s=1.0)
    except Exception as e:
        step("error", error=repr(e))
    finally:
        for x in (bus, ctrl):
            try:
                if x is not None:
                    x.close()
            except Exception:
                pass
    ok = [s for s in R["steps"] if s["step"].startswith("read_") and s.get("frames")]
    out = default_runs_dir() / f"SPIPROBE_{datetime.now():%Y%m%d_%H%M%S}.md"
    L = ["# SPI probe", "", f"preset `{cfg.name}` (link=spi), requested SPI clock {a.spi_clock/1e6:g} MHz"
         + (" — **SIMULATOR**" if a.sim else ""), "",
         f"**Result: {'frames received (' + ok[0]['step'] + ', ' + str(ok[0]['fps']) + ' fps)' if ok else 'NO FRAMES'}**",
         "", "| step | key facts |", "|---|---|"]
    for s in R["steps"]:
        facts = {k: v for k, v in s.items() if k not in ("step", "parser", "link", "commands", "reads")}
        L.append(f"| {s['step']} | `{json.dumps(facts)[:300]}` |")
    L += ["", "```json", json.dumps(R, indent=1, default=str), "```"]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n{'OK' if ok else 'NO FRAMES'} - report: {out}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------- bench
def bench_variants(quick: bool = False):
    """(group, label, config) with growing cube size / frame rate. Link settings are applied later."""
    from .config import presets
    P = presets()
    c1 = P["coarse-1d-20cm"]
    d2 = P["doppler-2d-spi"]
    V = [("1d-size", "1d-32bins-25ms", c1),
         ("1d-size", "1d-64bins-25ms", c1.copy(range_res_mm=100)),
         ("1d-size", "1d-128bins-25ms", c1.copy(range_res_mm=50)),
         ("1d-size", "1d-256bins-25ms", c1.copy(range_max_mm=12800, range_res_mm=50)),
         ("1d-rate", "1d-32bins-10ms", c1.copy(frame_period_ms=10)),
         ("1d-rate", "1d-32bins-5ms", c1.copy(frame_period_ms=5)),
         ("2d-size", "2d-64x8-50ms", d2.copy(veloc_res_mm_s=500, frame_period_ms=50)),
         ("2d-size", "2d-64x16-50ms", d2.copy(veloc_res_mm_s=250, frame_period_ms=50)),
         ("2d-size", "2d-64x32-50ms", d2.copy(frame_period_ms=50)),
         ("2d-size", "2d-64x64-60ms", d2.copy(veloc_res_mm_s=62, frame_period_ms=60)),
         ("2d-rate", "2d-64x32-100ms", d2.copy(frame_period_ms=100)),
         ("2d-rate", "2d-64x32-40ms", d2.copy(frame_period_ms=40)),
         ("2d-rate", "2d-64x32-30ms", d2.copy(frame_period_ms=30))]
    if quick:
        keep = {"1d-32bins-25ms", "1d-256bins-25ms", "1d-32bins-5ms", "2d-64x16-50ms", "2d-64x32-40ms",
                "2d-64x64-60ms"}
        V = [v for v in V if v[1] in keep]
    return [(g, lab, c.copy(name=f"bench-{lab}")) for g, lab, c in V]


def linkbench(a) -> int:
    from .config import has_errors
    from .engine import Engine
    from .session import default_runs_dir
    eng = Engine()
    rows = []
    try:
        aborted = None
        for group, label, cfg in bench_variants(a.quick):
            if aborted:
                rows.append({"group": group, "variant": label, "link": a.link, "cube_B": cfg.derive().cube_bytes,
                             "fps_req": round(1000 / cfg.frame_period_ms, 1), "model_ms": None, "validation_errors": [],
                             "skipped": f"not run: {aborted}"})
                continue
            if a.link == "spi":
                cfg = cfg.copy(link="spi", spi_clock_hz=int(a.spi_clock))
            else:
                cfg = cfg.copy(link="uart", baudrate=a.baud)
            iss = cfg.validate()
            errs = [i.code for i in iss if i.level == "error"]
            non_link = [e for e in errs if e != "link"]
            row = {"group": group, "variant": label, "link": cfg.link, "cube_B": cfg.derive().cube_bytes,
                   "fps_req": round(1000 / cfg.frame_period_ms, 1), "model_ms": round(cfg.derive().tx_time_ms, 1),
                   "validation_errors": errs}
            if non_link:
                row["skipped"] = f"invalid config: {non_link}"
                rows.append(row)
                print(f"--- {label}: SKIPPED ({non_link})")
                continue
            print(f"--- {label}: cube {row['cube_B']} B @ {row['fps_req']} fps requested"
                  + (" (model says link-limited)" if errs else ""))
            if a.sim:
                eng.start_sim(cfg, "static-target-1m")
            elif a.link == "spi":
                eng.start_spi(cfg, a.ctrl, a.spi_dev, int_bit_arg(a.int_bit), ch347_dll=a.ch347_dll,
                              label=f"bench_{label}")
            else:
                eng.start_serial(cfg, a.ctrl, a.data, diagnose=False, label=f"bench_{label}")
            # measure only the streaming part: wait for the first frame, then a fixed window
            t0 = time.monotonic()
            while eng.frame_counter == 0 and time.monotonic() - t0 < 6 and eng.state != "error":
                time.sleep(0.05)
            src = eng.source
            f0, g0 = src.stats.frames, src.stats.index_gaps
            b0 = src.parser.stats.bytes_in
            t1 = time.monotonic()
            while time.monotonic() - t1 < a.seconds and eng.state != "error":
                time.sleep(0.1)
            dt = max(1e-3, time.monotonic() - t1)
            frames = src.stats.frames - f0
            gaps = src.stats.index_gaps - g0
            row.update({"frames": frames, "fps": round(frames / dt, 1), "chip_frames_lost": gaps,
                        "loss_pct": round(100 * gaps / max(1, frames + gaps), 1),
                        "kB_s": round((src.parser.stats.bytes_in - b0) / dt / 1e3, 1),
                        "proc_ms_median": round(sorted(eng.session.proc_ms)[len(eng.session.proc_ms) // 2], 2)
                        if eng.session and eng.session.proc_ms else None,
                        "parse_errors": dict(src.parser.stats.errors), "state": eng.state,
                        "run_dir": str(eng.session.dir) if eng.session else None})
            if hasattr(src, "spi_stats"):
                s = src.spi_stats()
                row["spi"] = {k: s.get(k) for k in ("int_bit", "polls", "polls_empty", "int_timeouts", "messages",
                                                      "check32_fail", "mean_xfer_us", "pad_mode")}
            rep_ = (eng.session.config_report or {}) if eng.session else {}
            no_resp = [r["cmd"] for r in rep_.get("results", []) if r.get("status") == "no_response"]
            err = str(eng.state_info.get("error", "")) if eng.state == "error" else ""
            eng.stop(f"linkbench {label} done")
            if no_resp:
                row["chip_no_response"] = no_resp
            if err:
                row["error"] = err[:200]
            rows.append(row)
            if (len(no_resp) >= 3 or "could not open port" in err or "PermissionError" in err
                    or not rep_.get("boot_ok", True)):
                aborted = (f"board stopped answering after `{label}` ({'; '.join(no_resp[:2]) or err[:80]}). "
                           "Unplug/replug USB and rerun")
                print(f"    !! {aborted}")
            print(f"    -> {row['fps']} fps of {row['fps_req']} requested, chip frames lost {row['loss_pct']} %, "
                  f"{row['kB_s']} kB/s")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("interrupted - writing partial summary")
    except Exception as e:  # never lose the rows measured so far
        log.exception("linkbench aborted: %s", e)
        rows.append({"group": "-", "variant": "ABORTED", "link": a.link, "cube_B": 0, "fps_req": 0,
                     "model_ms": 0, "validation_errors": [], "skipped": f"crash: {e!r}"[:200]})
    finally:
        try:
            eng.stop("linkbench end")
        except Exception:
            pass
    out = default_runs_dir() / f"LINKBENCH_{a.link}_{datetime.now():%Y%m%d_%H%M%S}.md"
    L = [f"# Link bench — {a.link.upper()}" + (" (SIMULATOR)" if a.sim else ""), "",
         f"{a.seconds:.0f} s per variant (measured after the first frame). `chip frames lost` = gaps in the radar's own "
         "frame index: frames the chip produced that never reached the host (link, bridge or chip-side abort).", "",
         "| group | variant | cube B | fps req | fps got | lost % | kB/s | model ms/frame | notes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = r.get("skipped") or ("model: link-limited" if r["validation_errors"] else "")
        if r.get("chip_no_response"):
            note += f" chip stopped answering at `{r['chip_no_response'][0]}`"
        if r.get("error"):
            note += f" error: {r['error'][:60]}"
        if r.get("parse_errors"):
            note += f" parse errors {r['parse_errors']}"
        L.append(f"| {r['group']} | {r['variant']} | {r['cube_B']} | {r['fps_req']} | {r.get('fps', '-')} | "
                 f"{r.get('loss_pct', '-')} | {r.get('kB_s', '-')} | {r['model_ms']} | {note} |")
    L += ["", "```json", json.dumps(rows, indent=1), "```"]
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\nsummary: {out}")
    return 0
