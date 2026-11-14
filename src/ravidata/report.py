"""Builds ``SESSION_REPORT.md``: a compact, structured digest of one run for humans and agents.

Order: verdict first, then the things that usually explain a bad run (validation issues,
rejected chip commands, link/parse problems), then processing results and the timeline.
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .session import RunSession


def _short(s: str, n: int = 90) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return (s[: n - 1] + "…") if len(s) > n else s


def _verdict(S: "RunSession") -> list[str]:
    out = []
    dur = time.time() - S.t0
    states = [s for _, s, _ in S.state_history]
    rep = S.config_report or {}
    d = S.cfg.derive()
    fps = S.source_stats.get("fps_recent", 0) or 0
    if S.source_kind in ("serial", "sim"):
        if rep and not rep.get("boot_ok", True):
            out.append("**FAILED: the radar did not answer on the control port.** Check port assignment (the higher COM "
                       "number is UART0 on the dev board), baud rate and power.")
        elif rep.get("failed"):
            out.append(f"**{len(rep['failed'])} config command(s) rejected by the chip:** `{'`, `'.join(rep['failed'])}`. "
                       "The stream may not match the requested configuration.")
        if rep.get("check_mismatches"):
            out.append(f"**Chip read-back differs from the config:** {', '.join(rep['check_mismatches'])}.")
        if "streaming" in states and S.frames == 0:
            out.append("**No frames decoded** although streaming was started. See 'Data link' below.")
    if S.frames:
        ratio = fps / d.expected_fps if d.expected_fps else 0
        out.append(f"{S.frames} frames in {dur:.0f} s; recent rate {fps:.1f} fps vs {d.expected_fps:.1f} expected "
                   f"({100*ratio:.0f} %).")
    if S.source_stats.get("shape_mismatch_frames"):
        out.append(f"**Cube size mismatch**: decoded as {S.source_stats.get('shape_used')} instead of the configured shape.")
    if S.source_stats.get("index_gaps"):
        out.append(f"Radar frame-index gaps: {S.source_stats['index_gaps']} frames missing (link or chip dropped them).")
    err = S.parser_stats.get("errors") or {}
    if err:
        out.append(f"HIF parse errors: {err}.")
    val_err = [v for v in S.meta.get("validation", []) if v["level"] == "error"]
    if val_err:
        out.append(f"Config had {len(val_err)} validation error(s) before sending: "
                   + ", ".join(v["code"] for v in val_err) + ".")
    if "error" in states:
        errs = [i.get("error") for _, s, i in S.state_history if s == "error"]
        out.append(f"**Source error:** {errs[-1] if errs else '?'}")
    if not out:
        out.append("Nothing notable.")
    return out


def build_report(S: "RunSession") -> str:
    L: list[str] = []
    A = L.append
    cfg, d = S.cfg, S.cfg.derive()
    A(f"# Run `{S.run_id}`")
    A("")
    A(f"- Started: {S.meta['started']}  |  duration: {time.time() - S.t0:.0f} s  |  source: **{S.source_kind}** "
      f"{json.dumps({k: v for k, v in S.meta['source'].items() if k != 'kind'})}")
    A(f"- Preset/config: **{cfg.name}** — {cfg.notes}")
    A(f"- Tool {S.meta.get('tool_version')} · git {S.meta.get('git', {}).get('commit')} "
      f"(dirty files: {S.meta.get('git', {}).get('dirty_files')}) · {S.meta['host']['platform']}")
    A(f"- Status: {'closed' if S.closed else 'RUNNING (report refreshed every 30 s)'}")
    A("")
    A("## Verdict")
    A("")
    for v in _verdict(S):
        A(f"- {v}")
    A("")

    A("## Configuration")
    A("")
    A("Derived (from the config, not measured):")
    A("")
    for line in d.summary_lines():
        A(f"- {line}")
    A("")
    val = S.meta.get("validation", [])
    A(f"Validation issues ({len(val)}):" if val else "Validation issues: none")
    A("")
    for v in val:
        A(f"- **{v['level']}** `{v['code']}`: {v['message']} _({v['source']})_")
    if val:
        A("")
    A("<details><summary>Full config JSON</summary>")
    A("")
    A("```json")
    A(json.dumps(cfg.to_dict(), indent=1))
    A("```")
    A("</details>")
    A("")

    rep = S.config_report
    if rep:
        A("## Chip command log")
        A("")
        failed = [r for r in rep["results"] if r["status"] in ("fail", "no_response", "warn")]
        if failed:
            A("Rejected / problematic commands:")
            A("")
            for r in failed:
                A(f"- `mmwc {r['cmd']}` → **{r['status']}**: {_short(r['response'], 200) or '<no bytes>'}")
            A("")
        A("| # | phase | command | status | ms | response |")
        A("|---|---|---|---|---|---|")
        for i, r in enumerate(rep["results"]):
            A(f"| {i} | {r['phase']} | `{r['cmd']}` | {r['status']} | {r['elapsed_ms']:.0f} | {_short(r['response'], 70)} |")
        A("")
        if rep.get("checks"):
            A("Read-back checks (chip's own answer vs. what the config predicts):")
            A("")
            A("| item | expected | chip | result |")
            A("|---|---|---|---|")
            for c in rep["checks"]:
                A(f"| {c['item']} | {c['expected']} | {c['chip']} | {'ok' if c['ok'] else '**MISMATCH** ' + c['note']} |")
            A("")
        if rep.get("queries"):
            A("Read-back values (raw):")
            A("")
            for q, a in rep["queries"].items():
                A(f"- `{q}`: {_short(a, 200) or '<empty>'}")
            A("")

    A("## Data link")
    A("")
    A(f"- Source stats: `{json.dumps(S.source_stats)}`")
    A(f"- Parser stats: `{json.dumps(S.parser_stats)}`")
    link_txt = (f"SPI at {cfg.spi_clock_hz/1e6:g} MHz (model)" if getattr(cfg, "link", "uart") == "spi"
                else f"the UART at {cfg.baudrate} baud")
    A(f"- Expected per frame: {d.cube_dwords} dwords ({d.cube_bytes} B), {d.tx_time_ms:.1f} ms on {link_txt}")
    dur_s = max(1e-3, time.time() - S.t0)
    if S.parser_stats.get("bytes_in"):
        A(f"- Delivered: {S.parser_stats['bytes_in']/dur_s/1e3:.1f} kB/s averaged over the run "
          f"({S.parser_stats['bytes_in']} B in {dur_s:.0f} s, including configuration time)")
    spi = S.source_stats.get("spi")
    if spi:
        A(f"- SPI link: `{json.dumps(spi)}`")
    A("")

    A("## Processing")
    A("")
    if S.proc_ms:
        A(f"- Host processing time per frame (Python, all detectors): median {np.median(S.proc_ms):.2f} ms, "
          f"p95 {np.percentile(S.proc_ms, 95):.2f} ms")
    for name, st in S.det_stats.items():
        r = np.array(st["ranges"]) if st["ranges"] else None
        q = (f"nearest range quantiles p10/p50/p90 = {np.percentile(r,10):.2f}/{np.median(r):.2f}/"
             f"{np.percentile(r,90):.2f} m" if r is not None else "no alerts")
        A(f"- **{name}**: alert in {st['alert_frames']}/{st['frames']} frames "
          f"({100*st['alert_frames']/max(1,st['frames']):.0f} %), {st['transitions']} on/off transitions, "
          f"first alert at {st['first_alert_t']} s; {q}")
    if S.truth_stats:
        ts = dict(S.truth_stats)
        hits = ts.get("hit", 0)
        mean_err = ts.pop("range_err_cm_sum", 0) / max(1, hits)
        A(f"- Simulator ground truth (primary detector): {ts}; mean range error on hits {mean_err:.0f} cm")
    if S.params_last:
        A("")
        A("<details><summary>Processing parameters at the end of the run</summary>")
        A("")
        A("```json")
        A(json.dumps(S.params_last, indent=1, default=str))
        A("```")
        A("</details>")
    A("")

    if S.recordings:
        A("## Recordings")
        A("")
        for r in S.recordings:
            A(f"- `{r['file']}`: from t={r['start_t']} s, {r.get('frames', '(recording)')} frames")
        A("")

    A("## Timeline")
    A("")
    tl = [(t, f"state **{s}** {_short(json.dumps(info, default=str), 150) if info else ''}")
          for t, s, info in S.state_history]
    tl += [(n["t"], f"USER {n['kind']}: {n['text']} (frame {n['frame']})") for n in S.notes]
    tl += [(r["start_t"], f"recording {r['file']} started") for r in S.recordings]
    for t, txt in sorted(tl, key=lambda x: x[0]):
        A(f"- {t:7.1f} s  {txt}")
    A("")

    issues = sorted(S.logs.issues.items.values(), key=lambda x: (-{"CRITICAL": 3, "ERROR": 2, "WARNING": 1}.get(x["level"], 0),
                                                                   -x["count"]))
    A(f"## Warnings and errors ({len(issues)} distinct)")
    A("")
    for it in issues[:60]:
        A(f"- **{it['level']}** ×{it['count']} `{it['logger']}`: {_short(it['example'], 220)}")
    if len(issues) > 60:
        A(f"- … {len(issues) - 60} more in summary.log")
    A("")
    A("## Files")
    A("")
    A("`summary.log` (strict), `verbose.log` (everything), `meta.json`, `events.jsonl`"
      + (", " + ", ".join(f"`{r['file']}`" for r in S.recordings) if S.recordings else "") + ".")
    A("")
    return "\n".join(L)
