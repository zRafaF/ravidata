"""Progress of a pod run, readable while `make pod` runs in another tmux window: `make pod-status`
(`python3 pod/status.py [work_dir] [N log lines, default 25]`).

Standard library only (works before / without the project env). Reads work/: the stage times
(metrics/stages.csv), the windows each model has written so far (out/<model>/<take>/win_*.npz), each run.json
(written after every window) and the check results, then prints done / total windows per model and take, an
estimate of the time left, the GPU right now and the last lines of the log.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

ARGS = [a for a in sys.argv[1:] if not a.isdigit()]
WORK = Path(ARGS[0]) if ARGS else Path("work")
N_LINES = int(next((a for a in sys.argv[1:] if a.isdigit()), "25"))     # log lines shown at the end
STAGES = ["setup", "data", "frames", "check", "run", "upload", "stop"]


def n_windows(n: int, size: int, overlap: int) -> int:
    """Same rule as pod/runners/common.py windows()."""
    if size <= 0 or size >= n:
        return 1
    step = max(1, size - overlap)
    starts = list(range(0, max(1, n - overlap), step))
    last_end = min(n, starts[-1] + size)
    return len(starts) + (0 if last_end == n else 1)


def load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def main():
    if not WORK.exists():
        print(f"{WORK}/ does not exist yet: run this inside the ravidata folder (cd /workspace/ravidata)")
        return
    log = WORK / "logs" / "pod.log"
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    st = WORK / "metrics" / "stages.csv"
    rows = [(r.split(",") + ["", "", "", ""])[:4] for r in st.read_text().splitlines()] if st.exists() else []
    done_stages = [r[0] for r in rows]
    failed = [r[0] for r in rows if r[3] not in ("ok", "")]
    if failed:
        stage = f"{failed[0]} FAILED -> upload / stop"
    else:
        stage = next((s for s in STAGES if s not in done_stages), "finished")
        if stage == "upload" and any("=== stop ===" in ln for ln in lines[-50:]):
            stage = "stop"
    since = int(rows[-1][2]) if rows else None
    print(f"stage now: {stage}" + (f" (for {fmt(time.time() - since)})" if since else ""))
    for name, a, b, res in rows:
        print(f"  {name:7s} {fmt(int(b) - int(a)):>7s}  {res}")

    frames = {}
    for fj in sorted((WORK / "frames").glob("*/frames.json")):
        m = load(fj)
        frames[fj.parent.name] = len(m.get("frames", []))
    if frames:
        print("keyframes: " + ", ".join(f"{t} {n}" for t, n in frames.items()))

    ok = WORK / "check_ok.txt"
    if not ok.exists() or not frames:
        print("models: not checked yet")
    else:
        print(f"\n{'model':12s} {'take':16s} {'windows':>9s}  {'s/window':>8s}  {'peak GB':>7s}  status")
        left_total = 0.0
        for row in ok.read_text().split("\n"):
            if not row.strip():
                continue
            m, w = row.split()[0], int(row.split()[1])
            ov = min(w // 4, 16)
            ref = load(WORK / "metrics" / "check" / f"{m}_w{w}.json")
            for t, n in frames.items():
                total = n_windows(n, w, ov)
                r = load(WORK / "out" / m / t / "run.json")
                done = len(list((WORK / "out" / m / t).glob("win_*.npz")))
                ws = [x["seconds"] for x in r.get("windows", [])] or [x["seconds"] for x in ref.get("windows", [])]
                per = sum(ws) / len(ws) if ws else float("nan")
                load_s = r.get("model_load_s") or ref.get("model_load_s") or 0.0
                status = r.get("status", "waiting")
                left = 0.0 if status == "done" else (total - done) * (per if not math.isnan(per) else 0) + \
                    (0 if done else load_s)
                left_total += left
                peak = r.get("peak_vram_gb") or ref.get("peak_vram_gb")
                print(f"{m:12s} {t:16s} {done:4d}/{total:<4d}  {per:8.1f}  {peak if peak else '-':>7}  {status}"
                      + (f", ~{fmt(left)} left" if status != "done" else ""))
        print(f"\nmodels left: ~{fmt(left_total)} (+ upload ~5-10 min)")

    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,"
                            "temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
        if g:
            u, mu, mt, pw, tc = [x.strip() for x in g.split(",")]
            print(f"GPU now: {u} % busy, {mu}/{mt} MiB, {pw} W, {tc} C")
    except Exception:
        pass
    if lines:
        print(f"\nlast {N_LINES} log lines (work/logs/pod.log):")
        for ln in lines[-N_LINES:]:
            print("  " + ln[:160])
    print(time.strftime("(%H:%M:%S)"))


if __name__ == "__main__":
    main()
