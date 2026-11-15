"""Build the public Hugging Face dataset from the 2026-11-13 recordings: blur faces, export, upload.

    uv run python scripts/hf_dataset.py faces   [--only NAME] [--max-frames N]   # pass 1: detect faces -> faces.json + QA sheet
    uv run python scripts/hf_dataset.py blur    [--only NAME] [--max-frames N]   # pass 2: blurred, upright 10-bit HEVC video
    uv run python scripts/hf_dataset.py export  [--only NAME]                    # radar, IMU, sync, metadata, dataset card
    uv run python scripts/hf_dataset.py all                                      # faces + blur + export
    uv run python scripts/hf_dataset.py upload --repo <user>/<name>              # needs `hf auth login` (write token)

Output: ``dataset/hf/`` (git-ignored). Layout and conventions: the dataset card written by ``export``
(``dataset/hf/README.md``); the source of the card text is this file.

Privacy: faces are blurred (YuNet face detector on every frame, boxes kept ±8 frames, pixelated + blurred,
irreversible); the Sensor Logger ``device id`` is removed; Windows paths / host names are not exported; only safe
container tags are copied (no location). The sound is band-passed to 1.0–1.4 kHz by default (``--audio beeps``):
the recorder's 1.2 kHz beeps that align the video survive, speech does not.

Video: the phone's Log (BT.2020, 10-bit) is kept as is, turned upright (the phone stores it sideways), with the
original frame times (pts) and the original audio, so the calibration's "display" intrinsics and the beep times
apply unchanged. Convert Log to Rec.709 before using vision models (not done here: the data stay as recorded).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RECS = ROOT / "runs" / "recs"
OUT = ROOT / "dataset" / "hf"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
SAFE_TAGS = ("creation_time", "com.android.capture.fps", "com.samsung.android.logvideo")

# published name -> (folder under runs/recs, kind, description)
TAKES = {
    "static_pass1": ("s-static-pass1_20261113-174743", "static",
                     "Radar still, indoor lab: empty scene (S0) → objects placed (S1) → person walking (tags 0 chest, 1 back) "
                     "+ broom with four tags (20, 21, 50, 51) swept through the field of view (S2). Background 1."),
    "static_pass2": ("s-static-pass2_20261113-175741", "static", "As static_pass1, another background / objects."),
    "indoor_course": ("d1-carried-indoor-pass1_20261113-181354", "carried",
                      "Radar + phone carried along an indoor course; floor tags 10, 12, 17; one pass of the course per "
                      "radar config."),
    "outdoor_course": ("d2-carried-outdoor-pass1_20261113-182619", "carried", "As indoor_course, outdoors."),
    "pilot_chairs": ("closed-chairs", "static",
                     "Pilot: radar still, three chairs with tags 11/13/15 at ~2 m, then a person walking (two 1D, two "
                     "2D, one 1D recording). Muted video (may contain conversation)."),
}
# sound of the published video per name: "beeps" (only the sync beeps), "none" (muted), "copy" (original)
AUDIO = {"calibration": "none", "pilot_chairs": "none"}
CALIB = {"folder": "calib", "video": "samsung_2026-11-13.mp4", "json": "samsung_2026-11-13.calib.json"}

DATASET_VERSION = "v1.1"          # written into every take.json; the analysis refuses older local copies
DEFAULT_OFFSET_S = 0.56
DEFAULT_BECAUSE = ("no radar-vs-video motion check with corr >= 0.4: the default 0.56 s, the extra delay of the video "
                   "measured on the pilot take (pilot_chairs) by matching the radar range of a walking tagged person "
                   "with the tag range in the video")
# Corrections to the computed clock alignment, applied by `export` and by `scripts/hf_release.py patch` (idempotent).
# "new" replaces `field` and records the old value in sync["corrections"]; "because" replaces recommended_because.
SYNC_CORRECTIONS = {
    "indoor_course": {
        "dataset_version": "v1.1", "date": "2026-11-15", "field": "recommended_video_offset_s", "new": -2.44,
        "note": "the IMU moves with the video (imu_lag_s is measured against the video)",
        "because": "corrected on 2026-11-15 (dataset v1.1): the default 0.56 s (measured on the pilot take) left the "
                   "video 3.0 s late. Measured instead, in all three recordings: radar floor return vs camera height "
                   "above the floor tags (best offset -2.4 .. -2.7 s), radar range-time ridge vs the map and AprilTag "
                   "ranges (-2.7 .. -3.4 s, nothing at the old value), and radial speed from the radar Doppler vs "
                   "walking speed from the camera map (r -0.77 / -0.68 at zero lag with -2.44 s, about -0.1 at "
                   "+-3 s). With -2.44 s the phone clock minus host clock (imu_lag_s + offset) is -0.33 s, in line "
                   "with the other takes (-0.25 .. -0.37 s)."},
    "outdoor_course": {
        "because": DEFAULT_BECAUSE + ". Checked on 2026-11-15: the radar floor return vs camera height gives "
                   "+0.4 .. +0.6 s in the three recordings, consistent with it."},
}


def correct_sync(name: str, sync: dict) -> dict:
    """``sync`` with SYNC_CORRECTIONS[name] applied; running it again changes nothing."""
    c = SYNC_CORRECTIONS.get(name)
    if not c:
        return sync
    s = dict(sync)
    if "new" in c:
        f = c["field"]
        done = any(x.get("dataset_version") == c["dataset_version"] and x.get("field") == f
                   for x in s.get("corrections", []))
        if not done:
            s["corrections"] = list(s.get("corrections", [])) + [
                {"date": c["date"], "dataset_version": c["dataset_version"], "field": f, "old": s.get(f),
                 "new": c["new"], "note": c["note"]}]
        s[f] = c["new"]
    s["recommended_because"] = c["because"]
    return s


# ================================================================================================ helpers
def _video_of(folder: Path) -> Path:
    vids = sorted(p for p in folder.glob("*.mp4") if ".blurred" not in p.name)
    if not vids:
        raise SystemExit(f"no video in {folder}")
    return max(vids, key=lambda p: p.stat().st_size)          # the original, not a small copy


def _zip_of(folder: Path) -> Path | None:
    z = sorted(folder.glob("*.zip"))
    return z[0] if z else None


def _sources(only: str | None) -> list[tuple[str, Path]]:
    """(published name, source video) for the calibration clip + every take."""
    out = [("calibration", RECS / CALIB["folder"] / CALIB["video"])]
    out += [(n, _video_of(RECS / f)) for n, (f, _, _) in TAKES.items()]
    return [(n, p) for n, p in out if only in (None, n)]


def _work(name: str) -> Path:
    d = OUT / ("calibration" if name == "calibration" else f"takes/{name}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def yunet_path() -> Path:
    p = Path.home() / ".cache" / "ravidata" / "face_detection_yunet_2023mar.onnx"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading the YuNet face detector -> {p}")
        urllib.request.urlretrieve(YUNET_URL, p)
    return p


# ------------------------------------------------------------------------------------------------ frames
def planes(a: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PyAV yuv420p(10le) ndarray (h*3/2, w) -> Y (h, w), U (h/2, w/2), V (h/2, w/2)."""
    y = a[:h]
    u = a[h:h + h // 4].reshape(h // 2, w // 2)
    v = a[h + h // 4:].reshape(h // 2, w // 2)
    return y, u, v


def pack(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = y.shape
    return np.concatenate([y, u.reshape(h // 4, w), v.reshape(h // 4, w)])


def upright(y, u, v, rotation: int):
    k = int(round(rotation / 90.0)) % 4        # PyAV display rotation; -90 = turn clockwise (as ravidata.calib)
    if k == 0:
        return y, u, v
    return tuple(np.ascontiguousarray(np.rot90(p, k=k)) for p in (y, u, v))


def det_image(y: np.ndarray, bits: int, width: int) -> tuple[np.ndarray, float]:
    """Y plane -> contrast-stretched 8-bit BGR at ``width`` px for the detector; returns (image, scale)."""
    import cv2
    g = y.astype(np.float32) / (4.0 if bits == 10 else 1.0)
    lo, hi = np.percentile(g[::8, ::8], (1, 99.5))
    g8 = np.clip((g - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    s = width / g8.shape[1]
    g8 = cv2.resize(g8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR), s


# ================================================================================================ pass 1: faces
def detect_faces(name: str, src: Path, max_frames: int | None = None, det_width: int = 960,
                 score: float = 0.55) -> Path:
    import av
    import cv2
    out = _work(name) / "faces.json"
    det = None
    frames, t0 = [], time.monotonic()
    sheet = []
    with av.open(str(src)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        bits = 10 if "10" in vs.codec_context.pix_fmt else 8
        W, H = vs.codec_context.width, vs.codec_context.height
        rot = None
        for i, fr in enumerate(c.decode(vs)):
            if max_frames and i >= max_frames:
                break
            if rot is None:
                rot = int(round(getattr(fr, "rotation", 0) or 0))
            y, u, v = upright(*planes(fr.to_ndarray(), W, H), rot)
            img, s = det_image(y, bits, det_width)
            if det is None:
                det = cv2.FaceDetectorYN.create(str(yunet_path()), "", (img.shape[1], img.shape[0]), score, 0.3, 5000)
            _, f = det.detect(img)
            boxes = [] if f is None else [[round(float(b) / s, 1) for b in r[:4]] + [round(float(r[14]), 3)]
                                          for r in f]
            frames.append({"i": i, "pts": fr.pts, "boxes": boxes})
            if boxes and (len(sheet) < 48) and (i % 45 == 0):
                sheet.append((i, y, boxes))
            if i % 300 == 0:
                print(f"  {name}: frame {i}, {sum(bool(x['boxes']) for x in frames)} with faces "
                      f"({i / max(time.monotonic() - t0, 1e-3):.0f} fps)", flush=True)
    info = {"video": src.name, "rotation": rot, "bits": bits, "detector": "YuNet 2023mar (OpenCV zoo, MIT)",
            "det_width": det_width, "score_threshold": score, "frames": frames}
    out.write_text(json.dumps(info), encoding="utf-8")
    n = sum(bool(x["boxes"]) for x in frames)
    print(f"{name}: {len(frames)} frames, faces in {n} -> {out}")
    _qa_sheet(out.with_name("faces_qa.jpg"), sheet, bits)
    return out


def _qa_sheet(path: Path, sheet, bits):
    """Up to 48 frames with faces, boxes drawn (for a human check before blurring)."""
    import cv2
    tiles = []
    for i, y, boxes in sheet:
        img, s = det_image(y, bits, 480)
        for x, yy, w, h, sc in boxes:
            cv2.rectangle(img, (int(x * s), int(yy * s)), (int((x + w) * s), int((yy + h) * s)), (0, 0, 255), 2)
        cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        return
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


# ================================================================================================ pass 2: blur
def blur_boxes(frames: list[dict], n: int, keep: int = 8, grow_w: float = 1.5, grow_h: float = 1.7):
    """Per frame, the union of the face boxes of frames i-keep..i+keep (covers missed detections), enlarged
    (hair, ears) around the box centre."""
    raw = [[] for _ in range(n)]
    for f in frames:
        if f["i"] < n:
            raw[f["i"]] = f["boxes"]
    out = [[] for _ in range(n)]
    for i, bx in enumerate(raw):
        if not bx:
            continue
        for j in range(max(0, i - keep), min(n, i + keep + 1)):
            for x, y, w, h, _ in bx:
                cx, cy = x + w / 2, y + h / 2
                out[j].append((cx - w * grow_w / 2, cy - h * grow_h / 2, w * grow_w, h * grow_h))
    return out


def _anonymise(p: np.ndarray, boxes, scale: float):
    """Pixelate (16 px blocks at full size) + blur each box in place: irreversible."""
    import cv2
    H, W = p.shape
    for x, y, w, h in boxes:
        x0, y0 = max(0, int(x * scale)), max(0, int(y * scale))
        x1, y1 = min(W, int((x + w) * scale) + 1), min(H, int((y + h) * scale) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = p[y0:y1, x0:x1].astype(np.float32)
        b = max(1, int(16 * scale))
        small = cv2.resize(roi, (max(1, (x1 - x0) // b), max(1, (y1 - y0) // b)), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = 2 * max(1, b) + 1
        p[y0:y1, x0:x1] = cv2.GaussianBlur(big, (k, k), 0).astype(p.dtype)


def blur_still(path: Path, score: float = 0.5) -> int:
    """Faces in a still image (the calibration figure is made from the original video) -> pixelated + blurred,
    boxes enlarged as in the videos. Returns the number of faces."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return 0
    det = cv2.FaceDetectorYN.create(str(yunet_path()), "", (img.shape[1], img.shape[0]), score, 0.3, 5000)
    _, f = det.detect(img)
    boxes = [] if f is None else [r[:4] for r in f]
    grown = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2, y + h / 2
        grown.append((cx - w * 0.9, cy - h * 1.0, w * 1.8, h * 2.0))
    for c in range(3):
        ch = np.ascontiguousarray(img[..., c])
        _anonymise(ch, grown, 1.0)
        img[..., c] = ch
    cv2.imwrite(str(path), img)
    return len(boxes)


def blur_video(name: str, src: Path, max_frames: int | None = None, crf: int = 20, preset: str = "fast",
               audio: str = "copy") -> Path:
    import av
    work = _work(name)
    fj = json.loads((work / "faces.json").read_text(encoding="utf-8"))
    out = work / ("video.mp4" if name != "calibration" else "checkerboard.mp4")
    tmp = out.with_suffix(".part.mp4")
    with av.open(str(src)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        W, H = vs.codec_context.width, vs.codec_context.height
        n = vs.frames or len(fj["frames"])
        n = min(n, max_frames) if max_frames else n
        bx = blur_boxes(fj["frames"], max(n, len(fj["frames"])))
        rot = fj["rotation"] or 0
        bits = fj["bits"]
        o = av.open(str(tmp), "w", options={"movflags": "use_metadata_tags"})   # keep the Samsung Log tags
        o.metadata.update({k: v for k, v in c.metadata.items() if k in SAFE_TAGS})
        ov = o.add_stream("libx265", rate=vs.average_rate)
        k = int(round(rot / 90.0)) % 4
        ov.width, ov.height = (H, W) if k % 2 else (W, H)
        ov.pix_fmt = "yuv420p10le" if bits == 10 else "yuv420p"
        ov.time_base = vs.time_base
        ov.codec_context.time_base = vs.time_base
        xp = "log-level=error" + (":colorprim=bt2020:colormatrix=bt2020nc:range=limited" if bits == 10 else
                                  ":colorprim=bt709:colormatrix=bt709:range=limited")
        ov.options = {"crf": str(crf), "preset": preset, "x265-params": xp}
        aus = c.streams.audio[0] if c.streams.audio else None
        oa = None
        if aus is not None and audio == "copy":
            oa = o.add_stream_from_template(aus)
        beeps = _beep_band(src) if (aus is not None and audio == "beeps") else None
        sa = _audio_stream(o, beeps[1], beeps[3]) if beeps is not None else None   # before the first mux
        streams = [s for s in (vs, aus) if s is not None and (s is vs or oa is not None)]
        i, t0 = 0, time.monotonic()
        qa_step, thumbs = max(1, n // 60), []
        for pkt in c.demux(*streams):
            if pkt.stream is aus:
                if pkt.dts is None:
                    continue
                pkt.stream = oa
                o.mux(pkt)
                continue
            for fr in pkt.decode():
                if max_frames and i >= max_frames:
                    break
                y, u, v = upright(*planes(fr.to_ndarray(), W, H), rot)
                y, u, v = y.copy(), u.copy(), v.copy()
                if i < len(bx) and bx[i]:
                    _anonymise(y, bx[i], 1.0)
                    _anonymise(u, bx[i], 0.5)
                    _anonymise(v, bx[i], 0.5)
                if i % qa_step == 0:
                    thumbs.append((i, y.copy()))
                nf = av.VideoFrame.from_ndarray(pack(y, u, v), format=ov.pix_fmt)
                nf.pts, nf.time_base = fr.pts, vs.time_base
                for p in ov.encode(nf):
                    o.mux(p)
                i += 1
                if i % 300 == 0:
                    print(f"  {name}: {i}/{n} frames ({i / max(time.monotonic() - t0, 1e-3):.1f} fps)", flush=True)
            if max_frames and i >= max_frames:
                break
        for p in ov.encode():
            o.mux(p)
        if beeps is not None:
            _write_audio(o, sa, *beeps)
        o.close()
    tmp.replace(out)
    _qa_sheet(work / "blurred_qa.jpg", [(k, yy, []) for k, yy in thumbs], bits)   # look for missed faces here
    print(f"{name}: {i} frames -> {out} ({out.stat().st_size / 1e6:.0f} MB); check {work / 'blurred_qa.jpg'}")
    return out


def _beep_band(src: Path, lo: float = 1000.0, hi: float = 1400.0, pad: float = 0.1):
    """Only the recorder's beeps of the source's sound: band-passed to 1.0-1.4 kHz (zero-phase: the beep times do not
    move) and kept only around the beeps that ``ravidata.align.detect_tones`` finds in the original (±``pad`` s,
    10 ms fades); silence elsewhere, so no speech survives and no false beeps appear (the detector compares the beep
    band with its neighbours). Returns (samples (ch, n) float32, rate, t0, layout)."""
    from ravidata.align import detect_tones
    import av
    from scipy.signal import butter, sosfiltfilt
    chunks, rate, t0, layout = [], None, None, None
    with av.open(str(src)) as c:
        st = c.streams.audio[0]
        for fr in c.decode(st):
            if t0 is None:
                t0 = float(fr.time or 0.0)
                rate, layout = fr.sample_rate, fr.layout.name
            a = fr.to_ndarray().astype(np.float32)
            if fr.format.is_packed:
                a = a.reshape(-1, len(fr.layout.channels)).T
            chunks.append(a)
    x = np.concatenate(chunks, axis=1)
    if np.issubdtype(chunks[0].dtype, np.integer):
        x /= 32768.0
    sos = butter(6, [lo, hi], btype="bandpass", fs=rate, output="sos")
    y = sosfiltfilt(sos, x, axis=1).astype(np.float32)
    gate = np.zeros(x.shape[1], np.float32)
    for ts, dur, _ in detect_tones(x.mean(0), rate):
        a, b = max(0, int((ts - pad) * rate)), min(gate.size, int((ts + dur + pad) * rate))
        gate[a:b] = 1.0
    fade = max(1, int(0.01 * rate))
    gate = np.convolve(gate, np.ones(fade, np.float32) / fade, mode="same")
    return (y * gate).astype(np.float32), rate, t0, layout


def _audio_stream(o, rate: int, layout: str):
    from fractions import Fraction
    sa = o.add_stream("aac", rate=rate)
    sa.layout = layout
    sa.bit_rate = 96000
    sa.time_base = Fraction(1, rate)
    sa.codec_context.time_base = Fraction(1, rate)
    return sa


def _write_audio(o, sa, y: np.ndarray, rate: int, t0: float, layout: str):
    import av
    from fractions import Fraction
    n = 1024
    for k in range(0, y.shape[1], n):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(y[:, k:k + n]), format="fltp", layout=layout)
        fr.sample_rate = rate
        fr.time_base = Fraction(1, rate)
        fr.pts = int(round(t0 * rate)) + k
        for p in sa.encode(fr):
            o.mux(p)
    for p in sa.encode():
        o.mux(p)


# ================================================================================================ export
def _scrub_imu(src: Path, dst: Path):
    """The Sensor Logger zip -> a folder of its CSVs (one per sensor, as exported), without the device id and
    without empty files. A folder, not a zip: readable directly (pandas, the HF file viewer), nothing to unpack."""
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zi:
        for it in zi.infolist():
            if it.is_dir() or not it.filename.lower().endswith(".csv"):
                continue
            data = zi.read(it.filename)
            if not data.strip():
                continue
            if it.filename == "Metadata.csv":
                lines = data.decode("utf-8").splitlines()
                head = lines[0].split(",")
                drop = [k for k, h in enumerate(head) if h.strip().lower() in ("device id",)]
                lines = [",".join(x for k, x in enumerate(ln.split(",")) if k not in drop) for ln in lines]
                data = ("\n".join(lines) + "\n").encode("utf-8")
            (dst / Path(it.filename).name).write_bytes(data)


def _radar_files(name: str, folder: Path, work: Path) -> list[dict]:
    """Every radar recording of a take -> radar/NN_<part>_<config>.h5 (decoded cube + times + config)."""
    import h5py

    from ravidata.align import Take, _cubes_for, load_take_radar
    recs = []
    tj = folder / "take.json"
    if tj.exists():
        recs = json.loads(tj.read_text(encoding="utf-8"))["recordings"]
    else:                                     # pilot: the rows of TCC_TAKES.csv inside the video's time span
        import csv
        rows = list(csv.DictReader((ROOT / "runs" / "TCC_TAKES.csv").open(encoding="utf-8")))
        recs = [{"part": "walk" if "walking" in r["step"] else "chairs", "config": r["config"],
                 "run": Path(r["run_dir"].replace("\\", "/")).name, "t_beep_start": r["t_beep_start"],
                 "t_beep_end": r["t_beep_end"], "loss_pct": r["loss_pct"], "status": r["status"],
                 "frames": r["frames"], "lead_in_s": 0.0, "step": r["step"]}
                for r in rows if r["time"].startswith("2026-11-13T16:0") or r["time"].startswith("2026-11-13T16:10")]
    rd = work / "radar"
    if rd.exists():
        shutil.rmtree(rd)                     # no stale files from an earlier export
    rd.mkdir()
    out = []
    for k, r in enumerate(recs, 1):
        if not r.get("run") or not int(float(r.get("frames") or 0)):
            continue
        run_dir = ROOT / "runs" / r["run"]
        t0, t1 = float(r["t_beep_start"]), float(r["t_beep_end"])
        cubes = _cubes_for(run_dir, t0, t1)
        R = load_take_radar(Take(r["run"], run_dir, r["config"], t0, t1, {}, cubes))
        fn = f"{k:02d}_{r.get('part') or 'free'}_{r['config']}.h5"
        with h5py.File(rd / fn, "w") as f:
            f.create_dataset("cube", data=R["cube"], compression="gzip", compression_opts=4, chunks=True)
            for key in ("t_host", "frame_index", "range_m"):
                f.create_dataset(key, data=np.asarray(R[key]))
            if R.get("v_axis") is not None:
                f.create_dataset("v_axis", data=np.asarray(R["v_axis"]))
            cfg = R["cfg"]
            f.attrs["config_json"] = json.dumps(cfg.to_dict())
            f.attrs["derived"] = "\n".join(cfg.derive().summary_lines())
            meta = {"part": r.get("part"), "config": r["config"], "t_beep_start": t0, "t_beep_end": t1,
                    "lead_in_s": r.get("lead_in_s", 0.0), "loss_pct": r.get("loss_pct"), "status": r.get("status"),
                    "source_run": r["run"], "source_file": cubes}
            f.attrs["recording_json"] = json.dumps(meta)
            f.attrs["cube_axes"] = "frame, Doppler bin (2D) or interval (1D), RX (RX0..RX2), range bin"
        out.append({"file": f"radar/{fn}", **{k2: meta[k2] for k2 in ("part", "config", "t_beep_start",
                                                                      "t_beep_end", "lead_in_s", "loss_pct",
                                                                      "status", "source_run", "source_file")},
                    "frames": int(len(R["t_host"])), "cube_shape": list(R["cube"].shape[1:])})
    return out


def _sync(name: str, video: Path, imu: Path | None, radar_recs: list[dict]) -> dict:
    """Clock alignment of this take's video + IMU with its radar recordings (ravidata.align):

    * beeps heard in the video ↔ beep times of the recordings → ``t_host = video_a * t_video + video_b``;
    * IMU lag from gyro vs video motion → ``t_host = t_phone + imu_lag``;
    * radar check: radar motion vs video motion per recording → the extra delay of the video (default 0.56 s, pilot take);
    * SYNC_CORRECTIONS: values measured later, after the export (see there).
    ``recommended_video_offset_s`` is added to ``video_b`` (and to the IMU lag, which is measured against the video).
    """
    from ravidata.align import Take, auto_align, check_with_radar, load_sensorlogger, load_take_radar, scan_video
    scan = scan_video(video, use_cache=True)
    takes = [Take(r["file"], ROOT / "runs" / r["source_run"], r["config"], float(r["t_beep_start"]),
                  float(r["t_beep_end"]), {}, r["source_file"]) for r in radar_recs]
    al = auto_align(scan, load_sensorlogger(imu) if imu else None, takes)
    check_with_radar(al, scan, {t.run_id: load_take_radar(t) for t in takes})
    good = [(v["lag_ms"], v["corr"]) for v in al.radar_check.values() if v["corr"] >= 0.4]
    if good:
        w = np.array([c for _, c in good])
        rec = float(np.sum(w * np.array([lag for lag, _ in good])) / w.sum()) / 1000.0
        why = f"radar-vs-video motion, {len(good)} recording(s) with corr >= 0.4 (weighted mean)"
    else:
        rec, why = DEFAULT_OFFSET_S, DEFAULT_BECAUSE
    return correct_sync(name, {"video_a": al.video_a, "video_b": al.video_b, "beeps_matched": al.video_matches,
            "imu_lag_s": al.imu_lag, "imu_corr": al.imu_corr, "imu_method": al.imu_method,
            "radar_check": al.radar_check, "recommended_video_offset_s": round(rec, 3),
            "recommended_because": why, "notes": al.notes,
            "how_to": "t_host(video frame) = video_a * t_video + video_b + recommended_video_offset_s; "
                      "t_host(IMU sample) = t_phone + imu_lag_s + recommended_video_offset_s"})


def export(only: str | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    rig = json.loads((ROOT / "runs" / "RIG.json").read_text(encoding="utf-8"))["rig"]
    rig = {k: v for k, v in rig.items()}
    (OUT / "rig.json").write_text(json.dumps({"rig": rig, "notes": RIG_NOTES}, indent=1), encoding="utf-8")
    cal = OUT / "calibration"
    cal.mkdir(exist_ok=True)
    shutil.copy2(RECS / CALIB["folder"] / CALIB["json"], cal / "camera.calib.json")
    png = (RECS / CALIB["folder"] / CALIB["json"]).with_suffix(".png")
    if png.exists():
        shutil.copy2(png, cal / "camera.calib.png")
        blur_still(cal / "camera.calib.png")     # made from the original (unblurred) video
    write_apriltags()
    index = []
    for name, (fold, kind, desc) in TAKES.items():
        if only not in (None, name):
            continue
        folder = RECS / fold
        work = _work(name)
        print(f"export {name} …", flush=True)
        z = _zip_of(folder)
        if z is not None:
            _scrub_imu(z, work / "imu")
        recs = _radar_files(name, folder, work)
        video = work / "video.mp4"
        # the sync comes from the ORIGINAL video (full sound; same frames and frame times as the published one)
        sync = _sync(name, _video_of(folder), work / "imu" if z else None, recs)
        info = {"dataset_version": DATASET_VERSION, "name": name, "kind": kind, "description": desc, "source_folder": fold, "recordings": recs,
                "video": "video.mp4" if video.exists() else None, "imu": "imu/" if z else None,
                "sync": sync}
        tj = folder / "take.json"
        if tj.exists():
            t = json.loads(tj.read_text(encoding="utf-8"))
            info.update({k: t.get(k) for k in ("take_id", "started", "scene", "scene_title", "parts", "place",
                                               "configs", "rounds", "seconds_per_part", "sync_lead_in_s", "notes")})
            info["rig"] = t.get("rig")
        (work / "take.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        index.append({k: info.get(k) for k in ("name", "kind", "description")} | {"recordings": len(recs)})
    if only is None:
        (OUT / "takes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        write_card()
    print(f"dataset written to {OUT}")


def _config_rows() -> list[str]:
    from ravidata.config import presets
    rows = ["| config | mode | range bin | max range | frame period (fps) | cube per frame |", "|---|---|---|---|---|---|"]
    for n in ("tcc-1d-40mm", "tcc-2d-40mm-12fps", "tcc-2d-40mm"):
        c = presets()[n]
        d = c.derive()
        mode = "1D range (4 chirps averaged on chip)" if c.fft_mode != 2 else "2D range-Doppler (16 bins, ±2 m/s, 0.25 m/s)"
        rows.append(f"| `{n}` | {mode} | {d.bin_mm / 10:.0f} cm | {c.range_max_mm / 1000:.2f} m | "
                    f"{c.frame_period_ms} ms ({1000 / c.frame_period_ms:.1f}) | {d.doppler_len} × 3 RX × {d.range_fft_len} |")
    return rows


def _take_rows(root: Path = OUT) -> list[str]:
    rows = ["| take | radar | parts | recordings | video | description |", "|---|---|---|---|---|---|"]
    for name, (_, kind, desc) in TAKES.items():
        tj = root / "takes" / name / "take.json"
        if not tj.exists():
            continue
        t = json.loads(tj.read_text(encoding="utf-8"))
        parts = ", ".join(dict.fromkeys(r["part"] for r in t["recordings"]))
        v = root / "takes" / name / "video.mp4"
        vid = f"{v.stat().st_size / 1e9:.2f} GB" if v.exists() else "—"
        rows.append(f"| `{name}` | {'still' if kind == 'static' else 'carried'} | {parts} | {len(t['recordings'])} | "
                    f"{vid} | {desc} |")
    return rows


def neutralise():
    """The published text files name no company / internal tool: ravidata -> ravidata; fails on anything else left."""
    words = ["neo" + "senti", "ns" + "-radar", "bia" + "-radar", "assis" + "tiv", r"\bca" + r"nes?\b"]   # split: kept as is by the public export
    bad = re.compile("|".join(words), re.I)
    for p in sorted(OUT.rglob("*")):
        if p.suffix not in (".json", ".md") or not p.is_file():
            continue
        t = p.read_text(encoding="utf-8")
        t2 = t.replace("ns" + "radar", "ravidata")
        if t2 != t:
            p.write_text(t2, encoding="utf-8")
        m = bad.search(t2)
        if m:
            raise SystemExit(f"{p}: '{m.group(0)}' in a published file")


def write_apriltags():
    """apriltags.json = the printed tags (tags.json) + where each used tag was (tags_used_2026-11-13.json)."""
    tg = ROOT / "docs" / "protocols" / "targets"
    info = json.loads((tg / "tags.json").read_text(encoding="utf-8"))
    used = json.loads((tg / "tags_used_2026-11-13.json").read_text(encoding="utf-8"))
    info["used"] = {"date": used["date"], "note": used["note"], "tags": used["tags"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "apriltags.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")


def render_card(root: Path = OUT) -> str:
    return CARD.format(configs="\n".join(_config_rows()), takes="\n".join(_take_rows(root)))


def write_card(root: Path = OUT):
    (root / "README.md").write_text(render_card(root), encoding="utf-8")


CARD = """---
license: cc-by-nc-4.0
pretty_name: "RAVIDATA: 60 GHz radar, visual and inertial dataset"
task_categories:
- depth-estimation
- object-detection
tags:
- mmwave-radar
- fmcw
- radar
- imu
- apriltag
- obstacle-detection
size_categories:
- 10K<n<100K
---

# RAVIDATA: RAdar, VIsual and inertial DATAset

60 GHz FMCW radar, phone video and IMU, recorded together. Licence: **CC BY-NC 4.0** (attribution, no commercial
use). Tools to record, align and analyse it: <https://github.com/zRafaF/ravidata> (`make data` downloads
this dataset; `make pod` makes the maps on a GPU).

Recordings of a **Possumic RS6130 60 GHz FMCW radar** (MRS6130-P1812 board, 1 TX / 3 RX) next to a **phone camera**
(Samsung SM-S936W, 1440 × 1440, 30 fps, Log profile) and the **phone's IMU** (Sensor Logger, 100 Hz), for research on
obstacle detection with a radar on a moving, hand-held platform (bachelor thesis, UTFPR, 2026). Recorded on 2026-11-13 in Brazil, indoors and
outdoors.

- Radar still: an empty scene, then placed objects, then a person walking (tags 0 chest, 1 back) and a broom (tags 20, 21, 50, 51) swept
  through the field of view, two backgrounds; a pilot with three chairs.
- Radar + phone carried along an indoor and an outdoor course with AprilTags on the floor (10, 12,
  17) as metric checkpoints.
- Every scene is recorded with three radar configurations, one after the other, filmed by one continuous video.

## Takes

{takes}

## Radar configurations

{configs}

The chip runs the range FFT (and in 2D the Doppler FFT); each frame is the complex cube after those FFTs. The 2D
configs lose 5–15 % of the frames on the SPI link (mostly single frames); every frame carries its own host time
(`t_host`), so use it rather than assuming a constant rate.

## Files

```
README.md               this card
rig.json                mounting: radar rotation, antenna pairs, camera offset
apriltags.json          tag36h11 IDs, printed sizes (mm) and where each used tag was ("used")
calibration/
  camera.calib.json     intrinsics from the checkerboard clip ("display" block = the upright 1440 × 1440 video)
  camera.calib.png      corners used + an undistorted frame (faces blurred)
  checkerboard.mp4      the calibration clip (faces blurred)
takes/<take>/
  video.mp4             phone video: upright, faces blurred, original frame times and sound; HEVC 10-bit, Log
  imu/                  Sensor Logger export: one CSV per sensor (gyroscope, accelerometer, gravity, orientation, ...;
                        device id removed from Metadata.csv)
  take.json             what was recorded, every radar recording, the clock alignment ("sync")
  radar/NN_<part>_<config>.h5
      cube          (frames, Doppler bins or intervals, 3 RX, range bins) complex64
      t_host        (frames,) host clock, Unix seconds
      frame_index   (frames,) the radar's own frame counter (gaps = lost frames)
      range_m       (range bins,)        v_axis (Doppler bins,) m/s, 2D only
      attrs: config_json (the radar configuration), derived, recording_json (part, beeps, loss, lead-in)
derived/
  maps-v1/              feed-forward 3D maps of the two carried takes (GPU pod, `make pod` in the code repository)
    <model>/<take>/win_NNN.npz, run.json   per window of 64 (or 48) keyframes: poses, depth, confidence, K
    frames_<take>.json  the keyframes given to the models (video frame, time, flow, sharpness), K, tone curve
    fused/<model>/<take>/  global map made on a CPU (`make fuse`): poses.npz (pose graph + ICP, per keyframe
                        camera-to-world), graph.json, fused.npz / fused.ply (TSDF points)
    logs/, metrics/     every log and running metric of the pod (GPU, time, memory, energy)
  static-depth-v1/<model>/<take>/   monocular metric depth of the still takes (CPU; Depth Pro `depthpro`,
                        Depth Anything V2 Small `da2s`), scaled on the AprilTags: .npz depth, .jpg frame, .ply, summary.json
```

The derived files are the exact inputs of the published analysis (`results/analysis-v2` in the code repository).
`fused/` and `static-depth-v1/` can be remade with `make fuse` / `make static-depth`, but not bit for bit (a re-run of
the pose graph on another machine moved the keyframe poses by 5 cm median, up to 0.8 m), so use the uploaded ones to
reproduce the published numbers.

Parts: `S0` empty scene, `S1` objects placed, `S2` things moving (person, broom), `D1` / `D2` the indoor / outdoor
course, `chairs` / `walk` in the pilot. The first recording of a take starts with a **sync lead-in** (`lead_in_s`,
8 s): a hand swipe or a tilt of the rig to the floor, not part of the scene.

## AprilTags

tag36h11; size = edge of the black square. Only these IDs were placed (the others in `apriltags.json` were printed
but not used):

| IDs | on | size |
|---|---|---|
| 0, 1 | the person: 0 chest, 1 back | 150 mm |
| 10, 12, 17 | the floor | 90 mm |
| 11, 13, 15 | the chairs | 90 mm |
| 20, 21, 50, 51 | the broom | 55 mm (20, 21), 40 mm (50, 51) |

## Time alignment

All radar frames are on the host clock (`t_host`). The recorder beeps (1.2 kHz) once when a recording starts and
twice when it stops; the beeps are found in the video's sound and matched to the recordings' beep times
(`take.json` → `sync`):

```
t_host(video frame) = video_a * t_video + video_b + recommended_video_offset_s
t_host(IMU sample)  = t_phone + imu_lag_s + recommended_video_offset_s
```

`recommended_video_offset_s` is the remaining delay between the logged beep time and the video. For the still takes
it comes from radar-vs-video motion (`recommended_because` says which recordings); for `outdoor_course` it is the
default 0.56 s (measured on the pilot take), which the radar floor return agrees with
(+0.4 .. +0.6 s). For `indoor_course` the default was wrong: since v1.1 the offset is **−2.44 s** (the video was
3.0 s late), measured from the radar floor return against the camera height, the radar range against the map and
tag ranges, and the radar Doppler against the walking speed from the map (details and the old value in
`take.json` → `sync` → `recommended_because`, `corrections`). The beep fit itself is good to about one video frame;
check the offset on your use.

## Camera

The video is the phone's **Log** profile (BT.2020, flat): convert it (e.g. a Log-to-Rec.709 LUT or a tone curve)
before using vision models. Intrinsics: `calibration/camera.calib.json` → `display` (fx ≈ fy ≈ 1281 px, centre ≈
(724, 712), k1 0.075, k2 −0.065), valid for the upright 1440 × 1440 video. Autofocus was not locked; the focal
length is known to about ±1.5 %.

## Known limitations

- Radar height above the ground and the camera's dy / dz offsets were not measured (estimate them from the ground
  return and the IMU pitch, or from the video).
- The pilot (`pilot_chairs`) and the calibration clip are muted (the pilot may contain conversation); their clock
  alignment was computed from the original sound.
- One pass per config per course: configurations are compared on similar, not identical, motion.
- No lidar or motion-capture ground truth: positions come from AprilTags (and from vision models you run).
- The live radar angle shown during recording was uncalibrated; calibrate the RX phase offsets before using angles.
  The code repository fits them on the walking person of the still takes (azimuth: about 3° median error between
  takes); elevation stays approximate (the targets span only a few degrees).

## Privacy

Faces are blurred (YuNet detector on every frame, boxes kept ±8 frames, pixelated and blurred). The Sensor Logger
device id is removed; no location is stored. The calibration clip and the pilot are muted. Elsewhere the published
sound contains **only the beeps**: band-passed to
1.0–1.4 kHz (zero-phase) and kept ±0.1 s around each beep found in the original, silence elsewhere. The clock
alignment in `take.json` was computed from the original videos (full sound, identical frames and frame times);
the beep onsets in the published sound match the originals within a few ms.

## Versions

| version | date | change |
|---|---|---|
| v1.0 | 2026-11-14 | recordings, calibration, `derived/maps-v1` (pod) |
| v1.1 | 2026-11-15 | `indoor_course` video offset 0.56 → −2.44 s (`take.json`, with the old value in `sync.corrections`); `outdoor_course` `recommended_because` reworded (value unchanged); `derived/maps-v1/fused` and `derived/static-depth-v1` added; this card: derived files, time alignment, versions |

`hf download DoninhaD/ravidata --repo-type dataset --revision v1.0` gets the dataset as it was before v1.1 (the
inputs of the code repository's `results/analysis-v1`, apart from `fused/` and `static-depth-v1`, which were made
locally then).

## Loading

```python
import json, h5py, av
take = json.load(open("takes/indoor_course/take.json"))
rec = take["recordings"][0]
with h5py.File("takes/indoor_course/" + rec["file"]) as f:
    cube, t_host, r = f["cube"][:], f["t_host"][:], f["range_m"][:]
s = take["sync"]
with av.open("takes/indoor_course/video.mp4") as c:
    for fr in c.decode(video=0):
        t = s["video_a"] * fr.time + s["video_b"] + s["recommended_video_offset_s"]   # host clock
        ...
```
"""


RIG_NOTES = ("Radar MRS6130-P1812 (Possumic RS6130, 60 GHz FMCW, 1 TX / 3 RX) on a selfie-stick tripod; module "
             "rotated so that pin 1 is bottom right seen from the front (180°): RX2-RX1 horizontal pair (azimuth), "
             "RX1-RX0 vertical pair (elevation). Phone (Samsung SM-S936W) portrait, lens 4 cm from the radar chip "
             "(dx); dy, dz and the radar height were not measured.")


def upload(repo: str, private: bool):
    from huggingface_hub import HfApi
    cal_png = OUT / "calibration" / "camera.calib.png"
    if cal_png.exists():                          # made from the original video: blur faces before publishing
        print(f"faces blurred in {cal_png.name}: {blur_still(cal_png)}")
    write_apriltags()
    write_card()                      # the card as in this script (the code link may be newer than the export)
    neutralise()
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    ignore = ["**/faces.json", "**/faces_qa.jpg", "**/blurred_qa.jpg", "**/*.part.mp4", "**/*.scan.npz",
              "**/_check*", "**/imu.zip"]                 # imu.zip: older exports (now imu/)
    if hasattr(api, "upload_large_folder"):        # huggingface_hub < 2
        api.upload_large_folder(folder_path=str(OUT), repo_id=repo, repo_type="dataset", ignore_patterns=ignore)
    else:                                           # >= 2: upload_folder batches large folders and resumes
        api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="dataset", ignore_patterns=ignore,
                          commit_message="RAVIDATA v1: 2026-11-13 recordings")
    print(f"https://huggingface.co/datasets/{repo}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["faces", "blur", "export", "all", "upload"])
    ap.add_argument("--only", help="one name: calibration, " + ", ".join(TAKES))
    ap.add_argument("--max-frames", type=int, help="test on the first N frames")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--audio", choices=["beeps", "copy", "none"], default=None,
                    help="beeps = only the sync beeps (default); none = muted (default for calibration and "
                         "pilot_chairs, see AUDIO); copy = the original sound. The sync is always computed from the "
                         "original video, whatever is published.")
    ap.add_argument("--repo")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "upload":
        if not a.repo:
            raise SystemExit("--repo <user>/<name>")
        return upload(a.repo, a.private)
    for name, src in _sources(a.only):
        if a.cmd in ("faces", "all"):
            detect_faces(name, src, a.max_frames)
        if a.cmd in ("blur", "all"):
            blur_video(name, src, a.max_frames, a.crf, a.preset, a.audio or AUDIO.get(name, "beeps"))
    if a.cmd in ("export", "all"):
        export(a.only if a.only != "calibration" else None)


if __name__ == "__main__":
    main()
