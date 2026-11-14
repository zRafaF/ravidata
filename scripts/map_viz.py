"""RAVIDATA map viewer: the pod's maps in 3D, in the browser, with measuring.

    uv sync --group viz
    uv run python scripts/map_viz.py                       # then open http://localhost:8080
    uv run python scripts/map_viz.py --model pi3x --take outdoor_course --port 8081

Reads ``data/ravidata/derived/maps-v1`` (``make maps`` or the ``hf download`` in docs/pod.md): per model and take
the windows ``win_NNN.npz`` (poses, depth, confidence, intrinsics). The keyframe images used for the colours are
re-made from the course video the first time (``pod/frames.py --from-json``: the exact frames the models saw).

What it shows

- **View → Show**: the whole take chained into one map; **one window** (one batch exactly as the model returned
  it, in its own frame: is the model consistent inside a batch?); or **two windows** N and N+1 in red / blue as the
  chaining placed them (is the join between batches what breaks the map?). The panel gives each window's
  self-consistency (its depth re-projected into a frame 3 keyframes later with its own poses) and how well each
  join fits; **◀ previous / next ▶** steps through the batches; the info box lists the worst joins.
- One model and take at a time, its windows **chained** into one map: consecutive windows share keyframes; the
  scale between them is the median depth ratio on the shared frames and the rigid transform is the best fit of
  the shared camera poses. The panel reports how well the shared poses agree (a consistency figure per model).
- Points coloured by image / depth / height / window; filters: confidence (per frame percentile), maximum depth,
  flying pixels at depth edges, keyframe step and pixel stride (point budget).
- The camera path, camera frustums with their images, and optionally the paths of all models aligned to the
  current one (similarity fit on the shared keyframes) with the RMS difference.

Measuring (replaces the "close the window, read the console" flow of pano_viz.py)

- Press **Start measuring**, then click points in the scene. A click snaps to the nearest map point along the
  clicked ray. Two clicks = one measurement (or a polyline: every click adds a segment and the total grows).
- Each measurement is drawn (line, end points, label with the length) and listed in the panel with Δx Δy Δz;
  **Undo** removes the last point, **Clear** everything. Every measurement is also appended to
  ``<maps>/measurements.csv`` (model, take, points, length).
- Units: metres for the metric models (MapAnything, Depth Anything 3, Pi3X approximately); VGGT is up to an
  unknown scale. A printed AprilTag edge (40 / 55 / 90 / 150 mm, docs) is a quick scale check.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["mapanything", "pi3x", "da3", "vggt"]
METRIC = {"mapanything": "m", "pi3x": "m (approx.)", "da3": "m", "vggt": "arbitrary (up to scale)"}
MODEL_RGB = {"mapanything": (230, 85, 13), "pi3x": (49, 130, 189), "da3": (49, 163, 84), "vggt": (117, 107, 177)}


# ------------------------------------------------------------------------------------------------ geometry
def scaled(c2w: np.ndarray, s: float) -> np.ndarray:
    T = c2w.astype(np.float64).copy()
    T[..., :3, 3] *= s
    return T


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t with dst ≈ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var) if var > 1e-12 else 1.0
    return s, R, md - s * R @ ms


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def turbo(x: np.ndarray) -> np.ndarray:
    """0..1 -> RGB uint8 (OpenCV's TURBO colormap)."""
    import cv2
    v = (np.clip(x, 0, 1) * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(v, cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1]


# ------------------------------------------------------------------------------------------------ data
@dataclass
class Window:
    k: int
    idx: np.ndarray
    c2w: np.ndarray
    K: np.ndarray
    path: Path
    depth: np.ndarray | None = None
    conf: np.ndarray | None = None

    def load_maps(self):
        if self.depth is None:
            z = np.load(self.path)
            self.depth, self.conf = z["depth"], z["conf"]


@dataclass
class Chain:
    G: list[np.ndarray]                 # per window: rigid transform window (scaled) -> global
    S: list[float]                      # per window: scale window -> global
    rot_err_deg: list[float] = field(default_factory=list)   # disagreement of the shared poses after the fit
    pos_err: list[float] = field(default_factory=list)
    pt_err: list[float] = field(default_factory=list)     # "points": median residual / scene radius per pair
    inliers: list[int] = field(default_factory=list)      # shared keyframes that agreed with the chosen fit

    def pose(self, w: Window, j: int) -> np.ndarray:
        return self.G[w.k] @ scaled(w.c2w[j], self.S[w.k])


def load_windows(d: Path, with_maps: bool) -> list[Window]:
    out = []
    for k, p in enumerate(sorted(d.glob("win_*.npz"))):
        z = np.load(p)
        w = Window(k, z["idx"].astype(int), z["c2w"].astype(np.float64), z["K"].astype(np.float64), p)
        if with_maps:
            w.depth, w.conf = z["depth"], z["conf"]
        out.append(w)
    return out


def _cam_points(w: Window, j: int, stride: int = 8, conf_pct: float = 50.0):
    """(pixel mask, points in the camera frame) of keyframe j of a window, on a pixel grid."""
    d = w.depth[j][::stride, ::stride].astype(np.float32)
    c = w.conf[j][::stride, ::stride].astype(np.float32)
    ok = (d > 0) & np.isfinite(d) & (c >= np.percentile(c, conf_pct))
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W] * stride
    K = w.K[j]
    pc = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1)
    return ok, pc


def _fit_rigid(Tg: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Rigid G with G @ Tb ≈ Tg (mean rotation, then translation of the camera centres)."""
    R = mean_rotation(Tg[:, :3, :3] @ np.transpose(Tb[:, :3, :3], (0, 2, 1)))
    G = np.eye(4); G[:3, :3] = R
    G[:3, 3] = (Tg[:, :3, 3] - Tb[:, :3, 3] @ R.T).mean(0)
    return G


def _pose_diff(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.linalg.norm(A[:, :3, 3] - B[:, :3, 3], axis=1)
    rot = np.array([rot_angle_deg(x[:3, :3].T @ y[:3, :3]) for x, y in zip(A, B)])
    return pos, rot


def chain_windows(wins: list[Window], free: bool = False, method: str = "poses") -> Chain:
    """Chain consecutive windows through their shared keyframes into one map, robust to bad keyframes.

    Every shared keyframe proposes a transform window b -> global: method "points" a similarity fitted to that
    frame's dense 3D points (the same pixels seen by both windows); method "poses" the scale from the median depth
    ratio and the rigid transform of that frame's two camera poses. The proposal that most other shared frames
    agree with (camera within 5 % of the scene depth and 5°) wins; the transform is then refitted on those inlier
    frames only (dense points with trimming, or the poses). A single keyframe that a model got wrong (seen in the
    data: one frame turned 65° in one window and 135° in the other) no longer breaks the join.
    free=True: depth maps are loaded pair by pair and dropped again (paths only)."""
    ch = Chain([np.eye(4)], [1.0])
    for a, b in zip(wins[:-1], wins[1:]):
        a.load_maps(); b.load_maps()
        shared, ia, ib = np.intersect1d(a.idx, b.idx, return_indices=True)
        if len(shared) == 0:                 # no overlap: keep the previous transform (should not happen)
            ch.G.append(ch.G[-1].copy()); ch.S.append(ch.S[-1]); ch.pt_err.append(float("nan"))
            ch.pos_err.append(float("nan")); ch.rot_err_deg.append(float("nan")); ch.inliers.append(0); continue
        same_res = a.depth.shape[1:] == b.depth.shape[1:]
        Tg = np.stack([ch.pose(a, j) for j in ia])                   # shared cameras, already global
        depth_g = float(np.median([np.median(a.depth[j][::8, ::8]) for j in ia])) * ch.S[a.k]
        tol_p, tol_r = max(0.05 * depth_g, 1e-3), 5.0
        use_pts = method == "points" and same_res
        cand, frame_pts = [], []
        if use_pts:
            for ja, jb in zip(ia, ib):
                oka, pa = _cam_points(a, ja)
                okb, pb = _cam_points(b, jb)
                m = oka & okb
                Ta = ch.pose(a, ja)
                dst = (pa[m] * ch.S[a.k]) @ Ta[:3, :3].T + Ta[:3, 3]
                src = pb[m] @ b.c2w[jb][:3, :3].T + b.c2w[jb][:3, 3]
                frame_pts.append((src, dst))
                if len(src) >= 50:
                    sf, Rf, tf_ = umeyama(src, dst)
                    G = np.eye(4); G[:3, :3] = Rf; G[:3, 3] = tf_
                    cand.append((sf, G))
        else:
            s_rel = 1.0
            if same_res:
                da = a.depth[ia][:, ::4, ::4].astype(np.float32)
                db = b.depth[ib][:, ::4, ::4].astype(np.float32)
                m = (da > 0) & (db > 0) & np.isfinite(da) & np.isfinite(db)
                if m.sum() > 100:
                    s_rel = float(np.median(da[m] / db[m]))
            sf = ch.S[a.k] * s_rel
            Tb = scaled(b.c2w[ib], sf)
            for q in range(len(ia)):
                cand.append((sf, _fit_rigid(Tg[q:q + 1], Tb[q:q + 1])))
        best, best_in, best_err = None, None, np.inf
        for sf, G in cand:                                          # consensus over the shared frames
            pos, rot = _pose_diff(G @ scaled(b.c2w[ib], sf), Tg)
            inl = (pos <= tol_p) & (rot <= tol_r)
            err = float(np.median(pos[inl])) if inl.any() else np.inf
            if best is None or inl.sum() > best_in.sum() or (inl.sum() == best_in.sum() and err < best_err):
                best, best_in, best_err = (sf, G), inl, err
        if best is None:
            ch.G.append(ch.G[-1].copy()); ch.S.append(ch.S[-1]); ch.pt_err.append(float("nan"))
            ch.pos_err.append(float("nan")); ch.rot_err_deg.append(float("nan")); ch.inliers.append(0); continue
        sf, G = best
        if best_in.sum() >= 2:                                      # refit on the inlier frames
            if use_pts:
                src = np.concatenate([frame_pts[q][0] for q in np.flatnonzero(best_in)])
                dst = np.concatenate([frame_pts[q][1] for q in np.flatnonzero(best_in)])
                keep = np.ones(len(src), bool)
                for _ in range(3):
                    sf, R, t = umeyama(src[keep], dst[keep])
                    r = np.linalg.norm(sf * src @ R.T + t - dst, axis=1)
                    keep = r <= np.percentile(r, 80)
                G = np.eye(4); G[:3, :3] = R; G[:3, 3] = t
                ch.pt_err.append(float(np.median(r) / (np.median(np.linalg.norm(dst - dst.mean(0), axis=1)) + 1e-9)))
            else:
                G = _fit_rigid(Tg[best_in], scaled(b.c2w[ib], sf)[best_in])
                ch.pt_err.append(float("nan"))
        else:
            ch.pt_err.append(float("nan"))
        ch.G.append(G); ch.S.append(float(sf)); ch.inliers.append(int(best_in.sum()))
        pos, rot = _pose_diff(np.stack([ch.pose(b, j) for j in ib]), Tg)
        ch.pos_err.append(float(np.median(pos[best_in])) if best_in.any() else float(np.median(pos)))
        ch.rot_err_deg.append(float(np.median(rot[best_in])) if best_in.any() else float(np.median(rot)))
        if free:
            a.depth = a.conf = None
    if free and wins:
        wins[-1].depth = wins[-1].conf = None
    return ch


def owners(wins: list[Window]) -> dict[int, tuple[int, int]]:
    """Each keyframe once: from the window where it is most central (most context on both sides)."""
    best: dict[int, tuple[int, int, int]] = {}
    for w in wins:
        n = len(w.idx)
        for j, f in enumerate(w.idx):
            c = min(j, n - 1 - j)
            if f not in best or c > best[f][2]:
                best[f] = (w.k, j, c)
    return {f: (k, j) for f, (k, j, _) in best.items()}


def ensure_frames(maps: Path, data: Path, take: str) -> Path | None:
    """The keyframe JPEGs (re-made once from the course video, exactly the pod's frames)."""
    out = maps / "frames" / take
    if (out / "frames.json").exists():
        return out / "rgb"
    ref, video = maps / f"frames_{take}.json", data / "takes" / take / "video.mp4"
    if not ref.exists() or not video.exists():
        print(f"no keyframe images for {take} ({ref.name} or the video missing): colours by depth")
        return None
    print(f"re-making the keyframe images of {take} from {video} (once, a few minutes) ...", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "pod" / "frames.py"), "--data", str(data), "--take", take,
                        "--out", str(out), "--from-json", str(ref)])
    return out / "rgb" if r.returncode == 0 else None


# ------------------------------------------------------------------------------------------------ viewer
def window_consistency(w: Window, gap: int = 3, stride: int = 8) -> float:
    """Inside one window: frame i's depth re-projected into frame i+gap with the window's own poses, compared
    with frame i+gap's depth; median relative difference (0.01 = 1 %). Small = the window agrees with itself."""
    n, H, W = w.depth.shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    errs = []
    for i in range(0, n - gap, 2):
        j = i + gap
        d = w.depth[i][vv, uu].astype(np.float32)
        m = d > 0
        u, v, z = uu[m], vv[m], d[m]
        Ki, Kj = w.K[i], w.K[j]
        pc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
        pw = pc @ w.c2w[i][:3, :3].T + w.c2w[i][:3, 3]
        w2c = np.linalg.inv(w.c2w[j])
        pj = pw @ w2c[:3, :3].T + w2c[:3, 3]
        ok = pj[:, 2] > 1e-6
        uj = np.round(pj[ok, 0] / pj[ok, 2] * Kj[0, 0] + Kj[0, 2]).astype(int)
        vj = np.round(pj[ok, 1] / pj[ok, 2] * Kj[1, 1] + Kj[1, 2]).astype(int)
        inb = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
        dj = w.depth[j][vj[inb], uj[inb]].astype(np.float32)
        zz = pj[ok, 2][inb]
        g = dj > 0
        if g.sum() > 50:
            errs.append(float(np.median(np.abs(zz[g] - dj[g]) / dj[g])))
    return float(np.median(errs)) if errs else float("nan")


VIEWS = ["whole take (chained)", "one window", "two windows (overlap)", "fused map (TSDF)"]
CHAINING = {"camera poses": "poses", "dense points": "points", "pose graph (map_fuse)": "graph"}
PAIR_TINT = [np.array([255, 70, 50]), np.array([40, 140, 255])]


class Viewer:
    def __init__(self, maps: Path, data: Path, port: int, model: str | None = None, take: str | None = None):
        import viser
        self.maps, self.data = maps, data
        self.takes = sorted({p.name for m in MODELS for p in (maps / m).glob("*") if p.is_dir()})
        self.models = [m for m in MODELS if (maps / m).is_dir()]
        if not self.takes or not self.models:
            raise SystemExit(f"no maps in {maps} (download them first: docs/pod.md, 'Results')")
        self.server = viser.ViserServer(port=port)
        self.server.gui.configure_theme(control_width="large")
        self.lock = threading.Lock()
        self.cache: dict[tuple[str, str], tuple[list[Window], Chain]] = {}
        self.wcons: dict[tuple[str, str, int], float] = {}
        self.pts = np.zeros((0, 3), np.float32)
        self.col = np.zeros((0, 3), np.uint8)
        self.meas: list[list[np.ndarray]] = [[]]        # measurements: lists of picked points
        self.measuring = False
        self.up = np.array([0.0, 0.0, 1.0])
        self._take0 = take if take in self.takes else self.takes[0]
        self._model0 = model if model in self.models else ("pi3x" if "pi3x" in self.models else self.models[0])
        self.center, self.extent = np.zeros(3), 1.0
        self._gui()

    # ------------------------------------------------------------------ GUI
    def _gui(self):
        g = self.server.gui
        with g.add_folder("Data"):
            self.g_take = g.add_dropdown("Take", self.takes, initial_value=self._take0)
            self.g_model = g.add_dropdown("Model", self.models, initial_value=self._model0)
            self.g_info = g.add_markdown("")
        with g.add_folder("View"):
            self.g_view = g.add_dropdown("Show", VIEWS, initial_value=VIEWS[0],
                                         hint="one window = the model's raw output for that batch, in its own frame; "
                                              "two windows = window N and N+1 placed by the chaining (red / blue)")
            self.g_win = g.add_slider("Window", 0, 99, 1, 0)
            self.g_step = g.add_button_group("Step", ["◀ previous", "next ▶"])
            self.g_chain = g.add_dropdown("Chaining", list(CHAINING), initial_value="camera poses",
                                          hint="how consecutive windows are joined: similarity fit on the dense points "
                                               "of the shared keyframes, or scale from depth + the shared camera poses")
            self.g_follow = g.add_checkbox("Re-centre the view", True)
            self.g_fuse = g.add_button("Fuse this model + take (TSDF)", color="violet",
                                       hint="loop closures + pose graph + ICP + weighted TSDF on the CPU (~2-3 min); "
                                            "same as  make fuse MODEL=<model> TAKE=<take>")
            self.g_win_md = g.add_markdown("")
        with g.add_folder("Points"):
            self.g_kstep = g.add_slider("Keyframe step", 1, 10, 1, 2, hint="use every N-th keyframe")
            self.g_stride = g.add_slider("Pixel stride", 1, 16, 1, 6, hint="every N-th pixel of each depth map")
            self.g_conf = g.add_slider("Drop low confidence %", 0, 90, 5, 30,
                                       hint="per frame: drop the pixels below this confidence percentile")
            self.g_dmax = g.add_slider("Max depth", 0.5, 20.0, 0.5, 8.0, hint="model units (m for metric models)")
            self.g_edges = g.add_checkbox("Remove flying pixels", True, hint="drop pixels at depth discontinuities")
            self.g_color = g.add_dropdown("Colour", ["image", "depth", "height", "window"], initial_value="image")
            self.g_psize = g.add_slider("Point size", 0.002, 0.3, 0.002, 0.01,
                                        hint="world units; far, sparse scenes (outdoor) need large points")
            self.g_apply = g.add_button("Apply")
        with g.add_folder("Cameras"):
            self.g_path = g.add_checkbox("Camera path", True)
            self.g_frust = g.add_checkbox("Frustums", True)
            self.g_fstep = g.add_slider("Frustum every N keyframes", 1, 50, 1, 15)
            self.g_thumbs = g.add_checkbox("Images in frustums", True)
        with g.add_folder("Compare models"):
            self.g_cmp = g.add_checkbox("Paths of all models", False,
                                        hint="aligned to the current model (similarity fit on the shared keyframes)")
            self.g_cmp_info = g.add_markdown("")
        with g.add_folder("Measure"):
            self.g_mode = g.add_dropdown("Mode", ["two points", "polyline"], initial_value="two points")
            self.g_measure = g.add_button("Start measuring", color="green")
            with g.add_folder("Edit", expand_by_default=True):
                self.g_undo = g.add_button("Undo last point")
                self.g_clear = g.add_button("Clear all")
            self.g_meas_md = g.add_markdown("Click **Start measuring**, then click points in the scene.")

        self.g_take.on_update(lambda _: self.reload())
        self.g_model.on_update(lambda _: self.reload())
        self.g_view.on_update(lambda _: self.redraw_view())
        self.g_win.on_update(lambda _: self.redraw_view())
        self.g_step.on_click(lambda e: self._step(e.target.value))
        self.g_chain.on_update(lambda _: self.rechain())
        self.g_fuse.on_click(lambda _: threading.Thread(target=self.fuse, daemon=True).start())
        self.g_apply.on_click(lambda _: self.redraw_points())
        self.g_color.on_update(lambda _: self.redraw_points())
        self.g_psize.on_update(lambda _: self._set_psize())
        for h in (self.g_path, self.g_frust, self.g_fstep, self.g_thumbs):
            h.on_update(lambda _: self.redraw_cameras())
        self.g_cmp.on_update(lambda _: self.redraw_compare())
        self.g_measure.on_click(lambda _: self.toggle_measure())
        self.g_undo.on_click(lambda _: self.undo())
        self.g_clear.on_click(lambda _: self.clear_meas())
        self.g_mode.on_update(lambda _: self.clear_meas())

        @self.server.on_client_connect
        def _(client):
            self._place_camera(client)

    def fuse(self):
        """Run scripts/map_fuse.py for the current model + take (reusing the loaded windows), then show it."""
        import map_fuse
        if getattr(self, "_fusing", False) or not hasattr(self, "wins"):
            return
        self._fusing = True
        model, take = self.g_model.value, self.g_take.value
        lines: list[str] = []

        def log(msg):
            print(msg, flush=True)
            lines.append(msg.strip())
            self.g_win_md.content = "**fusing** " + f"{model} / {take}  \n" + "  \n".join(lines[-8:])
        try:
            self.g_fuse.disabled = True
            map_fuse.run(self.maps, self.data, model, take, wins=self.wins, log=log)
            self.g_view.value = VIEWS[3]                 # show the result
            self.redraw_view()
        except Exception as e:                           # keep the viewer alive
            log(f"fusion failed: {e!r}")
        finally:
            self._fusing = False
            self.g_fuse.disabled = False

    def _step(self, which: str):
        n = len(getattr(self, "wins", [])) or 1
        if self.g_view.value == VIEWS[0]:
            self.g_view.value = VIEWS[1]                 # stepping makes sense per window
        self.g_win.value = int(np.clip(self.g_win.value + (1 if "next" in which else -1), 0, n - 1))

    def _place_camera(self, client):
        try:
            pos = self.center + self.up * self.extent * 0.6 + self._side() * self.extent
            client.camera.position = tuple(float(x) for x in pos)   # position first: setting it moves look_at
            client.camera.look_at = tuple(float(x) for x in self.center)
        except Exception:
            pass

    def _side(self) -> np.ndarray:
        a = np.cross(self.up, [1.0, 0.0, 0.0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(self.up, [0.0, 1.0, 0.0])
        return a / np.linalg.norm(a)

    # ------------------------------------------------------------------ loading
    def method(self) -> str:
        return CHAINING.get(self.g_chain.value, "poses")

    def fused_dir(self, model: str, take: str) -> Path:
        return self.maps / "fused" / model / take

    def _chain(self, wins, model, take, free=False) -> Chain:
        """Robust chaining; with "pose graph", the window similarities optimised by scripts/map_fuse.py replace
        the chained ones (the join diagnostics stay those of the robust chaining)."""
        m = self.method()
        ch = chain_windows(wins, free=free, method="poses" if m == "graph" else m)
        f = self.fused_dir(model, take) / "poses.npz"
        if m == "graph":
            if not f.exists():
                print(f"no pose graph for {model}/{take} yet: run  make fuse MODEL={model} TAKE={take}")
                return ch
            z = np.load(f)
            if len(z["window_s"]) == len(wins):
                ch.S = [float(x) for x in z["window_s"]]
                ch.G = []
                for R, t in zip(z["window_R"], z["window_t"]):
                    G = np.eye(4); G[:3, :3] = R; G[:3, 3] = t
                    ch.G.append(G)
        return ch

    def border(self, shape) -> np.ndarray:
        """Valid-pixel mask (drops the black undistortion border the models were given)."""
        if getattr(self, "_mask518", None) is None:
            try:
                import map_fuse
                self._mask518 = map_fuse.valid_mask(self.data, 518)
            except Exception as e:                       # no calibration: keep everything
                print(f"no border mask ({e})")
                self._mask518 = np.ones((518, 518), bool)
        import map_fuse
        return map_fuse.mask_for(self._mask518, shape)

    def get(self, model: str, take: str, with_maps: bool = True):
        key = (model, take)
        if key in self.cache and (not with_maps or self.cache[key][0][0].depth is not None):
            return self.cache[key]
        if key in self.cache:                           # chained before for the paths only: load the maps now
            wins, ch = self.cache[key]
            for w in wins:
                w.load_maps()
            self.cache = {k: v for k, v in self.cache.items() if v[0][0].depth is None or k == key}
            return wins, ch
        wins = load_windows(self.maps / model / take, with_maps)
        ch = self._chain(wins, model, take, free=not with_maps)
        if with_maps:                                   # keep one full model in memory
            self.cache = {k: v for k, v in self.cache.items() if v[0][0].depth is None}
        self.cache[key] = (wins, ch)
        return wins, ch

    def reload(self):
        with self.lock:
            model, take = self.g_model.value, self.g_take.value
            self.g_info.content = f"loading **{model}** / {take} ..."
            t0 = time.time()
            self.wins, self.chain = self.get(model, take)
            self.own = owners(self.wins)
            self.rgb_dir = ensure_frames(self.maps, self.data, take)
            self.run = json.loads((self.maps / model / take / "run.json").read_text(encoding="utf-8"))
            fj = self.maps / f"frames_{take}.json"
            self.tvid = ({f["i"]: f.get("t_video", 0.0) for f in json.loads(fj.read_text(encoding="utf-8"))["frames"]}
                         if fj.exists() else {})
            try:
                self.g_win.max = len(self.wins) - 1
            except Exception:
                pass
            if self.g_win.value > len(self.wins) - 1:
                self.g_win.value = 0
            self.t_load = time.time() - t0
            self._info()
            self.clear_meas()
        self.redraw_view()
        self.redraw_compare()

    def rechain(self):
        with self.lock:
            if not hasattr(self, "wins"):
                return
            self.chain = self._chain(self.wins, self.g_model.value, self.g_take.value)
            self.cache[(self.g_model.value, self.g_take.value)] = (self.wins, self.chain)
            self._info()
        self.redraw_view()
        self.redraw_compare()

    def _info(self):
        model, take, ch, run = self.g_model.value, self.g_take.value, self.chain, self.run
        cs = np.stack([ch.pose(self.wins[k], j)[:3, 3] for _, (k, j) in sorted(self.own.items())])
        plen = float(np.linalg.norm(np.diff(cs, axis=0), axis=1).sum())
        q = [(ch.inliers[k], -ch.pos_err[k]) for k in range(len(ch.pos_err))]
        worst = sorted(range(len(q)), key=lambda k: q[k])[:5]          # fewest agreeing frames, then largest error
        self.g_info.content = (
            f"**{model}** / {take}: {len(self.wins)} windows (size {run.get('window')}), {len(self.own)} keyframes  \n"
            f"units: {METRIC.get(model, '?')}; chained camera path {plen:.2f}  \n"
            f"chaining ({self.g_chain.value}): window scale {min(ch.S):.3f}–{max(ch.S):.3f}; shared poses agree to "
            f"{np.nanmedian(ch.pos_err) if ch.pos_err else 0:.3f} / "
            f"{np.nanmedian(ch.rot_err_deg) if ch.rot_err_deg else 0:.2f}° (median, agreeing frames)  \n"
            f"worst joins (window → next: position / rotation): "
            + ", ".join(f"{k}→{k + 1}: {ch.pos_err[k]:.2f} / {ch.rot_err_deg[k]:.0f}° ({ch.inliers[k]} agree)"
                        for k in worst) + "  \n"
            f"pod: {run.get('seconds')} s, peak {run.get('peak_vram_gb')} GB, "
            f"{run.get('fps_model')} frames/s ({run.get('gpu', '')}); loaded in {getattr(self, 't_load', 0):.1f} s")

    # ------------------------------------------------------------------ what is shown
    def items(self) -> list[tuple[int, int, int, np.ndarray, float, int]]:
        """(keyframe, window, index in window, pose, scale, colour tag) of every frame shown in the current view."""
        view, step = self.g_view.value, int(self.g_kstep.value)
        if view == VIEWS[3]:
            f = self.fused_dir(self.g_model.value, self.g_take.value) / "poses.npz"
            if not f.exists():
                return []
            z = np.load(f)
            out = []
            for kf, T, sc in zip(z["keyframes"], z["c2w"], z["depth_scale"]):
                k, j = self.own.get(int(kf), (0, 0))
                out.append((int(kf), k, j, T, float(sc), 0))
            return out[::step] if step > 1 else out
        if view == VIEWS[0]:
            out = [(f, k, j, self.chain.pose(self.wins[k], j), self.chain.S[k], k)
                   for f, (k, j) in sorted(self.own.items())]
        else:
            i = int(np.clip(self.g_win.value, 0, len(self.wins) - 1))
            ks = [i] if view == VIEWS[1] else [k for k in (i, i + 1) if k < len(self.wins)]
            out = []
            for tag, k in enumerate(ks):
                w = self.wins[k]
                for j, f in enumerate(w.idx):
                    if view == VIEWS[1]:                # raw output of the window, its own frame
                        out.append((int(f), k, j, w.c2w[j], 1.0, tag))
                    else:
                        out.append((int(f), k, j, self.chain.pose(w, j), self.chain.S[k], tag))
        return out[::step] if step > 1 else out

    def _window_md(self):
        view = self.g_view.value
        if view == VIEWS[3]:
            g = self.fused_dir(self.g_model.value, self.g_take.value) / "graph.json"
            if not g.exists():
                self.g_win_md.content = (f"_no fused map yet: run_ `make fuse MODEL={self.g_model.value} "
                                         f"TAKE={self.g_take.value}`")
                return
            info = json.loads(g.read_text(encoding="utf-8"))
            lb = [e for e in info["edges_before"] if e["kind"] == "loop"]
            la = [e for e in info["edges_after"] if e["kind"] == "loop"]
            txt = (f"**fused map**: {info['points']:,} points, voxel {info['voxel_m']} m, depth ≤ {info['zmax']}, "
                   f"{info['keyframes']} keyframes  \n{len(info['closures'])} loop closures")
            if lb:
                txt += (f"; they disagreed by {np.median([e['trans'] for e in lb]):.2f} / "
                        f"{np.median([e['rot_deg'] for e in lb]):.1f}° before the pose graph, "
                        f"{np.median([e['trans'] for e in la]):.2f} / {np.median([e['rot_deg'] for e in la]):.1f}° after")
            self.g_win_md.content = txt
            return
        if view == VIEWS[0]:
            self.g_win_md.content = "_whole take: choose **one window** or **two windows** to inspect the batches_"
            return
        i = int(np.clip(self.g_win.value, 0, len(self.wins) - 1))
        rows = []
        for k in ([i] if view == VIEWS[1] else [k for k in (i, i + 1) if k < len(self.wins)]):
            w = self.wins[k]
            key = (self.g_model.value, self.g_take.value, k)
            if key not in self.wcons:
                self.wcons[key] = window_consistency(w)
            t0, t1 = self.tvid.get(int(w.idx[0]), float("nan")), self.tvid.get(int(w.idx[-1]), float("nan"))
            tint = "" if view == VIEWS[1] else (" (red)" if k == i else " (blue)")
            rows.append(f"**window {k}**{tint}: keyframes {w.idx[0]}–{w.idx[-1]}, video {t0:.1f}–{t1:.1f} s, "
                        f"self-consistency {self.wcons[key] * 100:.1f} % (depth re-projected 3 frames ahead), "
                        f"chain scale {self.chain.S[k]:.3f}")
        if view == VIEWS[2] and i + 1 < len(self.wins):
            ch = self.chain
            pe = ch.pt_err[i] if i < len(ch.pt_err) else float("nan")
            nin = ch.inliers[i] if i < len(ch.inliers) else 0
            nsh = len(np.intersect1d(self.wins[i].idx, self.wins[i + 1].idx))
            rows.append(f"**join {i}→{i + 1}**: {nin} of {nsh} shared keyframes agree; on those the poses differ by "
                        f"{ch.pos_err[i]:.3f} / {ch.rot_err_deg[i]:.2f}° after the fit"
                        + (f", dense points {pe * 100:.1f} % of the scene radius" if np.isfinite(pe) else ""))
        self.g_win_md.content = "  \n".join(rows)

    def redraw_view(self):
        if not hasattr(self, "wins"):
            return
        with self.lock:
            its = self.items()
            if not its:
                self._window_md()
                return
            poses = np.stack([T for _, _, _, T, _, _ in its])
            ys = poses[:, :3, 1]                         # camera y axis = image down
            up = -ys.mean(0)
            self.up = up / (np.linalg.norm(up) + 1e-12)
            cs = poses[:, :3, 3]
            self.center = cs.mean(0)
            self.extent = float(np.linalg.norm(cs.max(0) - cs.min(0))) or 1.0
            self.server.scene.set_up_direction(tuple(self.up))
            self._window_md()
        self.redraw_points()
        self.redraw_cameras()
        self.clear_meas()
        if self.g_follow.value:
            for c in self.server.get_clients().values():
                self._place_camera(c)
        self.redraw_compare()

    # ------------------------------------------------------------------ drawing
    def _image(self, f: int, shape) -> np.ndarray | None:
        if self.rgb_dir is None:
            return None
        import cv2
        im = cv2.imread(str(self.rgb_dir / f"{f:06d}.jpg"))
        if im is None:
            return None
        if im.shape[:2] != shape:
            im = cv2.resize(im, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        return im[..., ::-1]

    def redraw_points(self):
        if self.g_view.value == VIEWS[3]:
            return self._draw_fused()
        with self.lock:
            if not hasattr(self, "wins"):
                return
            stride = int(self.g_stride.value)
            pdrop, dmax, edges, cmode = self.g_conf.value, self.g_dmax.value, self.g_edges.value, self.g_color.value
            pair = self.g_view.value == VIEWS[2]
            P, C, D, W = [], [], [], []
            for f, k, j, T, s, tag in self.items():
                w = self.wins[k]
                d = w.depth[j].astype(np.float32)
                ok = (d > 0) & np.isfinite(d) & (d <= dmax / s) & self.border(d.shape)
                if pdrop > 0:
                    cf = w.conf[j].astype(np.float32)
                    ok &= cf >= np.percentile(cf, pdrop)
                if edges:
                    gx = np.zeros_like(d); gy = np.zeros_like(d)
                    gx[:, 1:] = np.abs(np.diff(d, axis=1)); gy[1:, :] = np.abs(np.diff(d, axis=0))
                    ok &= np.maximum(gx, gy) <= 0.04 * np.maximum(d, 1e-6) + 1e-6
                H, Wd = d.shape
                vv, uu = np.mgrid[0:H:stride, 0:Wd:stride]
                sel = ok[vv, uu]
                u, v, z = uu[sel], vv[sel], d[vv, uu][sel]
                K = w.K[j]
                pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1) * s
                P.append((pc @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
                D.append(z * s); W.append(np.full(len(z), tag))
                if cmode == "image":
                    im = self._image(f, (H, Wd))
                    C.append(im[v, u] if im is not None else None)
            if not P:
                return
            pts = np.concatenate(P)
            tags = np.concatenate(W)
            if cmode == "image" and all(c is not None for c in C):
                col = np.concatenate(C).astype(np.float32)
                if pair:                                 # tint the two windows red / blue
                    for t in (0, 1):
                        col[tags == t] = 0.55 * col[tags == t] + 0.45 * PAIR_TINT[t]
                col = col.astype(np.uint8)
            elif cmode == "window":
                col = (np.stack([PAIR_TINT[t % 2] for t in tags]).astype(np.uint8) if pair
                       else turbo((tags % 12) / 11.0))
            elif cmode == "height":
                h = pts @ self.up
                col = turbo((h - np.percentile(h, 2)) / (np.ptp(np.percentile(h, [2, 98])) + 1e-9))
            else:
                col = turbo(np.concatenate(D) / dmax)
            self.pts, self.col = pts, col
            self.server.scene.add_point_cloud("/map/points", pts, col, point_size=float(self.g_psize.value),
                                              point_shape="rounded")
            self.g_info.content = self.g_info.content.split("  \npoints shown:")[0] + f"  \npoints shown: {len(pts):,}"

    def _draw_fused(self):
        f = self.fused_dir(self.g_model.value, self.g_take.value) / "fused.npz"
        if not f.exists():
            return
        z = np.load(f)
        pts = z["xyz"].astype(np.float32)
        cmode = self.g_color.value
        if cmode == "height":
            h = pts @ self.up
            col = turbo((h - np.percentile(h, 2)) / (np.ptp(np.percentile(h, [2, 98])) + 1e-9))
        elif cmode in ("depth", "window"):                 # no per-point depth: weight (how well observed)
            w = np.log1p(z["weight"])
            col = turbo((w - w.min()) / (np.ptp(w) + 1e-9))
        else:
            col = z["rgb"]
        self.pts, self.col = pts, col
        self.server.scene.add_point_cloud("/map/points", pts, col, point_size=float(self.g_psize.value),
                                          point_shape="rounded")
        self.g_info.content = self.g_info.content.split("  \npoints shown:")[0] + f"  \npoints shown: {len(pts):,}"

    def _set_psize(self):
        if len(self.pts):
            self.server.scene.add_point_cloud("/map/points", self.pts, self.col, point_size=float(self.g_psize.value),
                                              point_shape="rounded")

    def redraw_cameras(self):
        sc = self.server.scene
        if getattr(self, "_path_h", None) is not None:
            self._path_h.remove()
            self._path_h = None
        for h in getattr(self, "_frusta", []):
            h.remove()
        self._frusta = []
        if not hasattr(self, "wins"):
            return
        its = self.items()
        if not its:
            return
        pair = self.g_view.value == VIEWS[2]
        cs = np.stack([T[:3, 3] for _, _, _, T, _, _ in its])
        if self.g_path.value and len(cs) > 1:
            seg = np.stack([cs[:-1], cs[1:]], 1).astype(np.float32)
            self._path_h = sc.add_line_segments("/map/path", seg, (255, 200, 0), thickness=3, thickness_units="screen")
        if self.g_frust.value:
            import viser.transforms as tf
            K = self.wins[0].K[0]
            H = self.wins[0].depth.shape[1] if self.wins[0].depth is not None else 518
            fov = 2 * np.arctan(H / 2 / K[1, 1])
            scale = max(self.extent * 0.03, 0.03)
            fstep = int(self.g_fstep.value) if self.g_view.value in (VIEWS[0], VIEWS[3]) else max(1, int(self.g_fstep.value) // 4)
            for n in range(0, len(its), fstep):
                f, k, j, T, s, tag = its[n]
                img = None
                if self.g_thumbs.value:
                    im = self._image(f, (H, H))
                    if im is not None:
                        import cv2
                        img = cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)
                color = tuple(int(x) for x in PAIR_TINT[tag]) if pair else (255, 200, 0)
                self._frusta.append(sc.add_camera_frustum(
                    f"/map/cams/{n:05d}", fov=float(fov), aspect=1.0, scale=scale, image=img, color=color,
                    wxyz=tf.SO3.from_matrix(T[:3, :3]).wxyz, position=T[:3, 3]))

    def redraw_compare(self):
        for h in getattr(self, "_cmp", []):
            h.remove()
        self._cmp = []
        if not self.g_cmp.value or not hasattr(self, "wins") or self.g_view.value != VIEWS[0]:
            self.g_cmp_info.content = ""
            return
        take, cur = self.g_take.value, self.g_model.value
        ref_f = np.array(sorted(self.own))
        ref_c = np.stack([self.chain.pose(self.wins[k], j)[:3, 3] for _, (k, j) in sorted(self.own.items())])
        lines = []
        for m in self.models:
            if m == cur or not (self.maps / m / take).is_dir():
                continue
            wins, ch = self.get(m, take, with_maps=False)
            own = owners(wins)
            f = np.array(sorted(own))
            c = np.stack([ch.pose(wins[k], j)[:3, 3] for _, (k, j) in sorted(own.items())])
            common, ia, ib = np.intersect1d(ref_f, f, return_indices=True)
            if len(common) < 3:
                continue
            s, R, t = umeyama(c[ib], ref_c[ia])
            ca = (s * c @ R.T + t).astype(np.float32)
            rms = float(np.sqrt(((ca[ib] - ref_c[ia]) ** 2).sum(1).mean()))
            seg = np.stack([ca[:-1], ca[1:]], 1)
            self._cmp.append(self.server.scene.add_line_segments(
                f"/cmp/{m}", seg, MODEL_RGB.get(m, (200, 200, 200)), thickness=3, thickness_units="screen"))
            lines.append(f"- <span style='color:rgb{MODEL_RGB.get(m)}'>■</span> **{m}**: RMS to {cur} "
                         f"{rms:.3f} after scale ×{s:.3f}")
        self.g_cmp_info.content = (f"current: <span style='color:rgb(255,200,0)'>■</span> **{cur}**  \n"
                                   + "\n".join(lines))

    # ------------------------------------------------------------------ measuring
    def toggle_measure(self):
        sc = self.server.scene
        if self.measuring:
            sc.remove_pointer_callback()
            self.measuring = False
            self._button("Start measuring", "green")
            self.draw_meas()
            return
        self.measuring = True
        self._button("Stop measuring", "red")
        self.draw_meas()

        @sc.on_pointer_event("click")
        def _(ev):
            p = self.pick(np.asarray(ev.ray_origin, float), np.asarray(ev.ray_direction, float))
            if p is None:
                self.g_meas_md.content = "no map point near that click; try again (zoom in for small things)"
                return
            self.add_point(p)

    def _button(self, label: str, color: str):
        for k, v in (("label", label), ("color", color)):
            try:
                setattr(self.g_measure, k, v)
            except Exception:
                pass

    def pick(self, o: np.ndarray, d: np.ndarray) -> np.ndarray | None:
        """The map point nearest the camera among those within a small angle of the clicked ray."""
        if len(self.pts) == 0:
            return None
        d = d / np.linalg.norm(d)
        v = self.pts.astype(np.float64) - o
        t = v @ d
        perp = np.sqrt(np.maximum((v * v).sum(1) - t * t, 0))
        ok = (t > 0) & (perp < np.maximum(0.004 * t, 0.002))     # ~0.25° cone, at least 2 mm
        if not ok.any():
            ok = (t > 0) & (perp < np.maximum(0.012 * t, 0.005))
            if not ok.any():
                return None
        cand = np.flatnonzero(ok)
        near = cand[t[cand] <= t[cand].min() * 1.02 + 1e-6]      # the front surface, not points behind it
        return self.pts[near[np.argmin(perp[near])]].astype(np.float64)

    def add_point(self, p: np.ndarray):
        cur = self.meas[-1]
        cur.append(p)
        if self.g_mode.value == "two points" and len(cur) == 2:
            self.save(cur)
            self.meas.append([])
        elif self.g_mode.value == "polyline" and len(cur) >= 2:
            self.save(cur[-2:], total=cur)
        self.draw_meas()

    def save(self, seg, total=None):
        f = self.maps / "measurements.csv"
        new = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["time", "model", "take", "mode", "x1", "y1", "z1", "x2", "y2", "z2", "length",
                             "polyline_total", "units"])
            a, b = seg
            tot = sum(np.linalg.norm(total[i + 1] - total[i]) for i in range(len(total) - 1)) if total else ""
            wr.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), self.g_model.value, self.g_take.value,
                         self.g_mode.value, *np.round(a, 4), *np.round(b, 4), round(float(np.linalg.norm(b - a)), 4),
                         round(float(tot), 4) if tot != "" else "", METRIC.get(self.g_model.value, "?")])
        print(f"measured {np.linalg.norm(seg[1] - seg[0]):.4f} ({METRIC.get(self.g_model.value)}) -> {f}")

    def draw_meas(self):
        sc = self.server.scene
        for h in getattr(self, "_mh", []):
            h.remove()
        self._mh = []
        r = max(self.extent * 0.005, 0.015)
        unit = "m" if self.g_model.value != "vggt" else "u"
        rows = []
        for mi, pts in enumerate(self.meas):
            if not pts:
                continue
            col = (255, 60, 60) if mi % 2 == 0 else (60, 200, 255)
            for pi, p in enumerate(pts):
                self._mh.append(sc.add_icosphere(f"/meas/{mi}/p{pi}", radius=r, color=col, position=p))
            if len(pts) >= 2:
                P = np.array(pts)
                seg = np.stack([P[:-1], P[1:]], 1).astype(np.float32)
                self._mh.append(sc.add_line_segments(f"/meas/{mi}/line", seg, col, thickness=4,
                                                     thickness_units="screen"))
                for si in range(len(P) - 1):
                    L = float(np.linalg.norm(P[si + 1] - P[si]))
                    self._mh.append(sc.add_label(f"/meas/{mi}/l{si}", f"{L:.3f} {unit}",
                                                 position=(P[si] + P[si + 1]) / 2, anchor="bottom-center"))
                dv = P[-1] - P[0]
                tot = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
                up = float(dv @ self.up)
                rows.append(f"**#{mi + 1}** {'total ' if len(P) > 2 else ''}{tot:.3f} {unit} "
                            f"(vertical {up:+.3f}, horizontal {np.sqrt(max(dv @ dv - up * up, 0)):.3f})")
        live = self.meas[-1]
        hint = ("click the second point" if (self.g_mode.value == "two points" and len(live) == 1)
                else "click to add a point" if self.measuring else "press Start measuring")
        self.g_meas_md.content = ("\n\n".join(rows) or "no measurement yet") + f"  \n_{hint}; units: " \
            f"{METRIC.get(self.g_model.value)}; saved to measurements.csv_"

    def undo(self):
        if self.meas[-1]:
            self.meas[-1].pop()
        elif len(self.meas) > 1:
            self.meas.pop()
            self.meas[-1].pop()
        self.draw_meas()

    def clear_meas(self):
        self.meas = [[]]
        self.draw_meas()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "ravidata"), help="the downloaded dataset")
    ap.add_argument("--tag", default="v1", help="derived/maps-<tag>")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--take")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    data = Path(a.data)
    v = Viewer(data / "derived" / f"maps-{a.tag}", data, a.port, a.model, a.take)
    v.reload()
    url = f"http://localhost:{a.port}"
    print(f"\nRAVIDATA map viewer: {url}  (Ctrl-C to quit)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
