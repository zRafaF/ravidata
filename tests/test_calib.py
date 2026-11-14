"""ravidata calib: synthetic checkerboard views rendered with a known camera are calibrated back to it, and the
stored -> upright pixel mapping matches np.rot90."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _render(K, rvec, tvec, size, pattern, square_mm, px_per_mm=4.0):
    """Board image (white margin of one square) warped into the camera with a plane homography."""
    nx, ny = pattern[0] + 1, pattern[1] + 1
    m = square_mm
    bw, bh = int((nx + 2) * m * px_per_mm), int((ny + 2) * m * px_per_mm)
    tex = np.full((bh, bw), 230, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0, y0 = int((i + 1) * m * px_per_mm), int((j + 1) * m * px_per_mm)
                tex[y0:y0 + int(m * px_per_mm), x0:x0 + int(m * px_per_mm)] = 25
    R, _ = cv2.Rodrigues(rvec)
    Hp = K @ np.column_stack([R[:, 0], R[:, 1], tvec])           # board plane (mm, origin = first inner corner)
    S = np.array([[1 / px_per_mm, 0, -2 * m], [0, 1 / px_per_mm, -2 * m], [0, 0, 1]])   # texture px -> board mm
    img = cv2.warpPerspective(tex, Hp @ S, size, flags=cv2.INTER_AREA, borderValue=120)
    return cv2.GaussianBlur(img, (3, 3), 0.6)


def test_synthetic_calibration():
    from ravidata.calib import VideoDetections, board, detect_frames, solve, verdict
    pattern, square = board()
    size = (720, 720)
    K = np.array([[640.0, 0, 362.0], [0, 640.0, 355.0], [0, 0, 1]])
    rng = np.random.default_rng(1)
    frames = []
    for i in range(30):
        rvec = np.radians(rng.uniform(-35, 35, 3)) * np.array([1, 1, 0.3])
        z = rng.uniform(450, 800)
        cx_mm = rng.uniform(-0.25, 0.25) * z - 100
        cy_mm = rng.uniform(-0.25, 0.25) * z - 60
        frames.append((i, i / 30, _render(K, rvec, np.array([cx_mm, cy_mm, z]), size, pattern, square)))
    det = detect_frames(frames, pattern, VideoDetections("syn", size[0], size[1], 0, 30.0, 30, 1))
    assert len(det.corners) >= 20
    r = solve(det.corners, size, "k1k2", n_boot=4)
    assert r["fx"] == pytest.approx(640, rel=0.01)
    assert r["fy"] == pytest.approx(640, rel=0.01)
    assert r["cx"] == pytest.approx(362, abs=6)
    assert r["cy"] == pytest.approx(355, abs=6)
    assert r["rms_px"] < 0.5
    assert any(line.startswith("ok") for line in verdict(r))


@pytest.mark.parametrize("rot", [0, -90, 90, 180, 270])
def test_to_display_matches_rot90(rot):
    from ravidata.calib import to_display, upright
    w, h = 7, 4
    img = np.zeros((h, w), int)
    pts = np.array([[1.0, 2.0], [6.0, 0.0], [0.0, 3.0]])
    for k, (x, y) in enumerate(pts):
        img[int(y), int(x)] = k + 1
    up = upright(img, rot)
    q, (W, H) = to_display(pts, w, h, rot)
    assert up.shape == (H, W)
    for k, (x, y) in enumerate(q):
        assert up[int(round(y)), int(round(x))] == k + 1
