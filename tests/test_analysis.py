"""Unit tests of the analysis building blocks on synthetic data (no dataset needed)."""

import numpy as np
import pytest

pytest.importorskip("pandas")

from analysis import cordic, detectors as D, geometry as G, score as SC
from analysis.cluster import dbscan
from analysis.grid import Grid


def test_cordic_matches_float():
    th = np.linspace(-np.pi, np.pi, 1001)
    s, c = cordic.sincos(th, 24, 31)
    assert np.abs(s - np.sin(th)).max() < 1e-6 and np.abs(c - np.cos(th)).max() < 1e-6
    a, m = cordic.atan2_mag(np.sin(th) * 1e-5, np.cos(th) * 1e-5, 24, 31)
    assert np.abs(np.angle(np.exp(1j * (a - th)))).max() < 1e-5
    assert np.allclose(m, 1e-5, rtol=1e-5)


def test_cfar_finds_a_target():
    rng = np.random.default_rng(0)
    P = rng.exponential(1e-7, size=(50, 128))
    P[:, 60] += 1e-5
    for kind in ("ca", "go", "so"):
        m = P > D.cfar_1d(P, kind, 2, 8, 10.0)
        assert m[:, 60].all() and m.sum() < 50 * 3
    m = P > D.cfar_1d(P, "os", 2, 8, 10.0, 0.75)
    assert m[:, 60].all()


def test_mofn():
    m = np.zeros((6, 5), bool)
    m[[0, 2, 3], 2] = True
    out = D.mofn(m, 2, 3, 0)
    assert out[2, 2] and out[3, 2] and not out[0, 2]


def test_dbscan():
    X = np.r_[np.random.default_rng(1).normal(0, 0.02, (10, 2)), [[5, 5]]]
    lab = dbscan(X, 0.1, 3)
    assert (lab[:10] == 0).all() and lab[10] == -1


def test_sectors_and_gospa():
    obs_r = np.full((1, G.N_SECT), np.nan)
    obs_az = np.full((1, G.N_SECT), np.nan)
    obs_r[0, 7], obs_az[0, 7] = 2.0, 2.0
    d = dict(f=np.array([0, 0], np.int32), r=np.array([2.05, 1.0]), az=np.array([2.0, -20.0]),
             el=np.zeros(2), score=np.array([10.0, 5.0]), snr_db=np.array([10.0, 5.0]))
    fr, dt = SC.score_rec(d, obs_r, obs_az, np.full_like(obs_r, np.nan), np.array([True]), np.array([2.0]),
                          np.array([True]))
    assert fr["tp10"][0] == 1 and fr["n_det"][0] == 2
    assert fr["gospa"][0] == pytest.approx(np.sqrt(0.05 ** 2 + 0.5 ** 2 / 2), rel=1e-6)


def test_grid_accumulates_and_moves():
    g = Grid({"decay": 1.0, "free": False})
    for _ in range(3):
        g.update(np.array([2.0]), np.array([0.0]), np.array([20.0]))
    r, az, sc = g.obstacles()
    assert len(r) and abs(r.min() - 2.0) < 0.15
    g.move(0.0, 0.5, 0.0)                                  # 0.5 m forward: the obstacle is 0.5 m nearer
    r2, _, _ = g.obstacles()
    assert abs(r2.min() - 1.5) < 0.15
