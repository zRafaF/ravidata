"""Detectors on Feat: threshold, CFAR family (CA / GO / SO / OS) along range, 2D range-Doppler CFAR, M-of-N
persistence. Each returns a detection table: dict of arrays f (frame), b (bin), r (m), snr_db, az, el, coh."""

from __future__ import annotations

import numpy as np

from . import geometry as G
from .radar import Feat, angles_at

EPS = 1e-300


def _window_sums(P, g, t):
    """Sums and counts of the training cells left and right of every bin (guard g, train t), (n, B) each."""
    n, B = P.shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(P, 1)], 1)
    i = np.arange(B)
    l0, l1 = np.clip(i - g - t, 0, B), np.clip(i - g, 0, B)
    r0, r1 = np.clip(i + g + 1, 0, B), np.clip(i + g + t + 1, 0, B)
    return cs[:, l1] - cs[:, l0], (l1 - l0)[None], cs[:, r1] - cs[:, r0], (r1 - r0)[None]


def cfar_1d(P, kind="ca", guard=1, train=6, k_db=10.0, os_q=0.75):
    """Threshold map (n, B) of a 1D CFAR along range (edges use the cells that exist)."""
    if kind == "os":
        n, B = P.shape
        w = guard + train
        pad = np.pad(P, ((0, 0), (w, w)), mode="reflect")
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1, axis=1)          # (n, B, 2w+1)
        idx = np.r_[0:train, train + 2 * guard + 1:2 * w + 1]
        est = np.quantile(win[:, :, idx], os_q, axis=2)
    else:
        sl, nl, sr, nr = _window_sums(P, guard, train)
        ml = sl / np.maximum(nl, 1)
        mr = sr / np.maximum(nr, 1)
        ml = np.where(nl > 0, ml, mr)
        mr = np.where(nr > 0, mr, ml)
        est = {"ca": (sl + sr) / np.maximum(nl + nr, 1), "go": np.maximum(ml, mr), "so": np.minimum(ml, mr)}[kind]
    return est * 10 ** (k_db / 10)


def cfar_rd(RD, guard=(1, 1), train=(2, 4), k_db=12.0):
    """2D CA-CFAR on range-Doppler maps (n, D, B); Doppler is circular. Returns the threshold map (n, D, B)."""
    from scipy.ndimage import uniform_filter
    gd, gr = guard
    td, tr = train
    full = (2 * (gd + td) + 1, 2 * (gr + tr) + 1)
    inner = (2 * gd + 1, 2 * gr + 1)
    sf = uniform_filter(RD, size=(1,) + full, mode=("nearest", "wrap", "nearest")) * np.prod(full)
    si = uniform_filter(RD, size=(1,) + inner, mode=("nearest", "wrap", "nearest")) * np.prod(inner)
    return (sf - si) / (np.prod(full) - np.prod(inner)) * 10 ** (k_db / 10)


def mofn(mask, m=2, n=3, spread=1):
    """Causal M-of-N persistence per range bin, a bin counts if any bin within +-spread fired."""
    if n <= 1 or m <= 1:
        return mask
    from scipy.ndimage import maximum_filter1d
    d = maximum_filter1d(mask.astype(np.uint8), 2 * spread + 1, axis=1) if spread else mask.astype(np.uint8)
    cs = np.concatenate([np.zeros((1, d.shape[1]), np.int32), np.cumsum(d, 0, dtype=np.int32)], 0)
    k = np.arange(1, d.shape[0] + 1)
    cnt = cs[k] - cs[np.maximum(k - n, 0)]
    return mask & (cnt >= m)


def peaks(F: Feat, mask, r_min=G.R_MIN, r_max=G.R_MAX, local_max=True, max_per_frame=16, nearest_only=False,
          refine=True):
    """Detection table from a (n, B) mask: local maxima of P inside the mask and the range gate."""
    P = F.P
    gate = (F.r >= r_min) & (F.r <= r_max)
    m = mask & gate[None]
    if local_max:
        Pl = np.pad(P, ((0, 0), (1, 1)), constant_values=0)
        m &= (P >= Pl[:, :-2]) & (P >= Pl[:, 2:])
    f, b = np.nonzero(m)
    snr = 10 * np.log10(np.maximum(P[f, b], EPS) / np.maximum(F.noise[f], EPS))
    if nearest_only and len(f):
        first = np.r_[True, f[1:] != f[:-1]]                # nonzero() is row-major: first hit = nearest bin
        f, b, snr = f[first], b[first], snr[first]
    elif max_per_frame and len(f):
        o = np.lexsort((-snr, f))
        f, b, snr = f[o], b[o], snr[o]
        rank = np.arange(len(f)) - np.searchsorted(f, f)
        k = rank < max_per_frame
        f, b, snr = f[k], b[k], snr[k]
    r = F.r[b].astype(float)
    if refine and len(f):                                   # parabolic peak interpolation on dB
        B = P.shape[1]
        ym = 10 * np.log10(np.maximum(P[f, np.clip(b - 1, 0, B - 1)], EPS))
        y0 = 10 * np.log10(np.maximum(P[f, b], EPS))
        yp = 10 * np.log10(np.maximum(P[f, np.clip(b + 1, 0, B - 1)], EPS))
        den = ym - 2 * y0 + yp
        d = np.where(np.abs(den) > 1e-9, 0.5 * (ym - yp) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
        r = r + np.clip(d, -0.5, 0.5) * (F.r[1] - F.r[0])
    az, el, coh = angles_at(F, f, b)
    return dict(f=f.astype(np.int32), b=b.astype(np.int32), r=r, snr_db=snr, az=az, el=el, coh=coh)


def threshold_mask(F: Feat, k_db=12.0):
    return F.P > F.noise[:, None] * 10 ** (k_db / 10)


def rd_mask(F: Feat, guard=(1, 1), train=(2, 4), k_db=12.0):
    if F.RD is None:
        return None
    return (F.RD > cfar_rd(F.RD, guard, train, k_db)).any(1)


def empty_dets():
    z = np.zeros(0)
    return dict(f=z.astype(np.int32), b=z.astype(np.int32), r=z, snr_db=z, az=z, el=z, coh=z)


def concat(ds):
    ds = [d for d in ds if d is not None and len(d["f"])]
    if not ds:
        return empty_dets()
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def select(d, keep):
    return {k: v[keep] for k, v in d.items()}
