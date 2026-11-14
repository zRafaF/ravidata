"""Small DBSCAN (scipy cKDTree) and the temporal DBSCAN filter on detections."""

from __future__ import annotations

import numpy as np

from . import geometry as G


def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Labels (-1 = noise) like sklearn.cluster.DBSCAN."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(X)
    lab = -np.ones(n, int)
    if n == 0:
        return lab
    tr = cKDTree(X)
    pairs = tr.query_pairs(eps, output_type="ndarray")
    deg = np.ones(n, int) + np.bincount(pairs.ravel(), minlength=n)
    core = deg >= min_samples
    if not core.any():
        return lab
    cc = pairs[core[pairs[:, 0]] & core[pairs[:, 1]]]
    A = coo_matrix((np.ones(len(cc)), (cc[:, 0], cc[:, 1])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    lab[core] = comp[core]
    for a, b in pairs:                                     # border points join a neighbouring core's cluster
        if core[a] and not core[b] and lab[b] < 0:
            lab[b] = lab[a]
        elif core[b] and not core[a] and lab[a] < 0:
            lab[a] = lab[b]
    u, lab[lab >= 0] = np.unique(lab[lab >= 0], return_inverse=True)
    return lab


def temporal_dbscan(d: dict, n_frames: int, window: int = 5, eps: float = 0.25, min_samples: int = 3,
                    use_az: bool = True) -> dict:
    """For every frame, DBSCAN over the detections of the last `window` frames (radar plane X, Z; range only if
    use_az is False) and report, per cluster that has a point in the current frame, its nearest point."""
    f = d["f"]
    if len(f) == 0:
        return d
    X, Z = G.plane(d["r"], d["az"] if use_az else np.zeros_like(d["r"]))
    pts = np.stack([X, Z], 1) if use_az else d["r"][:, None]
    starts = np.searchsorted(f, np.arange(n_frames + 1))
    keep = []
    for i in range(n_frames):
        a, b = starts[max(0, i - window + 1)], starts[i + 1]
        cur0 = starts[i]
        if b == cur0:
            continue
        lab = dbscan(pts[a:b], eps, min_samples)
        lc = lab[cur0 - a:]
        idx = np.arange(cur0, b)
        for c in np.unique(lc[lc >= 0]):
            j = idx[lc == c]
            keep.append(j[np.argmin(d["r"][j])])
    keep = np.array(sorted(keep), int)
    return {k: v[keep] for k, v in d.items()}
