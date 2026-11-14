"""Frame container shared by all sources, the recorder, processing and the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cube import CubeShape


@dataclass
class Frame:
    seq: int                         # host-side running counter (0, 1, 2, ...)
    frame_index: int                 # radar frame index from the HIF payload header
    t_host: float                    # time.time() when the frame was completed
    t_mono: float                    # time.monotonic() when completed
    cube: np.ndarray                 # complex64 (n_dop, n_rx, n_range), normalised to +-1 full scale
    shape: CubeShape
    words: np.ndarray | None = None  # raw uint32 DWORDs as received (None for legacy replays)
    source: str = ""
    extra: dict = field(default_factory=dict)   # e.g. simulator ground truth
