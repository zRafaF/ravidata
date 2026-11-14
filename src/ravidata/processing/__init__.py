"""Signal processing: shared pre-processing plus pluggable candidate detectors.

Add a new candidate by subclassing :class:`~ravidata.processing.detectors.Detector` and
registering it in ``DETECTORS``. Every registered detector runs on every frame, so live
and replayed data compare all candidates on identical input.
"""

from .base import ParamSpec  # noqa: F401
from .detectors import DETECTORS, Detection, Detector, DetectorOutput  # noqa: F401
from .pipeline import Features, Preprocess, Processor, ProcessResult  # noqa: F401
