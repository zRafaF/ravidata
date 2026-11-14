"""How the radar module is mounted, and which RX pairs end up horizontal and vertical.

Geometry [DS + DERIVED, verify with the hand test in the recorder]:

* RS6130 datasheet Fig. 5-1 (top view of the chip, pin-1 dot top left): RX2 above RX1, RX0 to the right of RX1
  (2.45 mm each), TX at the far top right. The chip's own axes: X to the left, Z up, Y out of the chip (boresight).
* MRS6130-P1812 module guide Fig. 3-1 (top view of the module, pins 1-8 on the left, the module's pin-1 dot top
  left): the chip sits rotated 90 degrees clockwise, its pin-1 corner at the top right. So on the module, seen from
  the FRONT (looking at the antennas), with the module's pin-1 dot top left: RX2 is to the right of RX1, RX0 is
  below RX1, TX is down and to the right.

The mounting is described by where the module's pin-1 dot is when you look at the radar face from the front:
top left = 0, top right = 90, bottom right = 180, bottom left = 270 (degrees, clockwise).
All coordinates here are seen from the FRONT: x to the right, y up, in mm, RX1 at the origin.
"""

from __future__ import annotations

import math

D_MM = 2.45
# module pin-1 dot top left (rotation 0), seen from the front
ANT_MM_ROT0 = {"RX1": (0.0, 0.0), "RX2": (D_MM, 0.0), "RX0": (0.0, -D_MM), "TX": (2.7, -5.9)}
ROTATIONS = {0: "top left", 90: "top right", 180: "bottom right", 270: "bottom left"}
RX_INDEX = {"RX0": 0, "RX1": 1, "RX2": 2}


def _rot_cw(x: float, y: float, deg: int) -> tuple[float, float]:
    a = math.radians(deg)
    return (x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a))


def antenna_positions(rotation_deg: int) -> dict[str, tuple[float, float]]:
    """Antenna centres seen from the front (x right, y up, mm), for the given mounting rotation."""
    return {k: tuple(round(v, 3) for v in _rot_cw(x, y, rotation_deg)) for k, (x, y) in ANT_MM_ROT0.items()}


def pin1_corner(rotation_deg: int) -> str:
    return ROTATIONS[int(rotation_deg) % 360]


def pairs(rotation_deg: int) -> dict[str, tuple[str, str]]:
    """{"horizontal": (a, b), "vertical": (a, b)} with b to the right of / above a, seen from the front."""
    pos = antenna_positions(rotation_deg)
    out = {}
    for other in ("RX2", "RX0"):
        dx, dy = pos[other][0] - pos["RX1"][0], pos[other][1] - pos["RX1"][1]
        if abs(dx) > abs(dy):
            out["horizontal"] = ("RX1", other) if dx > 0 else (other, "RX1")
        else:
            out["vertical"] = ("RX1", other) if dy > 0 else (other, "RX1")
    return out


def describe(rotation_deg: int) -> str:
    """One line for the session file and the GUI."""
    pr = pairs(rotation_deg)
    h, v = pr["horizontal"], pr["vertical"]
    return (f"module pin-1 dot {pin1_corner(rotation_deg)} (seen from the front): "
            f"horizontal pair {h[0]}-{h[1]} ({h[1]} to the right, seen from the front) = azimuth; "
            f"vertical pair {v[0]}-{v[1]} ({v[1]} above) = elevation")
