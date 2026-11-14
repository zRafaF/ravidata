"""Antenna geometry vs mounting rotation (RS6130 Fig. 5-1 + MRS6130-P1812 Fig. 3-1, seen from the front)."""

from ravidata.orientation import antenna_positions, describe, pairs


def test_rotation_0_module_as_in_its_guide():
    pos = antenna_positions(0)
    assert pos["RX2"][0] > pos["RX1"][0] and abs(pos["RX2"][1] - pos["RX1"][1]) < 1e-6    # RX2 right of RX1
    assert pos["RX0"][1] < pos["RX1"][1] and abs(pos["RX0"][0] - pos["RX1"][0]) < 1e-6    # RX0 below RX1
    assert pairs(0) == {"horizontal": ("RX1", "RX2"), "vertical": ("RX0", "RX1")}


def test_rotations():
    assert pairs(90) == {"horizontal": ("RX0", "RX1"), "vertical": ("RX2", "RX1")}
    assert pairs(180) == {"horizontal": ("RX2", "RX1"), "vertical": ("RX1", "RX0")}
    # pin-1 dot bottom left: RX0 right of RX1 and RX2 above RX1, like the thesis' fig. 10 (RD-61)
    assert pairs(270) == {"horizontal": ("RX1", "RX0"), "vertical": ("RX1", "RX2")}
    assert "RX1-RX0" in describe(270) and "bottom left" in describe(270)
