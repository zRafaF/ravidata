"""F02: numbers of the measurement rig for the native drawing (no raster). Height and pitch are medians over the
carried indoor take (camera height from the 3D-map truth, pitch from the phone's gravity sensor)."""
from __future__ import annotations

import json

import numpy as np

from .. import common as C, geometry as G, imu, radar as R, truth as TR
from . import common as PC

TAKE = "indoor_course"
ANT_SPACING_MM = 2.45          # RX spacing of the module (d / lambda = R.D_OVER_LAMBDA at 60 GHz band centre)


def run():
    T = C.load_table("truth_frames")
    g = T[T["take"] == TAKE]
    h = float(np.nanmedian(-g.floor_h.to_numpy()))
    pitch = float("nan")
    gv = imu.at(TAKE, g.t.to_numpy(), "Gravity", 0.3)
    if gv is not None:
        # camera z (forward) component of 'up' is negative when the boresight points below the horizon
        pitch = float(np.nanmedian(-np.degrees(np.arcsin(np.clip(G.up_from_gravity(gv)[:, 2], -1, 1)))))
    rig = json.loads((C.DATA / "rig.json").read_text(encoding="utf-8"))["rig"]
    PC.save_json("f02_arranjo", dict(
        take=TAKE, cam_height_m=h, pitch_down_deg=pitch, cam_dx_cm=float(rig["cam_dx_cm"]),
        ant_spacing_mm=ANT_SPACING_MM, d_over_lambda=R.D_OVER_LAMBDA,
        cam_half_fov_deg=G.HFOV_HALF_DEG, beam_half_deg=TR.ARC_HALF_DEG,
        pairs=dict(az=list(R.PAIRS["az"]), el=list(R.PAIRS["el"])), n_tx=1, n_rx=3))
    print(h, pitch)


if __name__ == "__main__":
    run()
