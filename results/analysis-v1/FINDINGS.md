# RAVIDATA radar analysis v1: findings

Status: first full run (2026-11-14, commit 3a2478d). Numbers come from `REPORT.md` (generated from the tables).
Figures are in `figures/` (`paper_*` = paper candidates, `story/` = explanatory). The run log is `runs.jsonl`.
Everything here is preliminary until the open questions at the end are settled.

## 1. What is being measured

**Truth.** The truth is the obstacle geometry from the camera, expressed in the radar frame at every radar frame
(`figures/story/truth_*.png`):

- **Carried takes:** Pi3X keyframe depth, made metric by the AprilTags (tag ÷ map depth 1.04 indoors, 0.93
  outdoors). "Up" comes from the phone's gravity sensor.
  - The floor is a RANSAC plane near the densest low level.
  - An obstacle is anything 0.12–2.0 m above that floor.
  - The arc truth uses a local map: the keyframes of the same map window within ±4 s, placed with that window's own
    poses. The fused map is not used for truth: its joins leave ghost surfaces, and its nearest obstacle came out
    0.76 m nearer than the keyframe depth.
- **Static takes:** Depth Pro depth, scaled on the tags (scale, or scale + offset, whichever fits all tags better).
  The tagged objects are added on top. People are placed frame by frame.

**Three views of "detected", from strict to lenient:**

1. **Per direction** (`f1_50`): one obstacle per 4° sector inside the camera view, Hungarian matching at 0.5 m.
   This needs angles the radar does not have; use it only as a stress test.
2. **Nearest in the camera view** (`near_*`): the nearest detection whose own angle falls in the camera view,
   compared with the camera truth.
3. **Arc view** (`arc_*`, carried takes only): the distance to the nearest obstacle anywhere in the beam (±60°
   azimuth, ±30° elevation), direction ignored. This is the handheld use case and the main metric from here on.
   - Found: within 0.25 m.
   - Early alarm: something reported more than 0.25 m nearer than the truth, or something reported when there is
     nothing there.

## 2. The sensor

- **Frame timing.**
  - The 1D config delivers 33.6 fps with 0.13 % frames lost and 3 ms jitter.
  - The 2D configs lose 9–10 % of frames, with 20–27 ms jitter and 99th-percentile gaps of 119–168 ms.
  - The link or host, not the chip, is the weak point of the 2D modes (`REPORT.md`, frame timing).
- **Noise floor.** −55 dB of full scale for the 1D config, −60 / −67 dB for the two 2D configs (16 chirps
  integrated). Static returns sit at −42 to −45 dB.
- **Range.** Range bias is +0.117 m over 39 still targets, with a median residual of 4 cm.
- **Angles.**
  - Azimuth from the RX2–RX1 pair is only moderately consistent: R 0.80, median residual 8°.
  - Elevation looks better (R 0.91, 5°), but the targets only span −16° to −3°.
  - Two receivers give one angle per range bin, a power-weighted mix of everything at that distance.

## 3. Main result: the arc view (nearest obstacle distance)

`figures/paper_operating_curves.png` and `REPORT.md` ("equal early-alarm budgets"):

| early-alarm budget | best "found" over all families | spread between families |
|---|---|---|
| 5 % | 0.12–0.13 | 0.06–0.13 |
| 10 % | 0.17 | 0.13–0.17 |
| 20 % | 0.29 | 0.16–0.29 |
| 30 % | 0.36–0.44 | 0.21–0.44 |

- **At equal early-alarm budgets the detector families are close.** CA-CFAR and CFAR + DBSCAN are on top, the
  threshold, OS-CFAR and our grid are within a few points, and the legacy energy grid is lowest.
- **The choice of detector is not the bottleneck; the scene geometry is.**
- **Default settings sit low on the curve.** CA-CFAR finds 0.15 with 0.11 early alarms; the threshold finds 0.21
  with 0.23.

## 4. Where it works: distance bands

`figures/paper_outcome_vs_distance_*.png` (tuned for the arc view, CA-CFAR):

| true nearest obstacle | found | early | what happens |
|---|---|---|---|
| 0.3–0.5 m | 0.12 | 0.00 | blind zone: leakage and the near cut (the tuned r_min is 0.6–0.8 m) |
| 0.5–0.75 m | 0.43 | 0.00 | partly inside the blind zone |
| 0.75–1.5 m | 0.69 | 0.03–0.30 | **the working band** |
| 1.5–2 m | 0.12 | 0.86 | an ever-present nearer return wins |
| > 2 m | 0.00 | ≈1.0 | always an early alarm |

- **When nothing is within 2 m, the radar still reports something.** Outdoors it sits at 1.21–1.37 m (IQR
  1.13–1.63 m), indoors at 0.78–1.1 m.
- **Outdoors, that band follows the "nearest floor in the beam" truth line** (`figures/story/waterfall_outdoor_*`).
  So the floor, at a range set by the radar's height and its elevation beam, is the dominant false return.
- **Indoors the band is nearer than the predicted floor.** Candidates are the carrier's own body or arm and the
  floor at steep angles; this is to be checked.
- **Consequence:** with a forward-looking radar at 0.8–1.4 m height, an obstacle beyond about 1.5 m competes with
  the floor at the same range. Separating floor from obstacle (height or elevation) is the problem, not detection.

## 5. Per-direction metrics

- **Scores are very low.** F1 at 0.5 m is 0.02–0.16 at default settings and up to 0.30 when tuned.
- **Two causes:**
  - The azimuth is unreliable (section 2).
  - One angle per range bin cannot fill 10–14 sectors of truth, so recall is capped by construction.
- **Range-only F1 is 2–4× higher** (0.08–0.54), confirming that range is the reliable measurement.
- **Use in the paper:** per-direction scores are a stress test that justifies the arc view, not a headline.

## 6. Temporal filtering

- **Our grid, temporal DBSCAN and M-of-N** cut early alarms at their default settings (grid 0.10, DBSCAN 0.08,
  against 0.11 for the base CA-CFAR). On the operating curve they do not beat CA-CFAR: they move along the curve
  instead of above it. The grid is a temporal filter; its benefit (persistence) costs latency and spreads
  returns.
- **Motion compensation does not help, even when perfect.**
  - Moving the grid with the IMU yaw plus an assumed walking speed (`grid_imu`) gives no gain.
  - Moving it with the map trajectory (`grid_oracle`, the best possible motion) is also no better.
  - The floor moves with the carrier, so compensating the motion does not remove it.
- **Legacy energy grid** (the earlier C firmware, re-implemented): it is the most sensitive at default settings
  (0.18 found) but has the worst curve, because it has no detector in front and its many low-energy cells add
  early alarms.
- **Added for the next run** (the ablation step):
  - non-coherent integration over frames before the CFAR;
  - a nearest-obstacle tracker (alpha-beta filter with confirm and hold).

## 7. Radar + IMU fusion

- **Truth:** the gravity "up" sets the floor plane and the obstacle band.
- **Grid motion:** the phone's yaw turns the grid, and the IMU's "walking" flag gates an assumed speed. The legacy
  C version also rotated its map with the orientation.
- **Height gate:** the gravity levels each detection's elevation; floor and overhead returns are dropped.
  - At default settings it is neutral to slightly negative: found −0.02 to −0.04, early alarms −0.01.
  - The tuned settings switch it on for CA-CFAR and the grid, with an elevation offset of −5° and −17°. That points
    to an elevation bias the static calibration could not see (its targets only span −16° to −3°).
  - It is the lever for the floor problem (section 4), but it needs an elevation calibration over −60° to 0°.

## 8. Was the optimisation worth it?

- **Cost:** 3.8 h on 8 workers (150 trials × 11 families × 2 objectives, pooled).
- **It generalises.**
  - Held-out takes reach arc found 0.30–0.39 against 0.08–0.31 at default settings.
  - The per-scene optimum is only about 0.04 better (CA-CFAR 0.376 against 0.419), so the price of
    universality is small.
- **But the gain is almost all movement along the same trade-off curve.**
  - Early alarms rise from 0.06–0.12 to 0.27–0.37.
  - As a "something within 2 m" alarm, the tuned CA-CFAR fires in about 100 % of frames.
  - The settings it converged to:
    - a near cut of 0.6–0.8 m;
    - EMA background removal (α 0.8–0.95);
    - coherent RX combining;
    - thresholds of 4–7 dB;
    - SO / OS-CFAR.

  The ablation (one change at a time, each over a threshold sweep) can show these with interpretable numbers, at
  a fraction of the cost. Keep the optimisation results as they are; do not rerun them.

## 9. Other facts worth a sentence in the paper

- **Cardboard is nearly transparent at 60 GHz.** The camera's box face is at 1.6 m; the radar's return is from the
  metal inside, at about 2.0 m.
- **Pi3X keyframe depth is close to metric** indoors (tag ÷ depth 1.04).
- **Monocular depth on the static takes:**
  - Depth Anything V2 Small is 1.4–2.6× too far and loses thin objects.
  - Depth Pro is near-metric (0.92–1.04) but stretches depth with range (±12 %).
- **Unplaced AprilTags** (sheets on the walls, or false detections) must be excluded: one of them pulled a scale
  estimate to 5.4.
- **Sync:** the numeric checks show no peak.
  - The range-time maps show the radar ridges on the truth lines within a few frames (for example indoor
    21–23 s), which supports the beep sync (±16 ms).
  - A tap on the rig in future recordings would close this.

## 10. Open questions (owner in the loop)

1. **Outdoor walkway roof.** The truth puts the roof 1.6–1.7 m above the walkway, with the radar 0.82 m high
   (`figures/story/truth_outdoor_*`). If the real roof is higher than about 2.2 m, the floor plane or the scale
   outdoors is off, and the roof is wrongly an obstacle. Was the rig held at hip height outdoors?
2. **Beam width of the arc truth.** Handrails 0.7 m to the side are inside ±60° and become the arc truth outdoors,
   while the radar's gain at ±60° is low. Should the arc be ±30° / ±45°, or should the truth be weighted by the
   antenna pattern?
3. **The indoor phantom band at 0.8–1.1 m:** is it the carrier's body or arm? A short recording pointing at open
   space would show it.
4. **static_pass2 S1 and pilot_chairs score near zero.** Chairs are seen 0.2–0.3 m off, and there are near
   returns at 0.5–1.5 m in the pilot. Watch them with `make play`.
5. **Elevation calibration** needs targets from −60° to 0° (floor tags at several distances).

## 11. Next runs (cheap)

- `make an-ablation WORKERS=8` (about 30–60 min, cached per recording), then `make plots report`.
- `make story` for more range-time maps (edit `CURATED` in `analysis/story.py`).
- After any truth change: `make analysis FROM=truth` (the optimisation studies resume, so they are not redone).
