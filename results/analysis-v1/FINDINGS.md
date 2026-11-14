# RAVIDATA radar analysis v1: findings

Status: second full run (2026-11-14, commit 1a35898): new truth (obstacle = anything more than 0.12 m above the
floor), eval, eval-tuned, ablation and the `floor` step. The optimisation trials are from the first run (commit
3a2478d, older truth); their best settings were re-evaluated on the new truth.

- **Numbers:** `REPORT.md` (generated from the tables).
- **Figures:** `figures/` (`paper_*` = paper candidates, `story/` = explanatory).
- **Run log:** `runs.jsonl`.

## 0. In one paragraph

The radar measures the distance to the nearest obstacle well where nothing nearer competes. Between 0.75 and
1.5 m, 0.69 of frames are found. Beyond about 1.3 m it reports something nearer in almost every frame. That
nearer return has two different causes, one per environment:

- **Outdoors (radar 0.8 m above the walkway):** it is the floor entering the bottom of the beam. The alarm lies
  where the floor enters the ±30° beam in 82 % of frames.
- **Indoors (radar 1.3–1.4 m above the floor):** it is something at 0.7–1.15 m that the camera does not see, and
  that a still radar does not see either. The leading hypothesis is the carrier's own body (legs).

Neither cause is a detection problem. One change at a time, pre- and post-processing moves the curve by at most
±0.02 AUC, and the multi-objective optimisation mostly slides along the same trade-off. The fixes are in the
mounting: tilt the radar level or up, independently of the camera, and keep the body out of the beam.

Still radars are a different story. With an empty-scene reference, placed chairs and brooms are found in
0.94–0.99 of frames.

## 1. What is being measured

**Truth.** The obstacle geometry from the camera, in the radar frame, at every radar frame
(`figures/story/truth_*.png`):

- **Carried takes:**
  - Pi3X keyframe depth, made metric by the AprilTags (tag ÷ map depth 1.04 indoors, 0.93 outdoors).
  - "Up" comes from the phone's gravity sensor, and the floor is a RANSAC plane.
  - An obstacle is anything more than 0.12 m above the floor. Nothing is removed overhead; ceilings and roofs count.
  - The arc truth uses the keyframes of the same map window within ±4 s, placed with that window's own poses. The
    fused map is not used: its joins leave ghost surfaces.
- **Static takes:** Depth Pro depth, scaled on the tags. The placed objects (still chairs and brooms) are scored at
  object level, table `truth_objects`.

**Three views of "detected", from strict to lenient:**

1. **Per direction** (`f1_50`): one obstacle per 4° sector at 0.5 m. A stress test only (section 8).
2. **Nearest in the camera view** (`near_*`).
3. **Arc view** (`arc_*`, carried takes): the distance to the nearest obstacle anywhere in the beam (±60° azimuth,
   ±30° elevation), direction ignored. This is the handheld use case and the main metric.
   - Found: within 0.25 m.
   - Early alarm: something reported more than 0.25 m nearer than the truth, or something reported when there is
     nothing there.
4. **Static takes** (`obj_*`): a placed object is found when any detection, at any angle, lies within 0.25 m of
   its range. Every other detection counts as "other".

## 2. The sensor

- **Timing.**
  - 1D config: 33.6 fps, 0.13 % frames lost, 3 ms jitter.
  - 2D configs: 9–10 % frames lost, 20–27 ms jitter. The link or host is the weak point.
- **Noise floor:** −55 dB of full scale (1D), −60 / −67 dB (2D). Static returns sit at −42 to −45 dB.
- **Range bias:** +0.117 m over 39 still targets, with a median residual of 4 cm.
- **Angles:**
  - Azimuth (RX2–RX1): R 0.80, median residual 8°.
  - Elevation (RX1–RX0): R 0.91, 5°, but only over −16° to −3°.
  - Two receivers give one angle per range bin, a power-weighted mix of everything at that distance.

## 3. Main result: the arc view

`figures/paper_operating_curves.png`; `REPORT.md`, "equal early-alarm budgets" (opt trials, first truth):

| early-alarm budget | best "found" over all families | spread between families |
|---|---|---|
| 5 % | 0.13 | 0.06–0.13 |
| 10 % | 0.17 | 0.13–0.17 |
| 20 % | 0.29 | 0.16–0.29 |
| 30 % | 0.36 | 0.21–0.36 |

- **At an equal budget the families are within a few points.** CA-CFAR and CFAR + DBSCAN are on top; the legacy
  energy grid is lowest.
- **On the new truth, the CA-CFAR base pipeline** finds 0.10 at a 10 % early budget and 0.20 at 20 % (ablation,
  section 5).
- **By distance** (`figures/paper_outcome_vs_distance_tuned.png`, tuned CA-CFAR):

| true nearest obstacle | found | early |
|---|---|---|
| 0.3–0.5 m | 0.12 | 0.00 |
| 0.5–0.75 m | 0.43 | 0.00 |
| 0.75–1.0 m | 0.69 | 0.03 |
| 1.0–1.5 m | 0.69 | 0.25 |
| 1.5–2.0 m | 0.12 | 0.86 |
| > 2 m | 0.00 | 1.00 |

The near band is lost to leakage and the near cut (r_min 0.6–0.8 m when tuned). Beyond 1.5 m, a nearer return
wins. Section 4 is about that return.

## 4. Why it fails beyond ~1.3 m: two causes (step `floor`)

### 4a. Outdoors: the floor entering the beam

`figures/paper_floor_margin.png`. The floor margin is the range at which the floor enters the ±30° beam minus the
true obstacle range.

| floor margin, outdoor (tuned CA-CFAR) | frames | found | early |
|---|---|---|---|
| < −0.5 m (floor well before the obstacle) | 1644 | 0.00 | 1.00 |
| −0.5 to −0.25 m | 246 | 0.29 | 0.71 |
| ±0.25 m | 1300 | 0.75 | 0.17 |
| > +0.25 m (floor behind the obstacle) | 1025 | 0.41–0.77 | 0.04–0.41 |

- **Outdoors the floor comes first** (by more than 0.25 m) **in 42 % of frames**, and in those frames the result is
  an early alarm.
- **With nothing within 2 m**, the alarm sits at 1.32 m (IQR 1.25–1.39). The map puts the floor's entry into the
  beam at 1.31 m, and the alarm lies within 0.25 m of it in 82 % of frames (table `phantom`).
- **This also supports the outdoor rig height.** The truth puts the radar 0.81 m above the walkway, which earlier
  looked suspicious. The radar's own floor return lands exactly where that height predicts.
- **Caveat:** the "found" bar at ±0.25 m may include the floor return coinciding with the obstacle.

### 4b. Indoors: something the camera does not see

| nothing within 2 m (tuned CA-CFAR, 200 sampled frames) | indoor | outdoor |
|---|---|---|
| alarm range, median (IQR) | 0.91 m (0.73–1.15) | 1.32 m (1.25–1.39) |
| radar height above the floor | 1.41 m | 0.81 m |
| floor enters the ±30° beam at | 1.91 m | 1.31 m |
| alarm within 0.25 m of that | 9 % | 82 % |
| nearest floor point at any angle | 1.30 m | 0.87 m |
| nearest obstacle in ±60° azimuth at any elevation | 2.12 m | 3.35 m |

- **Indoors the alarm is nearer than any floor point.** The radar is 1.4 m up, so the floor even directly below is
  further away than 0.9 m. The camera map has nothing at that range, at any elevation.
- **A still radar does not show it.** In the empty scene of static_pass1 on a tripod, the nearest detection is at
  3.55 m.
- **So it comes with carrying.** The leading hypothesis is the carrier's legs or body, seen at a steep angle below
  the radar, which was held at chest height and pitched down about 20°.
  - Outdoors the radar was at 0.8 m, which puts the legs at about 0.4–0.5 m, inside the near cut. That would
    explain why outdoors only the floor remains.
  - Other candidates are the arm or a rig reflection. Section 10 lists a 2-minute test to settle this.
- **Consequence:** indoors, obstacles at 1.25–2 m are found in only 0.10–0.18 of frames at every pitch.

### 4c. Pitch: weaker than the pooled numbers suggested

Pooled over both environments, level pointing seemed to double "found" at 1.25–2 m (0.31–0.38 → 0.72). That was
mostly a mix of environments: 166 of the 197 level frames in that band are outdoors. Split by environment
(`figures/eval_pitch.png`), at 1.25–2 m:

| found (1.25–2 m) | < −25° | −25..−15° | −15..−5° | level |
|---|---|---|---|---|
| outdoor | 0.73 | 0.67 | 0.44 | 0.83 |
| indoor | 0.16 | 0.15 | 0.18 | 0.10 (31 frames) |

- **Indoors pitch makes no difference**, because the indoor return is not the floor (4b).
- **Outdoors the effect is not monotonic.** The pitch the walker happened to hold is confounded with the scene.
- **The variable that decides is the floor margin (4a), not pitch.** Pitch and height set the floor margin through
  the floor-entry range, H / sin(30° + downward pitch). With the ±30° beam, which matches the outdoor data:

| radar height | pitched down 20° | down 10° | level | up 10° |
|---|---|---|---|---|
| 0.8 m | 1.04 m | 1.24 m | 1.60 m | 2.34 m |
| 1.0 m | 1.31 m | 1.56 m | 2.00 m | 2.92 m |
| 1.3 m | 1.70 m | 2.02 m | 2.60 m | 3.80 m |

The recorded takes sit in the left half of this table, and they match it: outdoors 0.83 m and −10° predict
1.28 m, and the alarm was at 1.32 m. A level or slightly up-tilted radar at chest height moves the floor beyond
2.5 m. It does not remove the indoor return (4b).

## 5. Ablation: does processing matter? (`figures/paper_ablation.png`)

The base pipeline is CA-CFAR with default settings. Each row changes one stage and is scored at the best point of
its own threshold sweep.

| change | carried: found @10 % / @20 % early | carried AUC (0–30 %) | static: objects @≤2 other/frame / max |
|---|---|---|---|
| **base** | 0.10 / 0.20 | 0.146 | 0.35 / 0.83 |
| near cut r_min 0.7 m | 0.12 / 0.18 | **0.166** | 0.35 / 0.83 |
| integration a=0.8 / M-of-N 3 of 5 | 0.11 / 0.23, 0.11 / 0.21 | 0.160 | 0.33 / 0.80 |
| RX combining (coh / max / rx1) | 0.09–0.11 / 0.16–0.21 | 0.137–0.149 | 0.21–0.33 / 0.89–0.95 |
| Doppler zero-bin only (2D) | 0.08 / 0.17 | 0.110 | **0.43** / 0.82 (0.40 at ≤1) |
| EMA background a=0.9–0.98 | 0.06 / 0.14 | 0.105–0.109 | 0.12–0.13 / 0.50–0.53 |
| empty-scene reference | (none for carried) | 0.146 | **0.46 / 0.94** |
| SO-CFAR | 0.00 / 0.18 | 0.103 | 0.32 / **0.99** |
| tracker (confirm/hold) | 0.08 / 0.18–0.19 | 0.133–0.141 | 0.14 (nearest only) |
| log-odds grid / + IMU / + map poses | 0.11–0.13 / 0.11–0.16 | 0.088–0.124 | 0.29 / 0.42 |
| legacy energy grid (± IMU) | 0.00–0.08 / 0.11–0.13 | 0.077–0.079 | 0.04 / 0.96 |
| IMU height gate (5 settings) | 0.08–0.10 / 0.17–0.20 | 0.128–0.145 | 0.24–0.33 / 0.60–0.62 |

- **Carried takes: processing moves the curve by at most ±0.02 AUC.**
  - Small gains: a near cut of 0.5–0.7 m, power integration over a few frames, and M-of-N.
  - Losses: EMA background removal (−0.04), the grids (−0.02 to −0.06; −0.09 with a fast decay) and the legacy energy grid (−0.07).
  - The losses come from latency and smearing. The same early return is there whatever the processing does.
- **The IMU height gate does not help.** The indoor return is not floor, and the outdoor floor enters at the beam
  edge, where the RX1–RX0 elevation is least reliable (calibrated only over −16° to −3°).
- **Static takes: the background model is the main factor.**
  - An empty-scene reference reaches 0.94 at its loosest threshold and 0.46 at ≤2 other/frame. SO-CFAR reaches
    0.99.
  - Zero-Doppler (2D) is the best low-clutter setting: 0.40 at ≤1 other/frame.
  - EMA background removal, which the carried tuning picks, halves static detection (max 0.50): it treats anything
    still as background.
  - **A moving radar and a still radar need opposite background models.**
- **Cost:** 11 min for the whole ablation (cached per recording), against 3.8 h for the optimisation.

## 6. Temporal filters and IMU motion compensation

- **Our log-odds grid, temporal DBSCAN, M-of-N, integration and the tracker** are all temporal filters. None of
  them lifts the operating curve: they trade latency for persistence along the same curve.
- **Motion compensation does not help, even when perfect.**
  - IMU yaw plus an assumed walking speed (`grid_imu`): no gain.
  - The map trajectory (`grid_oracle`): no better.
  - Both early-return causes (4a, 4b) move with the carrier, so compensating the carrier's motion cannot remove
    them.
- **Legacy energy grid** (the earlier C firmware, re-implemented): the most sensitive at default settings (arc
  found 0.18) but the worst curve. It has no detector in front, so low-energy cells add early alarms.

## 7. Was the optimisation worth it?

- **Cost:** 3.8 h (150 trials × 11 families × 2 objectives, pooled).
- **It generalises.** Held-out takes reach 0.29–0.39 found (legacy grids 0.19) against 0.08–0.31 at default settings. The per-scene
  optimum is only about 0.04 better.
- **But the gain is movement along the trade-off curve.** Early alarms rise from 0.06–0.12 to 0.27–0.37. The
  settings it converged to agree with the ablation, which got there at 1/20 of the cost:
  - a near cut of 0.6–0.8 m;
  - coherent RX combining;
  - low thresholds;
  - EMA background removal, the one setting the ablation shows hurting at equal budget.
- **Do not rerun it.** For the paper, the ablation is the interpretable result, and the optimisation is evidence
  that tuning cannot buy more.

## 8. Per-direction metrics

- **F1 at 0.5 m** is 0.02–0.17 at default settings and up to 0.31 when tuned.
- **Range-only F1** is 2–4× higher.
- **Two causes:** the azimuth is unreliable, and one angle per range bin cannot fill 10–14 sectors.
- **Use in the paper:** a stress test that justifies the arc view, not a headline.

## 9. Other facts worth a sentence in the paper

- **Cardboard is nearly transparent at 60 GHz.** The box face is at 1.6 m; the radar's return is from the metal
  inside, at about 2.0 m.
- **Pi3X keyframe depth is close to metric** (tag ÷ depth 1.04 indoors). Depth Pro on the static takes is
  near-metric but stretches with range (±12 %). Depth Anything V2 Small is 1.4–2.6× too far.
- **Unplaced AprilTags must be excluded:** one pulled a scale estimate to 5.4.
- **Sync:** no numeric peak, but radar ridges lie on the truth lines within a few frames in the range-time maps.
  A tap on the rig at the start of future takes would close this.

## 10. Next recordings (short, and they decide the paper's discussion)

1. **Body test (indoor return, 4b), about 2 min, same rig and settings, indoors, open space ahead:**
   - (a) Stand still, rig at chest, level.
   - (b) Walk on the spot.
   - (c) Hold the rig at arm's length.
   - (d) As (b) with a metal plate or absorber below or behind the radar.

   If the 0.7–1.15 m return appears in (b), weakens in (c) and disappears in (d), it is the body.
2. **Tilt test (floor, 4a):** the radar on its own bracket, level and +10° up, with the camera free to point down.
   Walk one indoor and one outdoor course. Prediction (table in 4c): at 1.3 m height the floor moves out to
   2.6–3.8 m.
3. **Elevation calibration:** floor tags at 1–4 m, so the elevation is calibrated over −60° to 0°. The height gate
   is untestable without it.
4. **Static takes:** always record an empty scene (S0). pilot_chairs has none, which is why it scores near zero.

## 11. Open questions

1. Is the indoor return at 0.9 m the body (test 1)?
2. **Arc truth beam width.** Should ±60° azimuth be narrowed or weighted by the antenna pattern?
   - Outdoors, the floor data fit the ±30° elevation convention well.
   - The azimuth is untested.
3. **static_pass2 S0:** the still radar reports 1.19 m with the floor at 2.2 m in the beam and the radar 1.1 m up.
   It could be the floor at a steep angle, or something by the tripod; to check with `make play`.

## 12. Cheap reruns

- `make an-floor report`: seconds, after any eval change.
- `make plots` (paper figures); `uv run --group analysis python -m analysis plots --all-figs` adds the diagnostics.
- `make an-ablation`: cached per recording; only changed variants rerun.
- After a truth change: `make analysis FROM=truth` (the optimisation studies resume, not redone).
