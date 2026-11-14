"""Alignment of a phone recording (video + Sensor Logger IMU) with radar takes, and the pack, on synthetic data:
a small video with the recorder's beeps in its sound and a flash, an IMU export with a gyro spike at the flash, and a
radar take whose beeps are 1000 s later on the host clock."""

import io
import json
import zipfile
from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from ravidata import align as A  # noqa: E402

HOST = 1000.0        # t_host = t_video + HOST
LAG = 0.3            # t_host = t_phone + LAG


def make_video(path, dur=6.0, fps=30):
    fs = 48000
    t = np.arange(int(dur * fs)) / fs
    x = np.zeros_like(t)
    for t0 in (1.0, 4.0, 4.26):                          # start beep, end double beep
        m = (t >= t0) & (t < t0 + 0.18)
        x[m] = 0.5 * np.sin(2 * np.pi * 1200 * t[m])
    x += 0.01 * np.random.default_rng(0).normal(size=t.size)
    with av.open(str(path), "w") as c:
        vs = c.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        aus = c.add_stream("aac", rate=fs)
        aus.layout = "mono"
        for i in range(int(dur * fps)):
            v = 200 if 2.0 <= i / fps < 2.1 else 40       # a flash = a motion spike at 2.0 s
            img = np.full((48, 64, 3), v, np.uint8)
            img[:, (i * 2) % 64] = 255
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            for p in vs.encode(fr):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
        n = 1024
        for k in range(0, x.size - n, n):
            af = av.AudioFrame.from_ndarray(x[k:k + n].astype(np.float32)[None, :], format="flt", layout="mono")
            af.sample_rate = fs
            af.pts = k
            af.time_base = Fraction(1, fs)
            for p in aus.encode(af):
                c.mux(p)
        for p in aus.encode():
            c.mux(p)


def make_imu(path):
    tp = np.arange(990.0, 1010.0, 0.01)                   # phone clock
    w = np.full_like(tp, 0.05)
    w[np.abs(tp - (HOST + 2.05 - LAG)) < 0.06] = 3.0      # twist at the flash (host 1002.05)
    ns = (tp * 1e9).astype(np.int64)
    gyro = "time,seconds_elapsed,z,y,x\n" + "\n".join(f"{n},{t - tp[0]},0,0,{v}" for n, t, v in zip(ns, tp, w))
    grav = "time,seconds_elapsed,z,y,x\n" + "\n".join(f"{n},{t - tp[0]},0,9.8,0" for n, t in zip(ns, tp))
    ori = "time,seconds_elapsed,qz,qy,qx,qw,roll,pitch,yaw\n" + "\n".join(
        f"{n},{t - tp[0]},0,0,0.7071068,0.7071068,0,0,0" for n, t in zip(ns, tp))   # +90 deg about x
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Metadata.csv", "version,device name\n3,test\n")
        z.writestr("Gyroscope.csv", gyro)
        z.writestr("Gravity.csv", grav)
        z.writestr("Orientation.csv", ori)


def make_take(runs, name="20990101_000000_spi_tcc-1d-40mm"):
    import h5py

    from ravidata.config import presets
    from ravidata.cube import iq_to_words
    cfg = presets()["tcc-1d-40mm"]
    rd = runs / name
    rd.mkdir(parents=True)
    th = np.arange(HOST + 1.0, HOST + 4.0, 0.03)
    cube = np.zeros((1, 3, 128), np.complex64)
    cube[0, :, 45] = 0.1
    w = iq_to_words(cube, cfg)
    with h5py.File(rd / "cubes.h5", "w") as f:
        f.attrs["config_json"] = json.dumps(cfg.to_dict())
        f["words"] = np.tile(w, (th.size, 1))
        f["shape"] = np.tile([1, 3, 128], (th.size, 1))
        f["frame_index"] = np.arange(th.size)
        f["t_host"] = th
    (runs / "TCC_TAKES.csv").write_text(
        "time,session,env,step,config,t_beep_start,t_beep_end,run_dir\n"
        f"x,S1,closed,free,tcc-1d-40mm,{HOST + 1.0},{HOST + 4.0},C:\\\\somewhere\\\\runs\\\\{name}\n", encoding="utf-8")
    return name


def test_tones_and_beep_groups():
    fs = 16000
    t = np.arange(3 * fs) / fs
    x = 0.01 * np.random.default_rng(1).normal(size=t.size)
    for t0 in (0.5, 2.0, 2.26):
        m = (t >= t0) & (t < t0 + 0.18)
        x[m] += 0.5 * np.sin(2 * np.pi * 1200 * t[m])
    tones = A.detect_tones(x, fs)
    assert [round(tt[0], 1) for tt in tones] == [0.5, 2.0, 2.3]
    assert [k for _, k in A.group_beeps(tones)] == ["single", "double"]


def test_match_beeps_many_takes():
    takes = [A.Take(f"r{i}", None, "c", 100 + 10 * i, 105 + 10 * i) for i in range(4)]   # identical lengths
    beeps = [(t.t_beep_start - 50.0, "single") for t in takes] + [(t.t_beep_end - 50.0, "double") for t in takes]
    a, b, m = A.match_beeps(beeps, takes)
    assert abs(b - 50.0) < 1e-6 and len(m) == 8


def test_two_recordings_in_one_run(tmp_path):
    """The recorder writes the 2nd recording of a run to cubes_02.h5: each take must load its own file."""
    import h5py
    runs = tmp_path / "runs"
    runs.mkdir()
    name = make_take(runs)
    rd = runs / name
    with h5py.File(rd / "cubes.h5", "r") as f, h5py.File(rd / "cubes_02.h5", "w") as g:
        for k in ("words", "shape", "frame_index"):
            g[k] = f[k][:40]
        g["t_host"] = np.arange(40) * 0.03 + HOST + 10.0
        g.attrs["config_json"] = f.attrs["config_json"]
    with (runs / "TCC_TAKES.csv").open("a", encoding="utf-8") as f:
        f.write(f"y,S1,closed,free,tcc-1d-40mm,{HOST + 10.0},{HOST + 11.2},C:\\\\somewhere\\\\runs\\\\{name}\n")
    takes = A.find_takes(runs)
    assert [t.run_id for t in takes] == [name, name + "_02"]
    assert [t.cubes_name for t in takes] == ["cubes.h5", "cubes_02.h5"]
    r = [A.load_take_radar(t) for t in takes]
    assert len(r[0]["t_host"]) == 100 and len(r[1]["t_host"]) == 40
    assert r[1]["t_host"][0] == pytest.approx(HOST + 10.0)


def test_align_and_pack(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    name = make_take(runs)
    rec = tmp_path / "phone"
    rec.mkdir()
    make_video(rec / "clip.mp4")
    make_imu(rec / "sensors.zip")
    v, z = A.find_phone_files(rec)
    scan = A.scan_video(v)
    assert scan.n_frames == 180 and len(A.group_beeps(scan.tones)) == 2
    out, al, takes = A.headless_pack(v, z, runs)
    assert [t.run_id for t in takes] == [name]
    assert al.video_ok and abs(al.video_b - HOST) < 0.05, al
    assert al.imu_ok and abs(al.imu_lag - LAG) < 0.06, al
    P = A.load_pack(out)
    tk = P["takes"][name]
    assert tk["cube"].shape == (100, 1, 3, 128) and P["video"]["mp4"][:12].find(b"ftyp") >= 0
    assert abs(P["video"]["t_host"][0] - HOST) < 0.1
    vf = tk["video_frame"]
    assert vf[0] == pytest.approx(30, abs=2)                   # radar frame at host+1.0 -> video frame ~30
    # q = +90 deg about x: the phone's y axis (gravity says up) -> world up, the camera (-z) -> world +y (north)
    att = tk["attitude"]
    assert np.nanmedian(att["pitch_deg"]) == pytest.approx(0, abs=1)
    assert np.nanmedian(att["roll_deg"]) == pytest.approx(0, abs=1)
    assert np.nanmedian(att["yaw_deg"]) == pytest.approx(0, abs=1)
    assert A.rig_axes(A.load_sensorlogger(z))["phone_orientation"] == "portrait"


def test_aligner_gui(tmp_path):
    import os
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as e:            # no libEGL in a headless Linux VM
        pytest.skip(f"Qt not available: {e}")
    from ravidata.gui.aligner import AlignWindow
    runs = tmp_path / "runs"
    runs.mkdir()
    make_take(runs)
    rec = tmp_path / "phone"
    rec.mkdir()
    make_video(rec / "clip.mp4")
    make_imu(rec / "sensors.zip")
    app = QApplication.instance() or QApplication([])
    w = AlignWindow(runs)
    w.timer.stop()
    try:
        w.load(rec / "clip.mp4", rec / "sensors.zip")
        t0 = time.monotonic()
        while w._job is not None and time.monotonic() - t0 < 60:
            app.processEvents()
            w._tick()
            time.sleep(0.02)
        assert w.scan is not None and w.takes and w.radar[w.takes[0].run_id] is not None
        w.set_time(1.5)
        assert w._last_frame > 0 and w.video.pixmap() is not None
        w.sp_video.setValue(100)                              # manual offset moves the video by 3 frames
        assert w.al.manual_video_ms == 100
        out = w.export_to(tmp_path / "x.tccpack.h5")
        assert A.load_pack(out)["alignment"]["manual_video_ms"] == 100
    finally:
        w.close()
