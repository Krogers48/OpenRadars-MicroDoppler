#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realTimeProc_d435i_view.py — Unified depth-camera skeleton viewer and gait-metric extractor

Adds WiGait-style walking/stable-phase gating and stride metrics; emits a 'gait' dict in PUB payload.
"""

import argparse
import time
import csv
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pyrealsense2 as rs
from datetime import datetime, timezone

# ---------------------- Optional dependencies ----------------------
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

ET = ZoneInfo("America/New_York") if ZoneInfo else None
def ts_to_et_iso(ts: float) -> str:
    """UTC epoch seconds -> America/New_York ISO‑8601 with ms and offset."""
    if ET:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(ET).isoformat(timespec="milliseconds")
    # Fallback: local time if zoneinfo unavailable
    return datetime.fromtimestamp(float(ts)).isoformat(timespec="milliseconds")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except Exception:
    ZMQ_AVAILABLE = False

try:
    import zmq
    ZMQ_AVAILABLE = True
except Exception:
    ZMQ_AVAILABLE = False

# ---------------------- Demo logging control ----------------------
# When True, disable all CSV writes so this script is purely a live demo.
DISABLE_CSV_LOGGING = True


class _NullCSVWriter:
    def writerow(self, *args, **kwargs):
        pass



# ---------------------- Parameters ----------------------
PARAMS = {
    "LPF_SKELETON_HZ": 10.0,
    "STEP_MIN_TIME_S": 0.30,
    "STEP_MAX_TIME_S": 1.20,
    "WINDOW_S": 10.0,
    "ZMQ_PORT": "tcp://127.0.0.1:5558",
    "FLUSH_EVERY": 60,
}

WIGAIT = {
    "WINDOW_S": 4.0,
    "DIAMETER_B_M": 1.6,
    "STABLE_DELTA": 0.001,
    "STABLE_DV": 0.45,
    "PEAK_MIN_DT_S": 0.10
}

# ---------------------- Data classes ----------------------
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass
class Joint3D:
    name: str
    u: float
    v: float
    x: float
    y: float
    z: float
    visibility: float

# ---------------------- RealSense wrapper ----------------------
class DepthCamera:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        verbose: bool = True,
        use_wait: bool = False,
        enable_spatial: bool = True,
        enable_temporal: bool = True,
        enable_hole_filling: bool = True,
    ):
        self.verbose = verbose
        self.use_wait = use_wait

        # New flags
        self.enable_spatial = bool(enable_spatial)
        self.enable_temporal = bool(enable_temporal)
        self.enable_hole_filling = bool(enable_hole_filling)

        ctx = rs.context()
        if ctx.query_devices().size() == 0:
            raise RuntimeError("No RealSense device found.")

        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipe.start(self.cfg)

        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        # Filters are still constructed, but may be skipped in get_frame()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.thr = rs.threshold_filter(0.1, 6.0)

        color_intr = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intr = CameraIntrinsics(
            color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy
        )
        if self.verbose:
            print(
                f"[INFO] RealSense {width}x{height}@{fps} started. "
                f"Depth scale {self.depth_scale} m/unit"
            )

    def get_frame(self):
        frames = self.pipe.wait_for_frames() if self.use_wait else self.pipe.poll_for_frames()
        if not frames:
            return False, None, None, None
        if frames.get_color_frame():
            frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return False, None, None, None

        # Conditionally apply filters
        if self.enable_spatial:
            depth = self.spatial.process(depth)
        if self.enable_temporal:
            depth = self.temporal.process(depth)
        if self.enable_hole_filling:
            depth = self.hole_filling.process(depth)

        depth = self.thr.process(depth)

        depth_viz = np.asanyarray(self.colorizer.colorize(depth).get_data())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        color_bgr = np.asanyarray(color.get_data())
        return True, depth_viz, color_bgr, depth_m

    def release(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

# ---------------------- Pose estimator ----------------------
class PoseEstimator3D:
    MP_NAMES = [
        "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye",
        "right_eye_outer","left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
        "left_heel","right_heel","left_foot_index","right_foot_index"
    ]

    SIMPLE_EDGES = [
        (11,13),(13,15),(15,17),(12,14),(14,16),(16,18),
        (11,12),(23,24),(11,23),(12,24),
        (23,25),(25,27),(27,29),(29,31),
        (24,26),(26,28),(28,30),(30,32)
    ]

    def __init__(self, intr: CameraIntrinsics, enabled=True):
        self.intr = intr
        self.enabled = enabled and MP_AVAILABLE
        self._pose = None
        if self.enabled:
            self._pose = mp.solutions.pose.Pose(model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def _pix_to_m(self, u, v, z):
        X = (u - self.intr.cx) / self.intr.fx * z
        Y = (v - self.intr.cy) / self.intr.fy * z
        return X, Y, z

    def _valid_depth(self, depth_m, u, v, win=2):
        h, w = depth_m.shape
        uu, vv = int(round(u)), int(round(v))
        if 0 <= vv < h and 0 <= uu < w:
            z = float(depth_m[vv, uu])
            if z > 0: return z
        for r in range(1, win+1):
            patch = depth_m[max(0,vv-r):min(h,vv+r+1), max(0,uu-r):min(w,uu+r+1)]
            nz = patch[patch>0]
            if nz.size: return float(np.min(nz))
        return 0.0

    def estimate(self, color_bgr, depth_m):
        if not self.enabled or self._pose is None:
            return []
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return []
        h,w = color_bgr.shape[:2]
        out = []
        for idx,lm in enumerate(res.pose_landmarks.landmark):
            u,v = float(lm.x)*w, float(lm.y)*h
            z = self._valid_depth(depth_m,u,v)
            if z>0:
                x,y,z = self._pix_to_m(u,v,z)
                name = self.MP_NAMES[idx] if idx < len(self.MP_NAMES) else f"id{idx}"
                vis = float(getattr(lm,"visibility",1.0))
                out.append(Joint3D(name,u,v,x,y,z,vis))
        return out

    def draw(self, img, joints):
        out = img.copy()
        idx = {n:i for i,n in enumerate(self.MP_NAMES)}
        uv = [None]*len(self.MP_NAMES)
        for j in joints:
            if j.name in idx:
                uv[idx[j.name]] = (int(round(j.u)), int(round(j.v)))
            cv2.circle(out,(int(round(j.u)),int(round(j.v))),3,(0,255,0),-1)
        for a,b in self.SIMPLE_EDGES:
            if uv[a] and uv[b]:
                cv2.line(out,uv[a],uv[b],(0,255,0),2)
        return out

# ---------------------- Math helpers ----------------------
def _ma_zero_phase(x, fc, fs):
    N=max(3,int(round(0.443*fs/max(fc,1e-3))))
    if N%2==0: N+=1
    k=np.ones(N)/N; pad=N//2
    xp=np.pad(x,(pad,pad),mode="edge")
    y=np.convolve(xp,k,mode="same")[pad:-pad]
    return y

def _estimate_ap_sign(z): return +1 if np.median(np.diff(z))>=0 else -1

def _detect_peaks_1d(x,t,min_dt):
    if len(x)<3: return np.array([],int)
    pk=[]; last=-1e9
    for i in range(1,len(x)-1):
        if x[i]>=x[i-1] and x[i]>=x[i+1] and (t[i]-last)>=min_dt:
            pk.append(i); last=t[i]
    return np.array(pk,int)

def _skeleton_steps(ts,Lz,Rz,Hz,fs):
    t=ts.astype(float); dL=Lz-Hz; dR=Rz-Hz
    pkL=_detect_peaks_1d(dL,t,PARAMS["STEP_MIN_TIME_S"])
    pkR=_detect_peaks_1d(dR,t,PARAMS["STEP_MIN_TIME_S"])
    ev=np.concatenate([
        np.stack([pkL,np.zeros_like(pkL)],1),
        np.stack([pkR,np.ones_like(pkR)],1)
    ]) if (pkL.size or pkR.size) else np.zeros((0,2),int)
    if not ev.size: return {},None
    ev=ev[np.argsort(ev[:,0])]
    times=t[ev[:,0]]; feet=ev[:,1]
    sel=[]; last_t=-1e9; last_f=-1
    for ti,fi in zip(times,feet):
        if (ti-last_t)>=PARAMS["STEP_MIN_TIME_S"] and (ti-last_t)<=PARAMS["STEP_MAX_TIME_S"] and fi!=last_f:
            sel.append(ti); last_t=ti; last_f=fi
    if len(sel)<2: return {},None
    ST=np.diff(sel); SL=[]
    for ti in sel:
        i=(np.abs(t-ti)).argmin()
        SL.append(abs(Lz[i]-Rz[i]))
    SL=np.array(SL)
    cad=60/np.maximum(np.mean(ST),1e-9)
    return {"step_times":np.array(sel),"ST":ST,"SL":SL,"cad":cad},sel[-1]

# ---------------------- WiGait helpers (skeleton) ----------------------
def _rolling_diameter(t, x, win_s):
    t = np.asarray(t, float); x = np.asarray(x, float)
    if t.size == 0: return 0.0, (0.0, 0.0), np.zeros(0,bool)
    lo = t[-1] - float(win_s)
    mask = t >= lo
    if not np.any(mask): return 0.0, (0.0, 0.0), mask
    xx = x[mask]
    diam = float(xx.max() - xx.min())
    tm = t[mask]
    return diam, (float(tm[0]), float(tm[-1])), mask

def _stable_phase(t, v, dv=0.45, delta=0.001):
    t = np.asarray(t, float); v = np.asarray(v, float)
    if v.size < 8: return None, None, np.zeros_like(v,bool)
    vmed = float(np.median(v))
    keep = (v >= (vmed - float(dv)))
    best = (0, -1); cur = None
    for i, ok in enumerate(keep):
        if ok and cur is None: cur = [i,i]
        elif ok: cur[1] = i
        elif cur is not None:
            if (cur[1]-cur[0]) > (best[1]-best[0]): best = (cur[0],cur[1])
            cur = None
    if cur is not None and (cur[1]-cur[0]) > (best[1]-best[0]): best = (cur[0],cur[1])
    if best[1] <= best[0]: return None, None, np.zeros_like(v,bool)
    st,en = best
    mask = np.zeros_like(v,bool); mask[st:en+1]=True
    return float(t[st]), float(t[en]), mask

def _fft_stride_freq(t, v):
    t = np.asarray(t, float); v = np.asarray(v, float)
    if t.size < 16: return None
    dt = float(np.median(np.diff(t))); dt = max(dt, 1e-3)
    ts = np.arange(t[0], t[-1], dt)
    if ts.size < 32: return None
    vs = np.interp(ts, t, v)
    win = np.hanning(ts.size)
    X = np.fft.rfft((vs - vs.mean()) * win)
    f = np.fft.rfftfreq(ts.size, d=dt)
    band = (f >= 0.5) & (f <= 3.5)
    if not np.any(band): return None
    i = int(np.argmax(np.abs(X[band])**2))
    return float(f[band][i])

# ---------------------- Main ----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--width",type=int,default=640)
    ap.add_argument("--height",type=int,default=480)
    ap.add_argument("--fps",type=int,default=20)
    ap.add_argument("--no-mp",action="store_true")
    ap.add_argument("--wait",action="store_true")
    ap.add_argument("--verbose",action="store_true", default=False)  # fixed
    ap.add_argument("--session-dir", default=None,
                    help="Base session dir (else sessions/<timestamp>/depth/)")
    ap.add_argument("--log-wide", action="store_true",
                    help="Also log wide per-frame CSV")
    ap.add_argument("--pub", action="store_true",
                    help="Publish frames via ZMQ (ts + overlay image)")
    ap.add_argument("--headless", action="store_true",
                    help="Disable local OpenCV windows; publish frames only")

    # New flags
    ap.add_argument("--no-spatial", action="store_true",
                    help="Disable RealSense spatial filter")
    ap.add_argument("--no-temporal", action="store_true",
                    help="Disable RealSense temporal filter")
    ap.add_argument("--no-hole", action="store_true",
                    help="Disable RealSense hole-filling filter")

    args = ap.parse_args()

    if args.session_dir:
        depth_dir=Path(args.session_dir)/"depth"
    else:
        ts=time.strftime("%Y-%m-%d_%H%M%S")
        depth_dir= Path("sessions") / ts / "depth"
    depth_dir.mkdir(parents=True,exist_ok=True)
    if args.verbose: print(f"[INFO] Writing to {depth_dir}")

    meta = {
        "fps": args.fps,
        "lpf_skeleton_hz": PARAMS["LPF_SKELETON_HZ"],
        "step_min_time_s": PARAMS["STEP_MIN_TIME_S"],
        "step_max_time_s": PARAMS["STEP_MAX_TIME_S"],
        "ap_axis": "camera_Z(+away)",
        "notes": "Ref gait + WiGait"
    }
    with open(depth_dir / "skeleton_run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if DISABLE_CSV_LOGGING:
        long_fp = None
        metrics_fp = None
        wigait_fp = None
        wide_fp = None

        long_csv = _NullCSVWriter()
        metrics_csv = _NullCSVWriter()
        wigait_csv = _NullCSVWriter()
        wide_csv = _NullCSVWriter() if args.log_wide else None
    else:
        long_fp = open(depth_dir / "skeleton_joints.csv", "w", newline="")
        long_csv = csv.writer(long_fp)
        long_csv.writerow(["time_s", "name", "x", "y", "z", "visibility"])

        metrics_fp = open(depth_dir / "skeleton_gait_metrics.csv", "w", newline="")
        metrics_csv = csv.writer(metrics_fp)
        metrics_csv.writerow(["time_s", "cadence_spm", "last_step_time_s", "last_step_length_m"])

        wigait_fp = open(depth_dir / "wigait_skeleton_metrics.csv", "w", newline="")
        wigait_csv = csv.writer(wigait_fp)
        wigait_csv.writerow([
            "time_s", "mean_speed_mps", "stride_freq_hz", "mean_stride_len_m",
            "cadence_spm", "step_count",
            "walk_start", "walk_stop", "stable_start", "stable_stop"
        ])

        wide_fp = None
        wide_csv = None
        if args.log_wide:
            wide_fp = open(depth_dir / "skeleton_joints_wide.csv", "w", newline="")
            wide_csv = csv.writer(wide_fp)
            header = ["time_s"] + [
                f"{n}_{c}"
                for n in PoseEstimator3D.MP_NAMES
                for c in ["x", "y", "z", "vis", "u", "v"]
            ]
            wide_csv.writerow(header)


    if args.pub and not ZMQ_AVAILABLE:
        print("[WARN] pyzmq not installed; disabling publisher")
        args.pub=False
    pub_socket=None
    if args.pub:
        ctx=zmq.Context(); pub_socket=ctx.socket(zmq.PUB)
        pub_socket.bind(PARAMS["ZMQ_PORT"])
        if args.verbose: print(f"[INFO] ZMQ PUB bound {PARAMS['ZMQ_PORT']}")

    dc = DepthCamera(
        args.width,
        args.height,
        args.fps,
        verbose=args.verbose,
        use_wait=args.wait,
        enable_spatial=not args.no_spatial,
        enable_temporal=not args.no_temporal,
        enable_hole_filling=not args.no_hole,
    )
    pe=PoseEstimator3D(dc.intr,enabled=(not args.no_mp))

    if not args.headless:
        cv2.namedWindow("Color + 3D joints",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth",cv2.WINDOW_NORMAL)

    buf_t,buf_L,buf_R,buf_H=[],[],[],[]
    frame_count=0; t0=time.time(); fps=0
    last_gait_payload = None

    while True:
        ok,depth_viz,color_bgr,depth_m=dc.get_frame()
        if not ok:
            if not args.headless: cv2.waitKey(1)
            else: time.sleep(0.001)
            continue
        frame_count+=1
        if frame_count%10==0:
            t1=time.time(); fps=10/max(t1-t0,1e-6); t0=t1

        joints=pe.estimate(color_bgr,depth_m)
        overlay=pe.draw(color_bgr,joints)
        cv2.putText(overlay,f"FPS:{fps:.1f} joints:{len(joints)}",(10,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        tsec=time.time()

        if joints:
            for j in joints:
                long_csv.writerow([f"{tsec:.6f}", j.name, j.x, j.y, j.z, j.visibility])

            if args.log_wide and wide_csv:
                row = [f"{tsec:.6f}"]
                d = {j.name: j for j in joints}
                for n in PoseEstimator3D.MP_NAMES:
                    j = d.get(n, None)
                    row.extend([
                        j.x if j else "NaN",
                        j.y if j else "NaN",
                        j.z if j else "NaN",
                        j.visibility if j else "NaN",
                        j.u if j else "NaN",
                        j.v if j else "NaN",
                    ])
                wide_csv.writerow(row)

            if (not DISABLE_CSV_LOGGING) and (frame_count % PARAMS["FLUSH_EVERY"] == 0):
                if long_fp:
                    long_fp.flush()
                if wide_fp:
                    wide_fp.flush()


        # Gait (reference + WiGait)
        names={j.name:j for j in joints}
        need=("left_ankle","right_ankle","left_hip","right_hip")
        if all(n in names for n in need):
            L,R = names["left_ankle"], names["right_ankle"]
            hipL, hipR = names["left_hip"], names["right_hip"]
            H = (np.array([hipL.x,hipL.y,hipL.z])+np.array([hipR.x,hipR.y,hipR.z]))/2
            buf_t.append(tsec)
            buf_L.append([L.x,L.y,L.z]); buf_R.append([R.x,R.y,R.z]); buf_H.append(H)
            while buf_t and (tsec-buf_t[0]>PARAMS["WINDOW_S"]):
                buf_t.pop(0); buf_L.pop(0); buf_R.pop(0); buf_H.pop(0)
            if len(buf_t)>=8:
                ts=np.array(buf_t); dt=np.diff(ts).mean() if len(ts)>1 else 1/args.fps
                fs=1/max(dt,1e-6)
                Lz=np.array(buf_L)[:,2]; Rz=np.array(buf_R)[:,2]; Hz=np.array(buf_H)[:,2]
                ap_sign=+1 if np.median(np.diff(Hz))>=0 else -1
                Lz=_ma_zero_phase(ap_sign*Lz,PARAMS["LPF_SKELETON_HZ"],fs)
                Rz=_ma_zero_phase(ap_sign*Rz,PARAMS["LPF_SKELETON_HZ"],fs)
                Hz=_ma_zero_phase(ap_sign*Hz,PARAMS["LPF_SKELETON_HZ"],fs)

                # Reference gait from ankle vs hip
                res,last=_skeleton_steps(ts,Lz,Rz,Hz,fs)
                if res:
                    cad=res["cad"]; ST=res["ST"][-1] if len(res["ST"]) else 0; SL=res["SL"][-1] if len(res["SL"]) else 0
                    metrics_csv.writerow([f"{tsec:.6f}",f"{cad:.3f}",f"{ST:.3f}",f"{SL:.3f}"])
                    cv2.putText(overlay,f"Ref: cadence={cad:5.1f} spm ST={ST:4.2f}s SL={SL:4.2f}m",
                                (10,50),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)

                # WiGait gating on hip Z
                pos = Hz
                vel = np.gradient(Hz, ts)
                # walking window
                lo = ts[-1] - WIGAIT["WINDOW_S"]
                maskW = ts >= lo
                diam = (pos[maskW].max()-pos[maskW].min()) if np.any(maskW) else 0.0
                is_walking = diam >= float(WIGAIT["DIAMETER_B_M"])
                wwin = (float(ts[maskW][0]), float(ts[maskW][-1])) if np.any(maskW) else (0.0,0.0)
                # stable phase
                if is_walking and np.any(maskW):
                    tW, vW, pW = ts[maskW], np.abs(vel[maskW]), pos[maskW]
                    vmed = float(np.median(vW))
                    keep = (vW >= (vmed - float(WIGAIT["STABLE_DV"])))
                    # longest contiguous run
                    best=(0,-1);cur=None
                    for i,ok in enumerate(keep):
                        if ok and cur is None: cur=[i,i]
                        elif ok: cur[1]=i
                        elif cur is not None:
                            if (cur[1]-cur[0])>(best[1]-best[0]): best=(cur[0],cur[1]); cur=None
                    if cur is not None and (cur[1]-cur[0])>(best[1]-best[0]): best=(cur[0],cur[1])
                    if best[1]>best[0]:
                        st_ts,en_ts = float(tW[best[0]]), float(tW[best[1]])
                        tS, vS, pS = tW[best[0]:best[1]+1], vW[best[0]:best[1]+1], pW[best[0]:best[1]+1]
                    else:
                        st_ts=en_ts=None; tS=vS=pS=np.array([])
                else:
                    st_ts=en_ts=None; tS=vS=pS=np.array([])

                mean_speed = float(np.mean(vS)) if np.size(vS) else 0.0
                f_stride = _fft_stride_freq(tS, vS) if np.size(vS) else None
                # steps: prefer reference; else detect from hip velocity
                if res and res.get("step_times") is not None:
                    step_times = list(map(float, res["step_times"]))
                else:
                    step_times = []
                    if np.size(vS) >= 8:
                        dtS = float(np.median(np.diff(tS))) if tS.size>1 else 0.05
                        min_gap = max(1, int(round(WIGAIT["PEAK_MIN_DT_S"]/max(dtS,1e-3))))
                        last = -10**9
                        for i in range(1, vS.size-1):
                            if vS[i] >= 0.2 and vS[i] >= vS[i-1] and vS[i] >= vS[i+1]:
                                if (i-last) >= min_gap:
                                    step_times.append(float(tS[i])); last=i
                step_points = [float(np.interp(ti, ts, pos)) for ti in step_times] if len(step_times)>=1 else []
                step_lengths = list(np.diff(step_points)) if len(step_points)>=2 else []
                cadence = 60.0/float(np.mean(np.diff(step_times))) if len(step_times)>=2 else 0.0
                mean_stride_len = float(np.mean(step_lengths)) if len(step_lengths) else 0.0

                cv2.putText(overlay,
                    f"Gait: v={mean_speed:4.2f} m/s ", #f={0.0 if f_stride is None else f_stride:4.2f} Hz "
                    #f"cad={cadence:5.1f} spm SL={mean_stride_len:4.2f} m",
                    (10,74), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,200,255),2)

                wigait_csv.writerow([
                    f"{tsec:.6f}", f"{mean_speed:.6f}", f"{0.0 if f_stride is None else f_stride:.6f}",
                    f"{mean_stride_len:.6f}", f"{cadence:.6f}", int(len(step_times)),
                    f"{wwin[0]:.6f}", f"{wwin[1]:.6f}", f"{0.0 if st_ts is None else st_ts:.6f}", f"{0.0 if en_ts is None else en_ts:.6f}"
                ])
                last_gait_payload = {
                    "ts": float(tsec),
                    "walk_window": [float(wwin[0]), float(wwin[1])] if is_walking else [0.0,0.0],
                    "stable_phase": [0.0 if st_ts is None else float(st_ts), 0.0 if en_ts is None else float(en_ts)],
                    "metrics": {
                        "mean_speed_mps": float(mean_speed),
                        "stride_freq_hz": float(0.0 if f_stride is None else f_stride),
                        "mean_stride_len_m": float(mean_stride_len),
                        "cadence_spm": float(cadence),
                        "step_count": int(len(step_times))
                    }
                }

        if args.pub and pub_socket:
            msg = {"ts": tsec, "image": overlay}
            if last_gait_payload is not None:
                msg["gait"] = last_gait_payload
            try:
                pub_socket.send_pyobj(msg)
            except Exception:
                pass

        if not args.headless:
            cv2.imshow("Color + 3D joints",overlay)
            cv2.imshow("Depth",depth_viz)
            k=cv2.waitKey(1)&0xFF
            if k in (27,ord("q")): break
        else:
            time.sleep(0.001)

    if not DISABLE_CSV_LOGGING:
        if long_fp:
            long_fp.close()
        if metrics_fp:
            metrics_fp.close()
        if wigait_fp:
            wigait_fp.close()
        if wide_fp:
            wide_fp.close()

    if args.pub and pub_socket:
        pub_socket.close()
    dc.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__=="__main__":
    main()
