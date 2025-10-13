#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d435i_view.py — Live 3D human skeletons from an Intel RealSense D435i (non-blocking).

What it does
------------
- Opens a D435i and streams aligned color + depth (metric).
- By default uses *non-blocking* polling so the UI stays responsive even if frames stall.
- Optionally runs MediaPipe Pose on RGB and lifts 2D landmarks to metric 3D with the depth map.
- Draws a skeleton overlay; shows FPS and joint count.

Run examples
------------
# Camera-only, verbose logs (good sanity check)
python d435i_view.py --no-mp --verbose

# Enable pose once video looks good
python d435i_view.py --verbose

Flags
-----
--no-mp     Disable MediaPipe (camera-only)
--wait      Use blocking wait_for_frames (old behavior) instead of non-blocking poll
--width     Stream width (default 640)
--height    Stream height (default 480)
--fps       Stream FPS (default 30)
--verbose   Print detailed logs

Controls
--------
ESC or 'q' to quit.

Notes
-----
- Joint 3D coordinates are in the color camera frame: X right, Y down, Z forward (meters).
- If a joint lands on a depth hole, we search a small window for the nearest valid depth.
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import pyrealsense2 as rs

# Optional dependency: MediaPipe Pose
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Joint3D:
    name: str
    u: float  # pixel x
    v: float  # pixel y
    x: float  # meters
    y: float  # meters
    z: float  # meters
    visibility: float  # [0,1]


class DepthCamera:
    """
    RealSense depth+color with depth aligned to color.
    Non-blocking poll mode by default to keep UI responsive.
    """
    def __init__(self, width: int, height: int, fps: int, verbose: bool = True, use_wait: bool = False):
        self.verbose = verbose
        self.use_wait = use_wait

        # 1) Check device
        ctx = rs.context()
        if ctx.query_devices().size() == 0:
            raise RuntimeError("No Intel RealSense device found. Plug in the D435i via USB 3.0 and install the Intel RealSense SDK.")

        # 2) Start pipeline
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self.verbose:
            mode = "wait_for_frames (blocking)" if self.use_wait else "poll_for_frames (non-blocking)"
            print(f"[INFO] Starting RealSense pipeline {width}x{height}@{fps}fps using {mode}…")
        try:
            self.profile = self.pipe.start(self.cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense pipeline: {e}")

        # 3) Device info
        dev = self.profile.get_device()
        name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "Intel RealSense"
        serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else "unknown"
        if self.verbose:
            print(f"[INFO] Found RealSense: {name} (SN {serial})")

        # 4) Align depth->color, colorizer for preview
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        # ---------------------- ADDED: filters & range clamp (setup once) ----------------------
        self.spatial = rs.spatial_filter()           # edge-preserving smoothing + hole filling
        self.spatial.set_option(rs.option.holes_fill, 3)  # 0..5 (3 is a good start)
        self.temporal = rs.temporal_filter()         # stabilize depth over time
        self.hole_filling = rs.hole_filling_filter(1)  # fill from neighboring pixels
        self.thr = rs.threshold_filter()             # clamp range so outliers don't dominate
        self.thr.set_option(rs.option.min_distance, 0.30)  # meters
        self.thr.set_option(rs.option.max_distance, 6.00)  # meters
        # --------------------------------------------------------------------------------------

        # 5) Depth sensor & scale
        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        if self.verbose:
            print(f"[INFO] Depth scale: {self.depth_scale:.6f} m/LSB")
        if self.depth_sensor.supports(rs.option.frames_queue_size):
            try:
                self.depth_sensor.set_option(rs.option.frames_queue_size, 1)
            except Exception:
                pass
        if self.depth_sensor.supports(rs.option.emitter_enabled):
            try:
                self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
            except Exception:
                pass
        # ---------------------- ADDED: preset to fill more pixels ----------------------
        if self.depth_sensor.supports(rs.option.visual_preset):
            try:
                self.depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_density)
            except Exception:
                pass
        # --------------------------------------------------------------------------------

        # 6) Intrinsics for 3D lifting
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.intrinsics = CameraIntrinsics(float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy))
        if self.verbose:
            print(f"[INFO] Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}, "
                  f"cx={self.intrinsics.cx:.1f}, cy={self.intrinsics.cy:.1f}")

        # 7) Warm-up (non-blocking)
        if self.verbose:
            print("[INFO] Warming up stream…")
        warmup_deadline = time.time() + 2.0  # 2 seconds
        while time.time() < warmup_deadline:
            frames = self.pipe.poll_for_frames()
            if frames:
                break
            time.sleep(0.01)

    def get_frame(self, timeout_ms: int = 1000) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (ok, depth_viz_bgr, color_bgr, depth_meters_float32).
        In non-blocking mode, returns quickly with ok=False if no frames yet.
        """
        if self.use_wait:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms)
            except RuntimeError:
                return False, None, None, None
        else:
            frames = self.pipe.poll_for_frames()
            if not frames:
                return False, None, None, None

        # Align only if a color frame exists (avoids internal exceptions)
        if frames.get_color_frame():
            frames = self.align.process(frames)

        # ---------------------- MODIFIED: apply filters on aligned depth each frame ----------------------
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            return False, None, None, None

        # spatial -> temporal -> hole_filling, then clamp range
        depth = self.spatial.process(depth)
        depth = self.temporal.process(depth)
        depth = self.hole_filling.process(depth)
        depth = self.thr.process(depth)
        # -----------------------------------------------------------------------------------------------

        depth_viz = np.asanyarray(self.colorizer.colorize(depth).get_data())
        depth_m   = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
        color_bgr = np.asanyarray(color.get_data())
        return True, depth_viz, color_bgr, depth_m

    def release(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


class PoseEstimator3D:
    """MediaPipe Pose -> 3D joints via depth."""
    MP_NAMES = [
        "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
        "left_ear","right_ear","mouth_left","mouth_right",
        "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
        "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
        "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
        "left_heel","right_heel","left_foot_index","right_foot_index"
    ]
    SIMPLE_EDGES = [
        (11,13),(13,15),(15,17),(12,14),(14,16),(16,18),
        (11,12),
        (23,24),(11,23),(12,24),
        (23,25),(25,27),(27,29),(29,31),
        (24,26),(26,28),(28,30),(30,32)
    ]

    def __init__(self, intrinsics: CameraIntrinsics, enabled: bool = True, depth_search_win: int = 2, verbose: bool = True):
        self.intr = intrinsics
        self.depth_search_win = depth_search_win
        self.verbose = verbose

        self.enabled = enabled and MP_AVAILABLE
        self._pose = None
        self._connections = self.SIMPLE_EDGES

        if self.enabled:
            if self.verbose:
                print("[INFO] MediaPipe Pose: enabled")
            self._pose = mp.solutions.pose.Pose(
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            try:
                self._connections = list(mp.solutions.pose.POSE_CONNECTIONS)
            except Exception:
                pass
        else:
            if enabled and not MP_AVAILABLE and self.verbose:
                print("[WARN] mediapipe not installed; run `pip install mediapipe==0.10.14` to enable 3D skeletons.")
            elif self.verbose:
                print("[INFO] MediaPipe Pose: disabled")

    def _pixel_to_meters(self, u: float, v: float, z: float) -> Tuple[float,float,float]:
        X = (u - self.intr.cx) / self.intr.fx * z
        Y = (v - self.intr.cy) / self.intr.fy * z
        return X, Y, z

    @staticmethod
    def _find_valid_depth(depth_m: np.ndarray, u: float, v: float, max_win: int = 2) -> float:
        h, w = depth_m.shape
        uu = int(round(u)); vv = int(round(v))
        if 0 <= vv < h and 0 <= uu < w:
            z = float(depth_m[vv, uu])
            if z > 0:
                return z
        for r in range(1, max_win+1):
            v0, v1 = max(0, vv-r), min(h, vv+r+1)
            u0, u1 = max(0, uu-r), min(w, uu+r+1)
            patch = depth_m[v0:v1, u0:u1]
            nz = patch[patch > 0.0]
            if nz.size:
                return float(np.min(nz))
        return 0.0

    def estimate(self, color_bgr: np.ndarray, depth_m: np.ndarray) -> List['Joint3D']:
        if not self.enabled or self._pose is None:
            return []
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return []

        h, w = color_bgr.shape[:2]
        joints: List[Joint3D] = []
        for idx, lm in enumerate(res.pose_landmarks.landmark):
            u = float(lm.x) * w
            v = float(lm.y) * h
            z = self._find_valid_depth(depth_m, u, v, max_win=self.depth_search_win)
            if z > 0.0:
                x, y, z = self._pixel_to_meters(u, v, z)
                name = self.MP_NAMES[idx] if idx < len(self.MP_NAMES) else f"id{idx}"
                vis = float(getattr(lm, "visibility", 1.0))
                joints.append(Joint3D(name, u, v, x, y, z, vis))
        return joints

    def draw(self, frame_bgr: np.ndarray, joints: List['Joint3D']) -> np.ndarray:
        out = frame_bgr.copy()
        idx_by_name = {name: i for i, name in enumerate(self.MP_NAMES)}
        uv = [None] * len(self.MP_NAMES)
        for j in joints:
            if j.name in idx_by_name:
                uv[idx_by_name[j.name]] = (int(round(j.u)), int(round(j.v)))
            cv2.circle(out, (int(round(j.u)), int(round(j.v))), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for a, b in self.SIMPLE_EDGES:
            if 0 <= a < len(uv) and 0 <= b < len(uv) and uv[a] is not None and uv[b] is not None:
                cv2.line(out, uv[a], uv[b], (0, 255, 0), 2, lineType=cv2.LINE_AA)
        return out


def parse_args():
    ap = argparse.ArgumentParser(description="D435i live 3D skeletons (non-blocking)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-mp", action="store_true", help="Disable MediaPipe (camera-only)")
    ap.add_argument("--wait", action="store_true", help="Use blocking wait_for_frames instead of non-blocking poll")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    return ap.parse_args()


def main():
    args = parse_args()

    # Create windows upfront so they appear even before the first frames
    cv2.namedWindow("Depth (colorized)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Color + 3D joints", cv2.WINDOW_NORMAL)

    try:
        dc = DepthCamera(args.width, args.height, args.fps, verbose=args.verbose, use_wait=args.wait)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[HINT] Close the RealSense Viewer/other apps, use a USB 3.0 port, and verify the SDK is installed.")
        return

    pe = PoseEstimator3D(dc.intrinsics, enabled=(not args.no_mp), verbose=args.verbose)

    if args.verbose:
        print("[INFO] Press ESC or 'q' to quit.")

    t0 = time.time()
    frame_count = 0
    fps = 0.0
    last_log = time.time()
    last_ok  = time.time()

    try:
        while True:
            ok, depth_viz, color_bgr, depth_m = dc.get_frame()
            now = time.time()

            if ok:
                last_ok = now
                frame_count += 1
                if frame_count % 10 == 0:
                    t1 = time.time()
                    fps = 10.0 / max(t1 - t0, 1e-6)
                    t0 = t1

                joints = pe.estimate(color_bgr, depth_m) if pe.enabled else []
                overlay = pe.draw(color_bgr, joints) if joints else color_bgr.copy()
                cv2.putText(overlay, f"FPS: {fps:.1f}   3D joints: {len(joints)}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.imshow("Depth (colorized)", depth_viz)
                cv2.imshow("Color + 3D joints", overlay)

            else:
                # Keep UI responsive and report stalls periodically
                if now - last_log > 2.0:
                    print("[WARN] No frames yet… (is the camera in use, or on a USB 2.0 port/hub?)")
                    last_log = now
                cv2.waitKey(10)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):  # ESC or 'q'
                break

            # If frames have been missing for a long time, print a single strong hint
            if not ok and (now - last_ok) > 8.0 and args.verbose:
                print("[HINT] Try: unplug/replug the D435i, close 'RealSense Viewer', "
                      "use a direct USB 3.0 port/cable (avoid hubs), and rerun with `--wait` if needed.")
                last_ok = now  # avoid spamming

    finally:
        dc.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# import pyrealsense2 as rs
# import numpy as np
# import cv2
#
# class DepthCamera:
#     def __init__(self, width=640, height=480, fps=30, preset="high_accuracy"):
#         self.pipe = rs.pipeline()
#         self.cfg = rs.config()
#         self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
#         self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
#         self.profile = self.pipe.start(self.cfg)
#
#         # Align depth to color when color is present (don’t block on drops)
#         self.align = rs.align(rs.stream.color)
#         self.colorizer = rs.colorizer()
#
#         # ---- FIX: get a true depth_sensor, not a generic sensor
#         dev = self.profile.get_device()
#         self.depth_sensor = dev.first_depth_sensor()            # <—
#         # keep latency low
#         if self.depth_sensor.supports(rs.option.frames_queue_size):
#             self.depth_sensor.set_option(rs.option.frames_queue_size, 1)
#         # visual preset
#         if self.depth_sensor.supports(rs.option.visual_preset):
#             self.depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
#         # IR projector on, high laser power
#         if self.depth_sensor.supports(rs.option.emitter_enabled):
#             self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
#         if self.depth_sensor.supports(rs.option.laser_power):
#             rng = self.depth_sensor.get_option_range(rs.option.laser_power)
#             self.depth_sensor.set_option(rs.option.laser_power, rng.max)
#
#         # Depth scale (raw units → meters)
#         self.depth_scale = self.depth_sensor.get_depth_scale()
#
#     def get_frame(self, timeout_ms=1000):
#         """Returns (ok, depth_viz_bgr, color_bgr, depth_meters_float32)"""
#         try:
#             frames = self.pipe.wait_for_frames(timeout_ms)
#         except RuntimeError:
#             return False, None, None, None
#
#         if frames.get_color_frame():
#             frames = self.align.process(frames)
#
#         depth = frames.get_depth_frame()
#         color = frames.get_color_frame()
#         if not depth or not color:
#             return False, None, None, None
#
#         depth_viz = np.asanyarray(self.colorizer.colorize(depth).get_data())
#         depth_m   = np.asanyarray(depth.get_data()).astype(np.float32) * self.depth_scale
#         color_bgr = np.asanyarray(color.get_data())
#         return True, depth_viz, color_bgr, depth_m
#
#     def release(self):
#         self.pipe.stop()
#
# if __name__ == "__main__":
#     dc = DepthCamera()
#     try:
#         while True:
#             ok, depth_viz, color_frame, depth_m = dc.get_frame()
#             if not ok:
#                 continue
#
#             point = (400, 300)  # (x, y)
#             z = float(depth_m[point[1], point[0]])  # meters; 0.0 if invalid
#             cv2.circle(color_frame, point, 4, (0, 0, 255), -1)
#             cv2.putText(color_frame, f"{z:.2f} m" if z > 0 else "no depth",
#                         (point[0] + 8, point[1] - 8),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#
#             cv2.imshow("Depth (colorized)", depth_viz)
#             cv2.imshow("Color", color_frame)
#             if cv2.waitKey(1) & 0xFF == 27:  # ESC
#                 break
#             if cv2.waitKey(1) & 0xFF == ord('q'): break
#     finally:
#         dc.release()
#         cv2.destroyAllWindows()
#
#
#######################################################
# import pyrealsense2 as rs, numpy as np, cv2
# from realsense_dep
# p = rs.pipeline(); c = rs.config()
# c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# p.start(c)
# colorizer = rs.colorizer()
# try:
#     while True:
#         f = p.wait_for_frames()
#         d = f.get_depth_frame()
#         if not d: continue
#         img = np.asanyarray(colorizer.colorize(d).get_data())
#         cv2.imshow("Depth only", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break
# finally:
#     p.stop(); cv2.destroyAllWindows()
######################################################
#
#
