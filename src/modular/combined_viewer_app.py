# combined_viewer_app.py
# ZMQ radar + depth viewer. Drains all μD columns each tick. Depth image fills its half.

import sys
import time
from collections import deque

import numpy as np
import zmq
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from realTimeProc_IWR6843ISK_1_1 import MDViewer  # import the MD viewer class

LIVE_SOFT_FLOOR = False
LIVE_BASELINE_TAU = 0.995
DOPPLER_SMOOTH_K = 1
FRAME_EWMA = 0.00

# Purely visual: repeat each μD column horizontally without changing history length.
COL_REPEAT_X = 2
DEFAULT_MD_HIST = 512

# Timestamp pairing controls (radar ↔ depth)
SYNC_OFFSET_S = 0.000    # depth_ts ≈ radar_ts - SYNC_OFFSET_S
SYNC_MAX_SKEW_S = 0.080  # require |depth_ts - (radar_ts - SYNC_OFFSET_S)| ≤ this


class CombinedViewer(QtWidgets.QMainWindow):
    def __init__(self, hist=DEFAULT_MD_HIST):
        super().__init__()
        self.setWindowTitle("Radar + Depth Combined Viewer (ZMQ)")
        self.resize(1400, 800)

        # Left: μD viewer (Nd corrected on first packet)
        self.radar_viewer = MDViewer(Nd_eff=128, hist=hist)
        self.radar_viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Right: depth/skeleton — QLabel that always fills its half with the incoming image
        self.depth_label = QtWidgets.QLabel()
        self.depth_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.depth_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_label.setScaledContents(True)

        # Side-by-side layout via splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.radar_viewer)
        splitter.addWidget(self.depth_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])
        self.setCentralWidget(splitter)

        # Buffers
        self.radar_buffer = deque(maxlen=32768)   # (ts, col_db[Nd])
        # Now store full depth message dict ({"ts", "image", optional "gait"}) for sync diagnostics
        self.skeleton_buffer = deque(maxlen=512)  # (ts, msg dict)

        # State
        self._live_floor = None
        self._last_col = None
        self._last_gait_metrics = None        # (v, f, cadence, SL) from radar WiGait
        self._last_depth_gait_metrics = None  # (v, f, cadence, SL) from depth WiGait (if present)
        self._last_sync_dt_ms = None          # depth–radar time offset in ms

        # Robust level-control state (for μD color scale; replaces autoLevels)
        self._lev_lo = None
        self._lev_hi = None
        self._lev_alpha = 0.05    # EMA smoothing factor for levels
        self._lev_p_lo = 5.0      # lower percentile
        self._lev_p_hi = 99.5     # upper percentile
        self._lev_freeze = False  # if True, stop updating levels
        self._vel_axis = None     # vel_mps (Nd,) from radar, used to gate percentiles

        # ZMQ
        self._start_zmq_threads()

        # UI timer ~100 Hz
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._sync_and_display)
        self.timer.start(5)

    # ---------- ZMQ ----------
    def _start_zmq_threads(self):
        ctx = zmq.Context.instance()

        # Radar μD columns
        self.rad_sub = ctx.socket(zmq.SUB)
        self.rad_sub.connect("tcp://127.0.0.1:5557")
        # Raise HWM so we don't drop μD columns in bursts (subscriber side)
        self.rad_sub.setsockopt(zmq.RCVHWM, 100000)
        self.rad_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.rad_thread = QtCore.QThread(self)
        self.rad_worker = _ZMQListener(self.rad_sub, kind="radar")
        self.rad_worker.moveToThread(self.rad_thread)
        self.rad_worker.new_data.connect(self._on_new_radar)
        self.rad_thread.started.connect(self.rad_worker.run)
        self.rad_thread.start()

        # Depth / skeleton images (+ optional gait)
        self.skel_sub = ctx.socket(zmq.SUB)
        self.skel_sub.connect("tcp://127.0.0.1:5558")
        self.skel_sub.setsockopt(zmq.RCVHWM, 10000)
        self.skel_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.skel_thread = QtCore.QThread(self)
        self.skel_worker = _ZMQListener(self.skel_sub, kind="skeleton")
        self.skel_worker.moveToThread(self.skel_thread)
        self.skel_worker.new_data.connect(self._on_new_skeleton)
        self.skel_thread.started.connect(self.skel_worker.run)
        self.skel_thread.start()

        # Radar WiGait gait metrics (tcp://127.0.0.1:5559)
        self.gait_sub = ctx.socket(zmq.SUB)
        self.gait_sub.connect("tcp://127.0.0.1:5559")
        self.gait_sub.setsockopt(zmq.RCVHWM, 10000)
        self.gait_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.gait_thread = QtCore.QThread(self)
        self.gait_worker = _ZMQListener(self.gait_sub, kind="gait")
        self.gait_worker.moveToThread(self.gait_thread)
        self.gait_worker.new_data.connect(self._on_new_gait)
        self.gait_thread.started.connect(self.gait_worker.run)
        self.gait_thread.start()

    @QtCore.Slot(float, object, str)
    def _on_new_radar(self, ts, data, _kind):
        """
        Radar payload from realTimeProc_IWR6843ISK_1_1.py:
          {"ts", "md_db", "range", "vel_mps"}
        We keep the μD column for rendering, and cache vel_mps for robust-level gating.
        """
        try:
            if isinstance(data, dict):
                md = np.asarray(data.get("md_db"), dtype=np.float32)
                vel = data.get("vel_mps", None)
                if vel is not None:
                    v = np.asarray(vel, dtype=np.float32)
                    if v.ndim == 1:
                        self._vel_axis = v
            else:
                md = np.asarray(data, dtype=np.float32)
            self.radar_buffer.append((float(ts), md))
        except Exception:
            # If anything goes wrong, fall back to previous behaviour
            try:
                self.radar_buffer.append((float(ts), np.asarray(data, dtype=np.float32)))
            except Exception:
                pass

    @QtCore.Slot(float, object, str)
    def _on_new_skeleton(self, ts, data, _kind):
        # Store full depth message dict (includes image and optional "gait")
        self.skeleton_buffer.append((float(ts), data))

    @QtCore.Slot(float, object, str)
    def _on_new_gait(self, ts, data, _kind):
        """Receive WiGait gait metrics dict from radar (tcp://127.0.0.1:5559)."""
        try:
            m = data.get("metrics", {}) if isinstance(data, dict) else {}
            self._last_gait_metrics = (
                float(m.get("mean_speed_mps", 0.0)),
                float(m.get("stride_freq_hz", 0.0)),
                float(m.get("cadence_spm", 0.0)),
                float(m.get("mean_stride_len_m", 0.0)),
            )
        except Exception:
            self._last_gait_metrics = None

    # ---------- shaping ----------
    def _apply_live_soft_floor_lin(self, col_lin: np.ndarray) -> np.ndarray:
        if not LIVE_SOFT_FLOOR:
            return col_lin
        if self._live_floor is None:
            self._live_floor = col_lin.copy()
        else:
            self._live_floor = LIVE_BASELINE_TAU * self._live_floor + (1.0 - LIVE_BASELINE_TAU) * col_lin
        out = col_lin - self._live_floor
        out[out < 0.0] = 0.0
        return out

    def _doppler_smooth(self, col_db: np.ndarray) -> np.ndarray:
        k = int(max(1, int(DOPPLER_SMOOTH_K)))
        if k <= 1:
            return col_db
        if (k % 2) == 0:
            k += 1
        ker = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(col_db, ker, mode='same').astype(np.float32)

    def _frame_ewma(self, col_db: np.ndarray) -> np.ndarray:
        a = float(FRAME_EWMA)
        if not (0.0 < a < 1.0):
            self._last_col = col_db.copy()
            return col_db
        if self._last_col is None:
            self._last_col = col_db.copy()
            return col_db
        out = a * self._last_col + (1.0 - a) * col_db
        self._last_col = out.copy()
        return out

    def _update_radar_hud(self):
        """
        Update the radar Text HUD with WiGait metrics + optional Δt diagnostics.

        Uses radar gait metrics from _last_gait_metrics and the most recent
        depth–radar sync offset (self._last_sync_dt_ms), if available.
        """
        if self._last_gait_metrics is None:
            return
        v, f, cad, sl = self._last_gait_metrics
        # Match depth-style text, then append Δt if known.
        txt = f"Gait : v = {v:4.2f} m/s "  # f = {f:4.2f} Hz, cadence, SL omitted in HUD text
        if self._last_sync_dt_ms is not None:
            txt += f"  Δt={self._last_sync_dt_ms:+d} ms"
        self.radar_viewer.set_gait_text(txt)

    # ---------- robust levels ----------
    def _update_levels_from_column(self, col_db2: np.ndarray):
        """
        Robust, smoothed color levels to eliminate autoLevels flicker.
        Uses percentiles and EMA; optionally restricts to human-speed band using vel_mps.
        """
        if self._lev_freeze:
            return
        try:
            arr = np.asarray(col_db2, dtype=np.float32)
            # if self._vel_axis is not None and self._vel_axis.shape[0] == arr.shape[0]:
            #     # Optional: restrict to human band |v| <= 4 m/s for level estimation
            #     mask = (np.abs(self._vel_axis) <= 4.0)
            #     if np.any(mask):
            #         arr = arr[mask]
            lo = float(np.percentile(arr, self._lev_p_lo))
            hi = float(np.percentile(arr, self._lev_p_hi))
            if self._lev_lo is None:
                self._lev_lo, self._lev_hi = lo, hi
            else:
                a = self._lev_alpha
                self._lev_lo = (1.0 - a) * self._lev_lo + a * lo
                self._lev_hi = (1.0 - a) * self._lev_hi + a * hi
            if hasattr(self.radar_viewer, "set_levels"):
                self.radar_viewer.set_levels(self._lev_lo, self._lev_hi)
        except Exception:
            pass

    # ---------- render ----------
    def _sync_and_display(self):
        last_ts = None

        # Drain radar buffer and push μD columns into MDViewer
        while self.radar_buffer:
            tr, col_db = self.radar_buffer.popleft()
            last_ts = tr

            col_lin = np.power(10.0, col_db / 10.0, dtype=np.float32)
            col_lin = self._apply_live_soft_floor_lin(col_lin)
            col_db2 = 10.0 * np.log10(np.maximum(col_lin, 1e-12)).astype(np.float32)
            col_db2 = self._doppler_smooth(col_db2)
            col_db2 = self._frame_ewma(col_db2)

            # Update robust image levels based on this column (no autoLevels flicker)
            self._update_levels_from_column(col_db2)

            # If Nd changes, rebuild the MDViewer to match
            if col_db2.shape[0] != self.radar_viewer.Nd:
                new_viewer = MDViewer(Nd_eff=col_db2.shape[0], hist=self.radar_viewer.hist)
                splitter = self.centralWidget()
                idx = splitter.indexOf(self.radar_viewer)
                splitter.insertWidget(idx, new_viewer)
                self.radar_viewer.deleteLater()
                self.radar_viewer = new_viewer

            repeat_n = max(1, int(COL_REPEAT_X))
            for _ in range(repeat_n):
                self.radar_viewer.append_col(col_db2)

        # Depth: pair by timestamp with skew limit and optional offset
        if self.skeleton_buffer:
            img = None
            gait_depth = None

            if last_ts is None:
                # No radar yet: just use the latest depth frame; no Δt diagnostics.
                ts_depth, msg = self.skeleton_buffer[-1]
                self._last_sync_dt_ms = None
            else:
                target_ts = float(last_ts) - float(SYNC_OFFSET_S)
                # Find depth packet closest in time to target_ts
                ts_depth, msg = min(self.skeleton_buffer, key=lambda x: abs(x[0] - target_ts))
                dt = ts_depth - target_ts
                if abs(dt) <= float(SYNC_MAX_SKEW_S):
                    # Accept this pair and record Δt (depth - (radar - offset)) in ms
                    self._last_sync_dt_ms = int(round(dt * 1000.0))
                else:
                    # Too much skew: keep prior image/Δt; skip update this tick
                    msg = None

            if msg is not None:
                # Full depth packet: {"ts": ..., "image": overlay, optional "gait": {...}}
                img = msg.get("image", None)
                gait_depth = msg.get("gait", None)

                # Cache depth gait metrics if present (currently not drawn, but available for diagnostics)
                if gait_depth is not None and isinstance(gait_depth, dict):
                    try:
                        m = gait_depth.get("metrics", {})
                        self._last_depth_gait_metrics = (
                            float(m.get("mean_speed_mps", 0.0)),
                            float(m.get("stride_freq_hz", 0.0)),
                            float(m.get("cadence_spm", 0.0)),
                            float(m.get("mean_stride_len_m", 0.0)),
                        )
                    except Exception:
                        self._last_depth_gait_metrics = None

            if img is not None:
                try:
                    h, w = img.shape[:2]
                    # Convert BGR (OpenCV) to RGB for Qt and display in QLabel
                    rgb = img[:, :, ::-1].copy()
                    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
                    pixmap = QtGui.QPixmap.fromImage(qimg)

                    # IMPORTANT: do NOT overlay any extra text here.
                    # The depth script realTimeProc_d435i_view.py already draws:
                    #   - Ref: cadence/ST/SL   (green)
                    #   - WiGait: v/f/cad/SL   (cyan)
                    # So we just show the image as-is.
                    self.depth_label.setPixmap(pixmap)
                except Exception:
                    pass

        # After depth pairing, update radar HUD with latest gait metrics + Δt
        self._update_radar_hud()

    # clean exit
    def closeEvent(self, event):
        for worker, thread in [
            (self.rad_worker, self.rad_thread),
            (self.skel_worker, self.skel_thread),
            (self.gait_worker, self.gait_thread),
        ]:
            try:
                worker.running = False
            except Exception:
                pass
            thread.quit()
            thread.wait()
        super().closeEvent(event)


class _ZMQListener(QtCore.QObject):
    new_data = QtCore.Signal(float, object, str)

    def __init__(self, socket, kind="radar"):
        super().__init__()
        self.socket = socket
        self.kind = kind
        self.running = True

    def run(self):
        while self.running:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                ts = float(msg.get("ts", time.time()))
                if self.kind == "radar":
                    # Pass full radar message (md_db + range + vel_mps) so CombinedViewer can use vel_mps.
                    self.new_data.emit(ts, msg, self.kind)
                elif self.kind == "skeleton":
                    # Pass full depth/skeleton message (image + optional gait) for sync diagnostics.
                    self.new_data.emit(ts, msg, self.kind)
                elif self.kind == "gait":
                    # WiGait gait metrics dict from radar
                    self.new_data.emit(ts, msg, self.kind)
                else:
                    # Fallback: emit raw message object
                    self.new_data.emit(ts, msg, self.kind)
            except zmq.Again:
                time.sleep(0.005)
            except Exception:
                time.sleep(0.005)


if __name__ == "__main__":
    pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder='row-major')
    app = QtWidgets.QApplication(sys.argv)
    win = CombinedViewer(hist=DEFAULT_MD_HIST)
    win.show()
    sys.exit(app.exec())
