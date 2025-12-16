# combined_viewer_app.py
# ZMQ radar + depth viewer. Drains all μD columns each tick. Depth image fills its half.

import sys
import time
from collections import deque

import numpy as np
import zmq
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets, QtGui

from realTimeProc_IWR6843ISK_1_1 import MDViewer

LIVE_SOFT_FLOOR = False
LIVE_BASELINE_TAU = 0.995
DOPPLER_SMOOTH_K = 1
FRAME_EWMA = 0.00

# Purely visual: repeat each μD column horizontally without changing history length.
COL_REPEAT_X = 2
DEFAULT_MD_HIST = 512


class CombinedViewer(QtWidgets.QMainWindow):
    def __init__(self, hist=DEFAULT_MD_HIST):
        super().__init__()
        self.setWindowTitle("Radar + Depth Combined Viewer (ZMQ)")
        self.resize(1400, 800)

        # Left: μD viewer (Nd corrected on first packet)
        self.radar_viewer = MDViewer(Nd_eff=128, hist=hist)
        self.radar_viewer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Right: depth/skeleton — switch to QLabel that always fills
        self.depth_label = QtWidgets.QLabel()
        self.depth_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.depth_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_label.setScaledContents(True)  # key: fill the widget

        # Use a splitter to guarantee side-by-side layout. Give both sides equal stretch.
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.radar_viewer)
        splitter.addWidget(self.depth_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        # Start half-and-half. On window resizes, Qt honors stretch factors.
        splitter.setSizes([1, 1])
        self.setCentralWidget(splitter)

        # Buffers
        self.radar_buffer = deque(maxlen=32768)  # (ts, col_db[Nd])
        self.skeleton_buffer = deque(maxlen=512)  # (ts, image ndarray HxWx3)

        # State
        self._live_floor = None
        self._last_col = None

        # ZMQ
        self._start_zmq_threads()

        # UI timer ~100 Hz
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._sync_and_display)
        self.timer.start(5)

    # ---------- ZMQ ----------
    def _start_zmq_threads(self):
        ctx = zmq.Context.instance()

        self.rad_sub = ctx.socket(zmq.SUB)
        self.rad_sub.connect("tcp://127.0.0.1:5557")
        self.rad_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.rad_thread = QtCore.QThread(self)
        self.rad_worker = _ZMQListener(self.rad_sub, kind="radar")
        self.rad_worker.moveToThread(self.rad_thread)
        self.rad_worker.new_data.connect(self._on_new_radar)
        self.rad_thread.started.connect(self.rad_worker.run)
        self.rad_thread.start()

        self.skel_sub = ctx.socket(zmq.SUB)
        self.skel_sub.connect("tcp://127.0.0.1:5558")
        self.skel_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.skel_thread = QtCore.QThread(self)
        self.skel_worker = _ZMQListener(self.skel_sub, kind="skeleton")
        self.skel_worker.moveToThread(self.skel_thread)
        self.skel_worker.new_data.connect(self._on_new_skeleton)
        self.skel_thread.started.connect(self.skel_worker.run)
        self.skel_thread.start()

    @QtCore.Slot(float, object, str)
    def _on_new_radar(self, ts, data, _kind):
        self.radar_buffer.append((float(ts), np.asarray(data, dtype=np.float32)))

    @QtCore.Slot(float, object, str)
    def _on_new_skeleton(self, ts, data, _kind):
        self.skeleton_buffer.append((float(ts), data))

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

    # ---------- render ----------
    def _sync_and_display(self):
        last_ts = None

        # drain radar
        while self.radar_buffer:
            tr, col_db = self.radar_buffer.popleft()
            last_ts = tr

            col_lin = np.power(10.0, col_db / 10.0, dtype=np.float32)
            col_lin = self._apply_live_soft_floor_lin(col_lin)
            col_db2 = 10.0 * np.log10(np.maximum(col_lin, 1e-12)).astype(np.float32)
            col_db2 = self._doppler_smooth(col_db2)
            col_db2 = self._frame_ewma(col_db2)

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

        # depth: nearest to last radar ts, else latest
        if self.skeleton_buffer:
            if last_ts is None:
                _, img = self.skeleton_buffer[-1]
            else:
                _, img = min(self.skeleton_buffer, key=lambda x: abs(x[0] - last_ts))

            try:
                h, w = img.shape[:2]
                # Convert BGR (OpenCV) to RGB for Qt and display in QLabel
                rgb = img[:, :, ::-1].copy()
                qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
                self.depth_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
            except Exception:
                pass

    # clean exit
    def closeEvent(self, event):
        for worker, thread in [(self.rad_worker, self.rad_thread), (self.skel_worker, self.skel_thread)]:
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
                    self.new_data.emit(ts, np.asarray(msg["md_db"], dtype=np.float32), self.kind)
                else:
                    self.new_data.emit(ts, msg.get("image", None), self.kind)
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
