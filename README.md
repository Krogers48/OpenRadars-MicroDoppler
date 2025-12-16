# OpenRadars-MicroDoppler

A unified real-time radar + depth sensing pipeline for micro-Doppler visualization, radar-based gait analytics, and synchronized multi-sensor monitoring using:

* TI IWR6843ISK FMCW radar
* TI DCA1000EVM raw data capture board
* Intel RealSense D435i depth camera

This README reflects the **current architecture and behavior of `realTimeProc_IWR6843ISK_1_1.py`**, including recent refactors and the appended *End-to-End Script Capabilities* appendix.

---

## System Overview

The project is organized around **decoupled acquisition, processing, and visualization**:

* `realTimeProc_IWR6843ISK_1_1.py` is a **radar-only, real-time processing and gait-metrics engine**
* `realTimeProc_d435i_view.py` handles **depth + skeleton processing**
* `combined_viewer_app.py` is a **thin visualization client** driven entirely by ZMQ streams
* `session_start.py` orchestrates synchronized startup and session management

No visualization logic is required for metric computation.

---

## Requirements

### Hardware

* TI IWR6843ISK mmWave radar
* TI DCA1000EVM data capture board
* Intel RealSense D435i RGB-D camera
* Windows or Linux machine with Python 3.8+
* Ethernet port for DCA1000
* USB 3.0 for D435i

### Software

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows PowerShell
# or
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

---

## Quickstart

### Start All Sensors (Radar + DCA1000 + Depth)

```bash
python session_start.py
```

### What `session_start.py` Does

* Resets and configures the IWR6843ISK radar
* Starts the DCA1000 high-throughput capture thread
* Launches the radar processing pipeline (`realTimeProc_IWR6843ISK_1_1.py`)
* Launches the RealSense depth + skeleton pipeline
* Creates a timestamped session directory:

```
sessions/session_YYYY-MM-DD_HHMMSS/
    ├── radar/
    ├── depth/
    └── manifest.json
```

* Initializes ZMQ publishers:

  * Radar μD + tracking + gait → `tcp://127.0.0.1:5557`
  * Depth + skeleton → `tcp://127.0.0.1:5558`

---

### Open the Combined Viewer

In a second terminal:

```bash
python combined_viewer_app.py
```

### What `combined_viewer_app.py` Shows

* Live micro-Doppler spectrogram (from radar)
* Depth image with skeleton overlay (from RealSense)
* Radar-derived gait metrics rendered as HUD overlays

The viewer **does not perform any signal processing**. All metrics are computed upstream.

---

## Radar Processing Script (`realTimeProc_IWR6843ISK_1_1.py`)

This script is a **self-contained real-time radar gait engine**. It:

* Controls radar + DCA1000 hardware
* Parses raw ADC data
* Performs range FFT, Doppler FFT, MTI clutter removal
* Forms virtual arrays and performs azimuth beamforming
* Generates multi-range candidate detections
* Tracks a single walking subject using a 2D Kalman filter
* Computes walking speed from 2D motion (not Doppler-only)
* Extracts cadence, step timing, and temporal gait asymmetry from micro-Doppler
* Publishes results via ZMQ
* Optionally logs append-only CSV gait metrics
* Writes per-bout gait summaries on shutdown

### Important Clarifications

* Walking speed is derived from **range + azimuth tracking**, not raw Doppler
* Cadence and asymmetry are derived from **micro-Doppler leg motion**, not tracking
* The script does **not** implement CFAR, DBSCAN, or full range–azimuth heatmaps
* Ghost suppression relies on MTI, Doppler masking, confidence gating, and tracking consistency

---


**Appendix – End-to-End Script Capabilities**:

* Hardware control and capture
* Signal processing pipeline
* Azimuth AoA and candidate detection
* Tracking and walking speed estimation
* Cadence, step timing, and asymmetry computation
* Outputs, logging, and data products

---

## Logging Output

Each session produces:

```
sessions/session_TIMESTAMP/
    ├── radar/
    │   ├── session_meta.json
    │   ├── radar_gait_metrics.csv      (if enabled)
    │   └── radar_gait_summary.csv
    └── depth/
        ├── skeleton_run_meta.json
        └── skeleton_gait_metrics.csv
```

Raw ADC `.npy` saving is supported but disabled by default.

---

## Data Streams (ZMQ)

| Stream          | Address              | Description                                      |
| --------------- | -------------------- | ------------------------------------------------ |
| Radar μD + Gait | tcp://127.0.0.1:5557 | μD columns, range energy, tracking, gait metrics |
| Depth/Skeleton  | tcp://127.0.0.1:5558 | RGB-D frames with skeleton overlays              |

---

## Purpose

This repository supports the **Smart Sensing and Computing for Dementia Care** project at Kennesaw State University.

The goal is to develop a **privacy-preserving, continuous gait monitoring system** by fusing:

* Radar micro-Doppler signatures
* Radar-based 2D motion tracking
* Depth-based skeletal motion

for assessing mobility decline, fall risk, and cognitive health.

---

## License

MIT License
© 2025
