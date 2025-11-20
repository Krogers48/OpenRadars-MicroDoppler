# OpenRadars-MicroDoppler

A unified real-time radar + depth sensing pipeline for Micro-Doppler
visualization, gait analytics, and synchronized multi‑sensor monitoring
using:

-   TI IWR6843ISK FMCW Radar\
-   TI DCA1000EVM raw data capture board\
-   Intel RealSense D435i depth camera

This README reflects the updated workflow using
**session_start.py** and **combined_viewer_app.py**.

------------------------------------------------------------------------

## Requirements

### Hardware

-   TI IWR6843ISK mmWave radar\
-   TI DCA1000EVM data capture board\
-   Intel RealSense D435i RGB‑D camera\
-   Windows / Linux machine with Python 3.8+\
-   USB 3.0 port for D435i\
-   Ethernet port for DCA1000

### Software

``` bash
python -m venv .venv
source .venv/Scripts/activate      # Windows PowerShell
# or
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Quickstart

### Start All Sensors (Radar + DCA1000 + Depth Camera)

Run:

``` bash
python session_start.py
```

### What `session_start.py` Does

-   Initializes and configures the **IWR6843ISK** radar\

-   Starts DCA1000 raw-capture thread\

-   Launches the full radar processing pipeline\

-   Starts the Intel RealSense D435i pipeline\

-   Creates a timestamped session directory:

        sessions/session_YYYY-MM-DD_HHMMSS/
            ├── radar/
            ├── depth/
            └── manifest.json

-   Sets up ZMQ publishers:

    -   Radar μD stream → tcp://127.0.0.1:5557\
    -   Depth/skeleton stream → tcp://127.0.0.1:5558

Both pipelines begin streaming automatically.

------------------------------------------------------------------------

### Open the Combined Viewer

After launching session_start.py in one terminal, open another and run:

``` bash
python combined_viewer_app.py
```

### What `combined_viewer_app.py` Does

-   Subscribes to the radar μD ZMQ stream\
-   Subscribes to the depth+skeleton ZMQ stream\
-   Synchronizes timestamps between radar and depth\
-   Displays:
    -   **Micro-Doppler spectrogram** (radar)
    -   **Depth image + 3D skeleton** (RealSense)
-   Renders radar-derived gait metrics on the μD HUD\
-   Produces a unified real-time visualization window


------------------------------------------------------------------------

## Project Structure

    OpenRadars-MicroDoppler/
    │
    ├── session_start.py              # Launches radar + depth pipelines together
    ├── combined_viewer_app.py        # Unified dual-sensor viewer
    │
    ├── realTimeProc_IWR6843ISK_1_1.py   # Radar processing pipeline
    ├── realTimeProc_d435i_view.py       # Depth + skeleton + gait extraction
    │
    ├── configFiles/                  # Radar .cfg files
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## Logging Output

Each session creates:

    sessions/session_TIMESTAMP/
        ├── radar/
        │   ├── radar_md_timestamps.csv
        │   ├── session_meta.json
        │   └── micro_doppler_time.png
        │
        └── depth/
            ├── skeleton_run_meta.json
            ├── wigait_skeleton_metrics.csv   (if logging enabled)
            └── skeleton_gait_metrics.csv


------------------------------------------------------------------------

## Data Streams (ZMQ)

  -----------------------------------------------------------------------
  Stream             Address                Description
  ------------------ ---------------------- -----------------------------
  Radar μD           tcp://127.0.0.1:5557   micro-Doppler columns, range
                                            energy

  Depth/Skeleton     tcp://127.0.0.1:5558   color+depth frame with
                                            skeleton overlays

  Radar Gait Metrics tcp://127.0.0.1:5559   (if enabled) gait metrics
                                            over time
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Purpose

The goal of this project is to create a reliable, synchronized,
real-time sensing system for:

-   Radar micro‑Doppler visualization\
-   Skeleton-based motion analysis\
-   Combined gait metrics (speed, stride length, cadence)\
-   Assistive health monitoring and research


## License

MIT License\
© 2025


## Research Purpose
This repository supports the **Smart Sensing and Computing for Dementia Care** project at Kennesaw State University.  
It focuses on developing an **RF-Skeleton & Micro Doppler extraction pipeline** for continuous, privacy-preserving motion monitoring using mmWave radar and depth sensors.  
The goal is to fuse radar micro-Doppler signatures and depth-based skeletal data to analyze gait patterns and assess fall risk in dementia patients.
