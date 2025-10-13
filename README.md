# OpenRadars Micro-Doppler

## Overview
This project captures and processes radar and depth-camera data for micro-Doppler and gait analysis.  
It fuses TI IWR6843ISK mmWave radar data with Intel RealSense D435i depth frames to produce real-time and offline motion signatures.

The repository has been reorganized into a clean `src/` structure so modules and imports work correctly.

```
repo_root/
├─ src/
│  ├─ radar/
│  │   └─ realTimeProc_IWR6843ISK_1_1.py
│  ├─ depth/
│  │   └─ d435i_view.py
│  └─ mmwave/
│      ├─ dataloader/
│      └─ ...
├─ configFiles/
│   ├─ cf.json
│   └─ xWR1843_profile_3D.cfg
├─ requirements.txt
├─ run_radar.sh / run_radar.ps1
└─ run_d435i.sh / run_d435i.ps1
```

---

## Quickstart

### 1️⃣ Setup Environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # PowerShell
# or
source .venv/bin/activate      # Git Bash
pip install -r requirements.txt
```

### 2️⃣ Run the Live Radar Pipeline
```bash
./run_radar.sh
```
or on Windows PowerShell:
```powershell
.
un_radar.ps1
```

### 3️⃣ Run the Depth Camera Viewer
```bash
./run_d435i.sh
```
or on Windows PowerShell:
```powershell
.
un_d435i.ps1
```

---

## Hardware Requirements
- **TI IWR6843ISK + DCA1000EVM**
  - Connected via Ethernet; configured per `configFiles/*.cfg`.
  - Update COM ports in `src/radar/realTimeProc_IWR6843ISK_1_1.py`.
- **Intel RealSense D435i**
  - Requires Intel RealSense SDK and `pyrealsense2`.

---

## Notes
- `mmwave/` contains vendor-specific radar dataloader code.
- `configFiles/` must remain in the project root.
- Use the `.sh` or `.ps1` launchers to ensure correct `PYTHONPATH`.

---

## Research Purpose
This repository supports the **Smart Sensing and Computing for Dementia Care** project at Kennesaw State University.  
It focuses on developing an **RF-Skeleton & Micro Doppler extraction pipeline** for continuous, privacy-preserving motion monitoring using mmWave radar and depth sensors.  
The goal is to fuse radar micro-Doppler signatures and depth-based skeletal data to analyze gait patterns and assess fall risk in dementia patients.
