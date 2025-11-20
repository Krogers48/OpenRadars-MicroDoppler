#!/usr/bin/env python3
"""
start_session.py — launches radar and depth with a unified session root.

Usage:
  python start_session.py
  python start_session.py --fps 30

Notes:
- Uses sys.executable so your current venv is honored.
- Creates sessions/session_<timestamp>/{radar,depth} relative to this file.
- Depth gets --session-dir <root>; it writes into <root>/depth/ itself.
- Radar gets  --session-dir <root>/radar --auto-start  (skips ENTER prompt).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from subprocess import Popen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30, help="Depth camera FPS to pass to realTimeProc_d435i_view.py")
    ap.add_argument("--radar-script", default="realTimeProc_IWR6843ISK_1_1.py")
    ap.add_argument("--depth-script", default="realTimeProc_d435i_view.py")
    ap.add_argument("--sessions-dir", default="sessions")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    # Session root
    ts = time.strftime("%Y-%m-%d_%H%M%S")
    session_root = root / args.sessions_dir / f"session_{ts}"
    radar_dir = session_root / "radar"
    depth_dir = session_root / "depth"
    radar_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Manifest
    manifest = {
        "session_id": ts,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "radar_dir": str(radar_dir.relative_to(root)),
        "depth_dir": str(depth_dir.relative_to(root)),
        "radar_pub": "tcp://127.0.0.1:5557",
        "depth_pub": "tcp://127.0.0.1:5558",
        "invocation": {
            "python": sys.executable,
            "radar_script": args.radar_script,
            "depth_script": args.depth_script,
            "depth_fps": args.fps
        }
    }
    (session_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Commands (relative; sys.executable honors your venv)
    radar_cmd = [
        sys.executable,
        str(root / args.radar_script),
        "--session-dir", str(radar_dir.relative_to(root)),
        "--auto-start",                              # <-- new flag
    ]
    depth_cmd = [
        sys.executable,
        str(root / args.depth_script),
        "--pub", "--headless", #"--no-mp" #"--log-wide",
        "--fps", str(args.fps),
        "--session-dir", str(session_root.relative_to(root)),
    ]

    print(f"[start] session: {session_root.relative_to(root)}")
    print(f"[start] radar : {' '.join(radar_cmd)}")
    print(f"[start] depth : {' '.join(depth_cmd)}")

    radar_proc = Popen(radar_cmd, cwd=str(root))
    depth_proc = Popen(depth_cmd, cwd=str(root))

    print("[start] processes running.")
    print("      Open another terminal and run:  python combined_viewer_app.py")
    print("      Ctrl+C here to stop this launcher (children keep running).")


    # Spawns two subprocesses, one for each script.
    # Both run in the same directory (cwd=str(root)).
    # They’re launched in the background and run concurrently.
    try:
        while True:
            r_code = radar_proc.poll()
            d_code = depth_proc.poll()
            if r_code is not None or d_code is not None:
                print(f"[start] radar exit={r_code}  depth exit={d_code}")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
