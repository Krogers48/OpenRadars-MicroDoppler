# -*- coding: utf-8 -*-
# Smart Sensing and Computing for Dementia Care
# The goal of this project is to develop sensing systems to collect data regarding risk assessment and prevention.
#
# realTimeProc_IWR6843ISK_1_1.py
#
# Current goals:
#   - Keep processing pure: postProc performs signal processing only (no ZMQ / file I/O / UI).
#   - Compute micro-Doppler columns for cadence/asymmetry extraction (μD pipeline).
#   - Compute walking speed from RADAR-ONLY tracking:
#         range + azimuth (beamforming/AoA) -> (x,y) measurements -> 2D tracking -> walking speed.
#
# Steps implemented (from plan):
#   1) Report speed as track-derived walking speed (sqrt(vx^2+vy^2)), not Doppler radial speed.
#      (Optional) Doppler radial speed can be computed only as QC/debug.
#   2) Beamforming: explicit azimuth virtual-channel selection + element positions + optional (bias/scale) calibration.
#   3) Detection: generate per-frame candidate detections over multiple range bins (approx range–az detection),
#      instead of a single snapshot at r_center.
#   4) Tracking: single-target CV Kalman filter with multi-candidate association (Mahalanobis gating) and speed smoothing.
#
# Important notes / assumptions:
#   - Monostatic Doppler alone gives radial (LOS) speed. We do NOT report that as "walking speed".
#   - Walking speed here is derived from 2D tracking and therefore depends on AoA quality and stable association.
#   - Antenna geometry / virtual-channel ordering is hardware-specific. Defaults included are a reasonable
#     starting point for xWR6843 with 4 RX and (>=2) TX (azimuth MIMO), but MUST be validated/calibrated.

from __future__ import annotations

# ---------- Standard / scientific ----------
import argparse
import csv
import json
import os
import time
import traceback
import hashlib
import platform
import subprocess
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# ---- Optional: SciPy for μD step-cycle detection (cadence) ----
try:
    from scipy.signal import butter, filtfilt, find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    butter = filtfilt = find_peaks = None
    SCIPY_AVAILABLE = False

# ---- Optional: ZMQ for live publishing (used by combined_viewer_app.py) ----
try:
    import zmq
    ZMQ_AVAILABLE = True
except Exception:
    zmq = None
    ZMQ_AVAILABLE = False

# ---- Optional: PySide6/pyqtgraph viewer (imported by combined_viewer_app.py) ----
try:
    import pyqtgraph as pg
    from PySide6 import QtWidgets, QtCore
    QT_AVAILABLE = True
except Exception:
    pg = None
    QtWidgets = QtCore = None
    QT_AVAILABLE = False


# ========================= Runtime defaults =========================
RADAR_PUB_ADDR = "tcp://127.0.0.1:5557"
# Change DEMO_MODE && DEMO_DISABLE_NPY_SAVES = False
# to save adc_loop.npy to session folder
DEMO_MODE: bool = True
DEMO_FRAMEBUF: int = 8                  # smaller UDP batch buffer (less bursty)
DEMO_NUMFRAMES: int = 8                 # numframes <= frameNumInBuf
DEMO_PUB_THIN: int = 1                  # publish every k-th μD column (1 = no thinning)
DEMO_DISABLE_NPY_SAVES: bool = True

EPS = 1e-12


# ========================= Signal processing tunables =========================
MTI_CANCELLER = 2        # 0: mean-subtract, 1: 2-pulse, 2: 3-pulse
NOISE_SUBTRACT_Q = 0.10  # quantile across range used as per-Doppler noise floor (0..1)
DOPPLER_SMOOTH_K = 1     # moving-average across Doppler bins (odd; 1 disables)
FRAME_EWMA = 0.00        # EWMA across frames within a batch (0 disables)

# Range-collapse weighting for μD (Gaussian around detected range bin)
RANGE_SIGMA_M = 0.8


# ========================= Beamforming / AoA (azimuth) =========================
BEAMFORM_ENABLE = True
BF_ANGLE_MIN_DEG = -70.0
BF_ANGLE_MAX_DEG = 70.0
BF_ANGLE_STEP_DEG = 2.0

# Default ULA assumption: element spacing in wavelengths (λ units).
BF_D_OVER_LAMBDA = 0.5

# Virtual channel selection for azimuth beamforming (indices into virtual-array axis).
# If None, a hardware-aware default is chosen when possible (e.g., 6843 w/4 RX uses first 2 TX -> 8 channels).
AZIM_VCHAN_IDXS: Optional[Tuple[int, ...]] = None

# Optional explicit element positions (in wavelengths) for the chosen channels.
# If provided, length must match len(AZIM_VCHAN_IDXS) (or V if AZIM_VCHAN_IDXS ends up using all channels).
AZIM_POS_WAVELENGTHS: Optional[np.ndarray] = None

# Optional linear calibration on azimuth (deg): az_cal = AZ_CAL_SCALE * az_raw + AZ_CAL_BIAS_DEG
AZ_CAL_SCALE: float = 1.0
AZ_CAL_BIAS_DEG: float = 0.0

# AoA quality thresholds
BF_MIN_SNAPSHOT_PWR = 1e-6    # absolute snapshot power threshold (scale-dependent)
BF_MIN_CONF = 3.0             # beam peak / median threshold


# ========================= Range–Azimuth candidate detection =========================
DETECT_ENABLE = True
DETECT_TOPK_RANGES = 10                 # evaluate this many range bins per frame
DETECT_E_FRAC_OF_MAX = 0.30             # keep range bins with E >= frac*maxE
DETECT_E_ABS_MIN = 0.0                  # optional absolute minimum range energy
DETECT_RANGE_MIN_M = 0.5                # ignore detections closer than this
DETECT_RANGE_MAX_M = 10.0               # ignore detections farther than this
DETECT_DOP_NEIGH_BINS = 1               # use peak Doppler bin +/- this many bins for AoA integration
DETECT_MIN_RANGE_BIN_SEP = 2            # suppress near-duplicate range bins (non-max suppression)

DETECT_CLUSTER_ENABLE = True            # simple clustering in XY (reduces duplicate range bins)
DETECT_CLUSTER_DIST_M = 0.45


# ========================= Tracking: single target 2D Cartesian =========================
TRACK_ENABLE = True
TRACK_SIGMA_A = 1.5               # process accel std (m/s^2)
TRACK_INIT_POS_STD = 1.0          # initial pos std (m)
TRACK_INIT_VEL_STD = 1.5          # initial vel std (m/s)
TRACK_MAX_MISS_S = 1.5            # drop track after this much time without measurement
TRACK_MAHA_GATE2 = 9.21           # chi2 2D ~99% (2 dof)

# Baseline measurement noise (converted to x/y covariance); angle noise is confidence-adaptive.
TRACK_MEAS_RANGE_STD_M = 0.20
TRACK_MEAS_ANGLE_STD_DEG = 8.0
TRACK_MEAS_ANGLE_STD_MIN_DEG = 2.0
TRACK_MEAS_ANGLE_STD_MAX_DEG = 25.0

# Speed smoothing (EMA) on output (does not change state; only reported metric)
TRACK_SPEED_EMA_ALPHA = 0.85


# ========================= Optional Doppler radial-speed QC =========================
RADIAL_QC_ENABLE = False  # Set True to compute Doppler radial speed as a debug/QC metric only (NOT walking speed)
RADIAL_QC_V_MIN_MPS = 0.12
RADIAL_QC_V_MAX_MPS = 2.2
RADIAL_QC_PEAK_NEIGHBOR_BINS = 2
RADIAL_QC_MIN_CONF = 3.0
RADIAL_QC_EMA_ALPHA = 0.85


# ========================= μD cadence and asymmetry =========================
GAIT_DETECT_ENABLE = True
GAIT_DETECT_WINDOW_S = 6.0
GAIT_DETECT_BAND_HZ = (0.8, 3.0)          # Hz
GAIT_DETECT_MIN_STEP_S = 0.30             # s
GAIT_DETECT_MAX_STEP_S = 1.20             # s
GAIT_DETECT_BIN_MIN_FRAC = 0.08           # exclude bins near 0-Doppler (torso / clutter)
GAIT_DETECT_BIN_MAX_FRAC = 0.45           # exclude extreme edges (often noisy)
GAIT_DETECT_COMPUTE_EVERY_S = 0.25        # recompute cadence at most this often
GAIT_DETECT_PROM_MULT = 0.50              # peak prominence = PROM_MULT * std(filtered)
GAIT_DETECT_USE_SYNTH_TS = True           # correct batch timestamp compression
GAIT_DETECT_DEFAULT_FRAME_PERIOD_S = 0.050  # fallback if cfg doesn't expose frame period

ASYM_ENABLE = True
ASYM_WINDOW_STEPS = 20
ASYM_MIN_INTERVALS = 10  # require this many step intervals before reporting AI


# ---------- Small caches ----------
_WIN_CACHE: Dict[Tuple[str, int], np.ndarray] = {}           # keys: ('wr', Nr), ('wd', Nd_eff)
_AXES_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}
_BF_CACHE: Dict[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]] = {}  # (angles_rad, W) cache


def _frame_period_s_from_params(params: dict, default_s: float) -> float:
    """Best-effort frame period (seconds) from TI cfg-derived params."""
    keys_ms = [
        "framePeriodicity_ms", "frame_periodicity_ms", "framePeriodicity", "frame_periodicity",
        "frameTimeMs", "frame_time_ms", "frame_ms"
    ]
    for k in keys_ms:
        if k in params and params[k] is not None:
            try:
                v = float(params[k])
                if v > 5.0:  # likely ms
                    return max(1e-3, v * 1e-3)
                if 0.001 <= v <= 2.0:  # plausible seconds
                    return v
                return max(1e-3, v * 1e-3)
            except Exception:
                pass
    return float(default_s)


def _compute_axes_from_cfg(params: dict, Nr_half: int, Nd_eff: int, Ntx_hint: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute range (m) and velocity (m/s) axes from TI cfg-style params."""
    c = 299_792_458.0
    rng_m = np.arange(Nr_half, dtype=np.float32)
    vel_mps = (np.arange(Nd_eff, dtype=np.float32) - (Nd_eff // 2)).astype(np.float32)

    try:
        slope_mhz_per_us = (params.get('slope_mhz_per_us') or params.get('freqSlopeConst') or
                            params.get('slopeMhzPerUs') or params.get('slope'))
        start_freq_ghz   = (params.get('start_freq_ghz') or params.get('startFreq') or
                            params.get('startFreqGHz') or params.get('start_freq'))
        ramp_end_time_us = (params.get('ramp_end_time_us') or params.get('rampEndTime') or
                            params.get('rampEndTime_us') or params.get('ramp_end_time'))
        idle_time_us     = (params.get('idle_time_us') or params.get('idleTime') or params.get('idle_time'))

        frame_ms         = (params.get('framePeriodicity_ms') or params.get('framePeriodicity') or
                            params.get('frame_periodicity_ms') or params.get('frame_periodicity'))
        chirp_start_idx  = params.get('chirpStartIdx')
        chirp_end_idx    = params.get('chirpEndIdx')
        num_loops        = params.get('numLoops')
        Ntx_cfg          = params.get('tx')

        slope_Hz_per_s = float(slope_mhz_per_us) * 1e12
        f0_GHz = float(start_freq_ghz)
        if f0_GHz < 10.0:
            f0_GHz *= 10.0
        lam = c / (f0_GHz * 1e9)

        Tr = float(ramp_end_time_us) * 1e-6
        Ti = float(idle_time_us) * 1e-6 if idle_time_us is not None else 0.0
        Ntx_eff = int(Ntx_hint) if int(Ntx_hint) > 0 else int(Ntx_cfg or 1)

        # Range
        B  = slope_Hz_per_s * Tr
        dR = c / (2.0 * B)
        rng_m = (dR * np.arange(Nr_half, dtype=np.float32)).astype(np.float32)

        # Doppler axis
        TD_chirp = Ntx_eff * (Ti + Tr)

        TD_frame = None
        if (frame_ms is not None) and (chirp_start_idx is not None) and (chirp_end_idx is not None) and (num_loops is not None):
            T_frame_s = float(frame_ms) * 1e-3
            chirps_per_loop = int(chirp_end_idx) - int(chirp_start_idx) + 1
            Nd_total = int(num_loops) * chirps_per_loop
            T_chirp_est = T_frame_s / max(Nd_total, 1)
            TD_frame = Ntx_eff * T_chirp_est

        def vmax_from_TD(TD): return lam / (4.0 * TD)

        TD = TD_chirp
        if TD_frame is not None:
            vA = vmax_from_TD(TD_chirp)
            vB = vmax_from_TD(TD_frame)
            if not (1.0 <= vA <= 15.0):
                TD = TD_frame
            else:
                TD = TD_chirp if abs(vA - 5.0) <= abs(vB - 5.0) else TD_frame

        fd = np.fft.fftshift(np.fft.fftfreq(Nd_eff, d=TD)).astype(np.float32)
        vel_mps = (fd * lam / 2.0).astype(np.float32)

    except Exception:
        pass

    return rng_m, vel_mps


def _apply_mti(x: np.ndarray, order: int) -> np.ndarray:
    """Simple MTI cancellers (2- and 3-pulse) implemented by differences along slow-time."""
    if order <= 0:
        return x
    y = np.zeros_like(x)
    if order == 1:
        y[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
    else:
        y[:, 2:, :, :] = x[:, 2:, :, :] - 2.0 * x[:, 1:-1, :, :] + x[:, :-2, :, :]
    return y


def _ensure_default_azimuth_geometry(Ntx: int, Nrx: int, V: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure AZIM_VCHAN_IDXS and AZIM_POS_WAVELENGTHS are set.
    Returns (ch_idx, pos_wavelengths) arrays.
    """
    global AZIM_VCHAN_IDXS, AZIM_POS_WAVELENGTHS

    if AZIM_VCHAN_IDXS is None:
        # Hardware-aware default: for 6843-style (4 RX, >=2 TX), use first 2 TX (azimuth MIMO), exclude 3rd TX (elevation).
        if (Nrx == 4) and (Ntx >= 2):
            AZIM_VCHAN_IDXS = tuple(range(0, 2 * Nrx))  # TX0 RX0..3, TX1 RX0..3
        else:
            # Fallback: use all virtual channels as a ULA (may be physically wrong).
            AZIM_VCHAN_IDXS = tuple(range(V))

    ch_idx = np.asarray(AZIM_VCHAN_IDXS, dtype=np.int32).reshape(-1)
    M = int(ch_idx.size)

    if AZIM_POS_WAVELENGTHS is None:
        # Default contiguous ULA positions (in wavelengths) with spacing BF_D_OVER_LAMBDA.
        AZIM_POS_WAVELENGTHS = (np.arange(M, dtype=np.float32) * float(BF_D_OVER_LAMBDA)).astype(np.float32)
    else:
        AZIM_POS_WAVELENGTHS = np.asarray(AZIM_POS_WAVELENGTHS, dtype=np.float32).reshape(-1)

    if AZIM_POS_WAVELENGTHS.size != M:
        raise ValueError(f"AZIM_POS_WAVELENGTHS length {AZIM_POS_WAVELENGTHS.size} != len(AZIM_VCHAN_IDXS)={M}")

    return ch_idx, AZIM_POS_WAVELENGTHS


def _get_beamformer(M: int, pos_wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (angles_rad, A) where A is the steering matrix (angles x M)."""
    ang_min = float(BF_ANGLE_MIN_DEG)
    ang_max = float(BF_ANGLE_MAX_DEG)
    ang_step = float(BF_ANGLE_STEP_DEG)

    pos = np.asarray(pos_wavelengths, dtype=np.float32).reshape(-1)
    if pos.size != M:
        raise ValueError(f"pos_wavelengths length {pos.size} != M={M}")

    pos_key = tuple(np.round(pos.astype(np.float64), 6).tolist()) if pos.size else tuple()
    key = (M, ang_min, ang_max, ang_step, pos_key)
    if key in _BF_CACHE:
        return _BF_CACHE[key]

    angles_deg = np.arange(ang_min, ang_max + 1e-9, ang_step, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg).astype(np.float32)

    # Steering: a_m(θ) = exp(+j 2π pos_m sinθ)  with pos in wavelengths.
    phase = 2.0 * np.pi * (np.sin(angles_rad)[:, None] * pos[None, :]).astype(np.float32)
    A = np.exp(1j * phase).astype(np.complex64)  # [A, M]

    _BF_CACHE[key] = (angles_rad, A)
    return angles_rad, A


def _apply_az_cal(az_deg_raw: float) -> float:
    if not np.isfinite(az_deg_raw):
        return np.nan
    az = float(AZ_CAL_SCALE) * float(az_deg_raw) + float(AZ_CAL_BIAS_DEG)
    # Clamp to beamformer sweep range for sanity (optional)
    az = max(float(BF_ANGLE_MIN_DEG), min(float(BF_ANGLE_MAX_DEG), az))
    return az


def _beamform_azimuth_multi(snapshots: np.ndarray, pos_wavelengths: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Bartlett beamforming using multiple snapshots X (S x M).
    Returns (az_deg, conf, peak_power, snapshot_power_sum).
      - conf = peak / median of beam power spectrum.
    """
    X = np.asarray(snapshots, dtype=np.complex64)
    if X.ndim != 2:
        X = X.reshape(1, -1)
    S, M = X.shape
    if M < 2 or S < 1:
        return np.nan, 0.0, 0.0, 0.0

    # Snapshot power (for absolute reject)
    snap_pwr = float(np.sum(np.abs(X) ** 2))

    angles_rad, A = _get_beamformer(M, pos_wavelengths)  # A: [Ang, M]
    Y = A @ X.T  # [Ang, S]
    p = np.sum(np.abs(Y) ** 2, axis=1).astype(np.float32)  # [Ang]

    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(p)) if p.size else 0.0
    med = float(np.median(p)) + EPS
    conf = float(peak / med) if med > 0 else 0.0
    i = int(np.argmax(p)) if p.size else 0
    az_deg = float(np.rad2deg(float(angles_rad[i]))) if p.size else np.nan
    az_deg = _apply_az_cal(az_deg)

    return az_deg, conf, peak, snap_pwr


def _nms_range_bins(rbins: np.ndarray, scores: np.ndarray, min_sep_bins: int) -> List[int]:
    """Simple 1D non-max suppression for range bins."""
    if rbins.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    kept: List[int] = []
    for idx in order:
        r = int(rbins[idx])
        if all(abs(r - kr) >= int(min_sep_bins) for kr in kept):
            kept.append(r)
    return kept


def postProc(adc_data: np.ndarray, ADC_PARAMS_l: dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pure processing:
    Returns:
      E_fr: (F,R) range energy per frame (linear)
      md_cols: (F,Nd_eff) μD columns (linear)
      det: dict with detection/beamforming outputs:
          - 'candidates': list (len F) of list of candidate dicts:
              {'r_bin','range_m','az_deg','az_conf','score','d_bin','range_energy','snap_pwr'}
          - 'best': dict of per-frame best candidate arrays:
              'range_m','az_deg','az_conf','r_bin'
    """
    adc_data = np.reshape(
        adc_data,
        (-1, ADC_PARAMS_l['chirps'], ADC_PARAMS_l['tx'], ADC_PARAMS_l['rx'],
         ADC_PARAMS_l['samples'] // 2, ADC_PARAMS_l['IQ'], 2)
    )
    adc_data = np.transpose(adc_data, (0, 1, 2, 3, 4, 6, 5))
    adc_data = np.reshape(
        adc_data,
        (-1, ADC_PARAMS_l['chirps'], ADC_PARAMS_l['tx'], ADC_PARAMS_l['rx'],
         ADC_PARAMS_l['samples'], ADC_PARAMS_l['IQ'])
    )
    adc_cplx = (1j * adc_data[..., 0] + adc_data[..., 1]).astype(np.complex64)
    F, Nd_total, Ntx, Nrx, Nr = adc_cplx.shape

    # 1) Range FFT with Hann
    win_key = ('wr', Nr)
    if win_key not in _WIN_CACHE:
        _WIN_CACHE[win_key] = np.hanning(Nr).astype(np.float32)[None, None, None, None, :]
    wr = _WIN_CACHE[win_key]
    Xr = np.fft.fft(adc_cplx * wr, n=Nr, axis=-1)

    # 2) Keep higher-energy beat half
    Nfft = Xr.shape[-1]
    N2 = Nfft // 2
    pos = Xr[..., :N2]
    neg = Xr[..., N2:]
    if (np.abs(neg) ** 2).sum() > (np.abs(pos) ** 2).sum():
        Xr = neg[..., ::-1]
    else:
        Xr = pos
    Nr_half = Xr.shape[-1]

    # 3) Doppler per TX (TDM)
    Nd_eff = Nd_total // max(Ntx, 1)
    dwin_key = ('wd', Nd_eff)
    if dwin_key not in _WIN_CACHE:
        _WIN_CACHE[dwin_key] = np.hanning(Nd_eff).astype(np.float32)
    wd = _WIN_CACHE[dwin_key]

    Xrd_virtual_list = []
    for m in range(Ntx):
        Xr_sel = Xr[:, m:Nd_total:Ntx, m, :, :]  # [F, Nd_eff, Nrx, Nr_half]

        if MTI_CANCELLER > 0:
            Xr_sel = _apply_mti(Xr_sel, MTI_CANCELLER)
        else:
            Xr_sel = Xr_sel - Xr_sel.mean(axis=1, keepdims=True)

        Xr_sel = Xr_sel * wd[None, :, None, None]
        Xd = np.fft.fft(Xr_sel, n=Nd_eff, axis=1)
        Xd = np.fft.fftshift(Xd, axes=1)

        Xd = np.transpose(Xd, (0, 3, 1, 2))  # [F, R, Nd, Nrx]
        Xrd_virtual_list.append(Xd)

    Xrd_virtual = np.concatenate(Xrd_virtual_list, axis=3)  # [F, R, Nd, V]
    Xrd_virtual = Xrd_virtual - Xrd_virtual.mean(axis=3, keepdims=True)

    Fnum, Rnum, Nd, V = Xrd_virtual.shape

    # Power cube (sum over virtual array)
    P_frd = (np.abs(Xrd_virtual) ** 2).sum(axis=3).astype(np.float32)  # [F, R, Nd]
    P = P_frd

    # Noise subtraction per Doppler using low quantile across range
    if 0.0 < NOISE_SUBTRACT_Q < 1.0:
        nf_all = np.quantile(P, NOISE_SUBTRACT_Q, axis=1).astype(np.float32)  # [F, Nd]
        P = np.maximum(P - nf_all[:, None, :], 0.0)

    # Doppler mask to suppress DC/very-low Doppler
    center = Nd // 2
    zero_w = max(3, Nd // 64)
    edge_w = max(center - 1, int(0.45 * Nd))
    d_idx = np.arange(Nd)
    dop_mask = (np.abs(d_idx - center) >= zero_w) & (np.abs(d_idx - center) <= edge_w)

    # Range energy using doppler mask
    E_fr = P[:, :, dop_mask].sum(axis=2)

    # Smooth E_fr across range (5-bin MA)
    if E_fr.size:
        kern = np.ones(5, dtype=np.float32) / 5.0
        E_fr = np.apply_along_axis(lambda x: np.convolve(x, kern, mode='same'), 1, E_fr)

    # Range center (for μD collapse) via EWMA of argmax (kept for μD stability)
    if E_fr.shape[0] > 0:
        r_center = np.zeros(Fnum, dtype=np.int32)
        r_center[0] = int(np.argmax(E_fr[0]))
        alpha = 0.8
        for fidx in range(1, Fnum):
            r_raw = int(np.argmax(E_fr[fidx]))
            r_center[fidx] = int(round(alpha * r_center[fidx - 1] + (1.0 - alpha) * r_raw))
    else:
        r_center = np.zeros(Fnum, dtype=np.int32)

    # Axes
    ax_key = (Nr_half, Nd_eff, Ntx,
              ADC_PARAMS_l.get('start_freq_ghz') or ADC_PARAMS_l.get('startFreq'),
              ADC_PARAMS_l.get('slope_mhz_per_us') or ADC_PARAMS_l.get('freqSlopeConst'),
              ADC_PARAMS_l.get('ramp_end_time_us') or ADC_PARAMS_l.get('rampEndTime'),
              ADC_PARAMS_l.get('idle_time_us') or ADC_PARAMS_l.get('idleTime'))
    if ax_key in _AXES_CACHE:
        rng_m, vel_mps = _AXES_CACHE[ax_key]
    else:
        rng_m, vel_mps = _compute_axes_from_cfg({**ADC_PARAMS_l}, Nr_half, Nd_eff, Ntx)
        _AXES_CACHE[ax_key] = (rng_m, vel_mps)

    # Gaussian range weights for μD collapse
    dR = float(abs(rng_m[1] - rng_m[0])) if (rng_m is not None and len(rng_m) > 1) else 0.1
    sigma_bins = max(2.0, float(RANGE_SIGMA_M) / max(dR, 1e-6))
    r_grid = np.arange(Rnum, dtype=np.float32)
    W = np.exp(-0.5 * ((r_grid[None, :] - r_center[:, None]) / sigma_bins) ** 2).astype(np.float32)
    W /= (W.sum(axis=1, keepdims=True) + EPS)

    # Collapse range with weights -> μD columns
    md_cols = (P * W[:, :, None]).sum(axis=1).astype(np.float32)  # [F, Nd]

    # Optional Doppler smoothing
    k = int(max(1, int(DOPPLER_SMOOTH_K)))
    if k > 1:
        if (k % 2) == 0:
            k += 1
        ker = np.ones(k, dtype=np.float32) / float(k)
        md_cols = np.apply_along_axis(lambda r: np.convolve(r, ker, mode='same'), 1, md_cols)

    # Optional frame EWMA inside batch
    a = float(FRAME_EWMA)
    if 0.0 < a < 1.0 and md_cols.shape[0] > 1:
        for fidx in range(1, md_cols.shape[0]):
            md_cols[fidx] = a * md_cols[fidx - 1] + (1.0 - a) * md_cols[fidx]

    # ---------- Candidate detection over multiple ranges ----------
    candidates: List[List[Dict[str, Any]]] = [[] for _ in range(Fnum)]
    best_range_m = np.full((Fnum,), np.nan, dtype=np.float32)
    best_az_deg = np.full((Fnum,), np.nan, dtype=np.float32)
    best_az_conf = np.zeros((Fnum,), dtype=np.float32)
    best_r_bin = np.full((Fnum,), -1, dtype=np.int32)

    if BEAMFORM_ENABLE and DETECT_ENABLE and (Fnum > 0):
        ch_idx, pos_wl = _ensure_default_azimuth_geometry(Ntx=Ntx, Nrx=Nrx, V=V)
        M = int(ch_idx.size)

        # Range mask for candidate generation
        rng_mask = np.ones((Rnum,), dtype=bool)
        if rng_m is not None and len(rng_m) >= Rnum:
            rng_mask &= (rng_m[:Rnum] >= float(DETECT_RANGE_MIN_M)) & (rng_m[:Rnum] <= float(DETECT_RANGE_MAX_M))

        for fidx in range(Fnum):
            E = E_fr[fidx].astype(np.float32)
            if E.size == 0:
                continue
            maxE = float(np.max(E))
            if not np.isfinite(maxE) or maxE <= 0:
                continue

            thr = max(float(DETECT_E_ABS_MIN), float(DETECT_E_FRAC_OF_MAX) * maxE)
            idxs = np.where((E >= thr) & rng_mask)[0]
            if idxs.size == 0:
                continue

            # Sort by energy desc and limit
            idxs = idxs[np.argsort(E[idxs])[::-1]]
            idxs = idxs[:int(max(1, DETECT_TOPK_RANGES))]

            # Optional NMS over range bins
            if int(DETECT_MIN_RANGE_BIN_SEP) > 0 and idxs.size > 1:
                kept = _nms_range_bins(idxs, E[idxs], int(DETECT_MIN_RANGE_BIN_SEP))
                idxs = np.asarray(kept, dtype=np.int32)

            for rb in idxs:
                rb = int(rb)
                # Choose peak Doppler bin at this range (excluding DC)
                p_d = P[fidx, rb, :]
                if np.any(dop_mask):
                    p_sel = p_d[dop_mask]
                    if p_sel.size == 0:
                        continue
                    d_rel = int(np.argmax(p_sel))
                    d_bin = int(np.where(dop_mask)[0][d_rel])
                else:
                    d_bin = int(np.argmax(p_d))

                # Build snapshot set around peak Doppler
                d0 = max(0, d_bin - int(DETECT_DOP_NEIGH_BINS))
                d1 = min(Nd, d_bin + int(DETECT_DOP_NEIGH_BINS) + 1)
                d_bins = [d for d in range(d0, d1) if (dop_mask[d] if dop_mask.size else True)]
                if not d_bins:
                    d_bins = [d_bin]

                X = Xrd_virtual[fidx, rb, d_bins, :][:, ch_idx]  # [S, M]
                snap_pwr = float(np.sum(np.abs(X) ** 2))
                if (not np.isfinite(snap_pwr)) or (snap_pwr < float(BF_MIN_SNAPSHOT_PWR)):
                    continue

                az_deg, conf, _peak, _sp = _beamform_azimuth_multi(X, pos_wl)
                if (not np.isfinite(az_deg)) or (conf < float(BF_MIN_CONF)):
                    continue

                cand = {
                    "r_bin": int(rb),
                    "range_m": float(rng_m[rb]) if (rng_m is not None and rb < len(rng_m)) else float(rb),
                    "az_deg": float(az_deg),
                    "az_conf": float(conf),
                    "range_energy": float(E[rb]),
                    "d_bin": int(d_bin),
                    "snap_pwr": float(snap_pwr),
                }
                cand["score"] = float(cand["range_energy"] * cand["az_conf"])
                candidates[fidx].append(cand)

            # Choose best per frame (for convenience)
            if candidates[fidx]:
                best = max(candidates[fidx], key=lambda c: float(c.get("score", 0.0)))
                best_range_m[fidx] = float(best["range_m"])
                best_az_deg[fidx] = float(best["az_deg"])
                best_az_conf[fidx] = float(best["az_conf"])
                best_r_bin[fidx] = int(best["r_bin"])

    det = {
        "candidates": candidates,
        "best": {
            "range_m": best_range_m.astype(np.float32),
            "az_deg": best_az_deg.astype(np.float32),
            "az_conf": best_az_conf.astype(np.float32),
            "r_bin": best_r_bin.astype(np.int32),
        }
    }
    return E_fr.astype(np.float32), md_cols.astype(np.float32), det


def estimate_steps(md_history: np.ndarray,
                   timestamps: np.ndarray,
                   fs_hz: float,
                   high_bin_start: int,
                   high_bin_end: int,
                   low_cut_hz: float = 0.8,
                   high_cut_hz: float = 3.0,
                   min_step_time_s: float = 0.30,
                   prom_mult: float = 0.50):
    """Minimal μD step-cycle detector (SciPy if available).
    Returns: cadence_spm, steps_ts, envelope, filtered, peaks_idx
    """
    md = np.asarray(md_history, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if md.ndim != 2 or t.ndim != 1 or md.shape[0] != t.shape[0] or md.shape[0] < 3:
        return np.nan, np.empty((0,), np.float64), None, None, np.array([], dtype=int)

    Nd = md.shape[1]
    c = Nd // 2

    hi0 = int(np.clip(high_bin_start, 0, Nd))
    hi1 = int(np.clip(high_bin_end,   0, Nd))
    if hi1 <= hi0:
        return np.nan, np.empty((0,), np.float64), None, None, np.array([], dtype=int)

    lo0 = int(np.clip(2 * c - hi1, 0, Nd))
    lo1 = int(np.clip(2 * c - hi0, 0, Nd))
    if lo1 <= lo0:
        return np.nan, np.empty((0,), np.float64), None, None, np.array([], dtype=int)

    env = md[:, lo0:lo1].sum(axis=1) + md[:, hi0:hi1].sum(axis=1)
    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env = np.log1p(np.maximum(env, 0.0))
    env = env - np.median(env)

    filt = env
    peaks = np.array([], dtype=int)

    if SCIPY_AVAILABLE and fs_hz and np.isfinite(fs_hz) and fs_hz > 0:
        nyq = 0.5 * float(fs_hz)
        lo = max(0.05, float(low_cut_hz))
        hi = min(float(high_cut_hz), 0.90 * nyq)

        if hi > lo and nyq > 0:
            try:
                b, a = butter(2, [lo / nyq, hi / nyq], btype="bandpass")
                pad_need = 3 * (max(len(a), len(b)) - 1)
                if env.size > (pad_need + 1):
                    filt = filtfilt(b, a, env)
                else:
                    filt = env
            except Exception:
                filt = env

        dist = max(1, int(round(float(min_step_time_s) * float(fs_hz))))
        prom = float(prom_mult) * float(np.std(filt)) if np.std(filt) > 0 else None
        try:
            peaks, _ = find_peaks(filt, distance=dist, prominence=prom)
        except Exception:
            peaks = np.array([], dtype=int)

    steps_ts = t[peaks] if peaks.size else np.empty((0,), np.float64)
    cadence_spm = np.nan
    if steps_ts.size >= 2:
        dt = np.diff(steps_ts)
        dt = dt[np.isfinite(dt)]
        if dt.size:
            cadence_spm = 60.0 / float(np.mean(dt))
    return float(cadence_spm) if np.isfinite(cadence_spm) else np.nan, steps_ts, env, filt, peaks


class MuDStepCycleDetector:
    """Rolling-window μD cadence/step-time estimator."""
    def __init__(self, Nd: int,
                 window_s: float = GAIT_DETECT_WINDOW_S,
                 band_hz: tuple = GAIT_DETECT_BAND_HZ,
                 min_step_s: float = GAIT_DETECT_MIN_STEP_S,
                 max_step_s: float = GAIT_DETECT_MAX_STEP_S,
                 bin_min_frac: float = GAIT_DETECT_BIN_MIN_FRAC,
                 bin_max_frac: float = GAIT_DETECT_BIN_MAX_FRAC,
                 compute_every_s: float = GAIT_DETECT_COMPUTE_EVERY_S,
                 prom_mult: float = GAIT_DETECT_PROM_MULT):
        self.Nd = int(Nd)
        self.window_s = float(window_s)
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))
        self.min_step_s = float(min_step_s)
        self.max_step_s = float(max_step_s)
        self.compute_every_s = float(compute_every_s)
        self.prom_mult = float(prom_mult)

        c = self.Nd // 2
        min_off = max(1, int(round(float(bin_min_frac) * self.Nd)))
        max_off = max(min_off + 1, int(round(float(bin_max_frac) * self.Nd)))
        self.hi_start = int(np.clip(c + min_off, 0, self.Nd))
        self.hi_end   = int(np.clip(c + max_off, 0, self.Nd))
        if self.hi_end <= self.hi_start:
            self.hi_start = int(np.clip(c + 1, 0, self.Nd - 2))
            self.hi_end   = int(np.clip(self.hi_start + 2, 0, self.Nd))

        self._cols: Deque[np.ndarray] = deque()
        self._ts: Deque[float] = deque()
        self._last_compute_t: Optional[float] = None
        self._last_emitted_step_t: Optional[float] = None

    def update(self, t_s: float, col_lin: np.ndarray) -> Optional[Dict[str, Any]]:
        t_s = float(t_s)
        col = np.asarray(col_lin, dtype=np.float32)
        if col.ndim != 1 or col.size != self.Nd:
            return None

        self._ts.append(t_s)
        self._cols.append(col)

        while self._ts and (self._ts[-1] - self._ts[0] > self.window_s):
            self._ts.popleft()
            self._cols.popleft()

        if self._last_compute_t is not None and (t_s - self._last_compute_t) < self.compute_every_s:
            return None
        self._last_compute_t = t_s
        return self.compute()

    def compute(self) -> Optional[Dict[str, Any]]:
        if len(self._ts) < 20:
            return None

        t = np.asarray(self._ts, dtype=np.float64)
        md = np.stack(list(self._cols), axis=0).astype(np.float64)

        dt = np.diff(t)
        fs = np.nan
        if dt.size:
            dt_med = float(np.median(dt))
            if dt_med > 0:
                fs = 1.0 / dt_med

        _cad_raw, steps_ts, _env, _filt, _peaks = estimate_steps(
            md_history=md,
            timestamps=t,
            fs_hz=float(fs) if np.isfinite(fs) else 0.0,
            high_bin_start=self.hi_start,
            high_bin_end=self.hi_end,
            low_cut_hz=self.band_hz[0],
            high_cut_hz=self.band_hz[1],
            min_step_time_s=self.min_step_s,
            prom_mult=self.prom_mult
        )

        intervals = np.diff(steps_ts) if steps_ts.size >= 2 else np.array([], dtype=np.float64)
        valid = (intervals >= self.min_step_s) & (intervals <= self.max_step_s)
        intervals_v = intervals[valid]

        cadence = np.nan
        mean_dt = np.nan
        last_dt = np.nan
        if intervals_v.size:
            mean_dt = float(np.mean(intervals_v))
            last_dt = float(intervals_v[-1])
            cadence = 60.0 / mean_dt if mean_dt > 0 else np.nan

        if self._last_emitted_step_t is None:
            new_mask = np.ones_like(steps_ts, dtype=bool)
        else:
            new_mask = steps_ts > (self._last_emitted_step_t + 1e-6)

        new_steps = steps_ts[new_mask]
        if steps_ts.size:
            intervals_full = np.concatenate([[np.nan], np.diff(steps_ts)])
            new_intervals = intervals_full[new_mask]
        else:
            new_intervals = np.array([], dtype=np.float64)

        if new_steps.size:
            self._last_emitted_step_t = float(new_steps[-1])

        return {
            "fs_hz": float(fs) if np.isfinite(fs) else np.nan,
            "cadence_spm": float(cadence) if np.isfinite(cadence) else np.nan,
            "step_count_window": int(steps_ts.size),
            "step_interval_mean_s": mean_dt,
            "step_interval_last_s": last_dt,
            "hi_bin_start": int(self.hi_start),
            "hi_bin_end": int(self.hi_end),
            "window_len_s": float(t[-1] - t[0]) if t.size else 0.0,
            "new_steps_ts_s": new_steps,
            "new_steps_interval_s": new_intervals,
        }


class StepAsymmetryEstimator:
    """Alternating-step timing asymmetry from step intervals (odd/even interval means)."""
    def __init__(self, max_intervals: int = ASYM_WINDOW_STEPS, min_intervals: int = ASYM_MIN_INTERVALS):
        self.max_intervals = int(max(4, max_intervals))
        self.min_intervals = int(max(6, min_intervals))
        self._dt: Deque[float] = deque(maxlen=self.max_intervals)

    def update(self, step_intervals_s: np.ndarray) -> Dict[str, Any]:
        if step_intervals_s is None:
            return {"asym_ai": np.nan, "mu_even_s": np.nan, "mu_odd_s": np.nan, "n": int(len(self._dt))}

        arr = np.asarray(step_intervals_s, dtype=np.float64).reshape(-1)
        for v in arr:
            if np.isfinite(v) and v > 0:
                self._dt.append(float(v))

        n = len(self._dt)
        if n < self.min_intervals:
            return {"asym_ai": np.nan, "mu_even_s": np.nan, "mu_odd_s": np.nan, "n": int(n)}

        dt = np.array(self._dt, dtype=np.float64)
        even = dt[0::2]
        odd = dt[1::2]
        if even.size < 3 or odd.size < 3:
            return {"asym_ai": np.nan, "mu_even_s": np.nan, "mu_odd_s": np.nan, "n": int(n)}

        mu_even = float(np.mean(even))
        mu_odd = float(np.mean(odd))
        denom = (mu_even + mu_odd) / 2.0
        ai = float(abs(mu_even - mu_odd) / denom) if denom > 0 else np.nan
        return {"asym_ai": ai, "mu_even_s": mu_even, "mu_odd_s": mu_odd, "n": int(n)}


class RadialSpeedEstimatorQC:
    """Doppler-only radial speed estimator from a μD column (linear power). Debug/QC only."""
    def __init__(self, vel_mps: np.ndarray):
        self.vel = np.asarray(vel_mps, dtype=np.float32)
        self.Nd = int(self.vel.size)
        self._ema: Optional[float] = None
        self._ema_conf: Optional[float] = None

        v = self.vel
        self._mask_pos = (v >= float(RADIAL_QC_V_MIN_MPS)) & (v <= float(RADIAL_QC_V_MAX_MPS))
        self._mask_neg = (v <= -float(RADIAL_QC_V_MIN_MPS)) & (v >= -float(RADIAL_QC_V_MAX_MPS))

    def update(self, col_lin: np.ndarray) -> Dict[str, Any]:
        p = np.asarray(col_lin, dtype=np.float32).reshape(-1)
        if p.size != self.Nd:
            return {"v_rad_mps": np.nan, "conf": 0.0, "dir": 0}

        e_pos = float(np.sum(p[self._mask_pos])) if np.any(self._mask_pos) else 0.0
        e_neg = float(np.sum(p[self._mask_neg])) if np.any(self._mask_neg) else 0.0
        direction = +1 if e_pos >= e_neg else -1
        mask = self._mask_pos if direction > 0 else self._mask_neg
        if not np.any(mask):
            return {"v_rad_mps": np.nan, "conf": 0.0, "dir": direction}

        idxs = np.where(mask)[0]
        band = p[idxs]
        pk_rel = int(np.argmax(band))
        pk_idx = int(idxs[pk_rel])
        pk_val = float(band[pk_rel])
        med = float(np.median(band)) + EPS
        conf = float(pk_val / med) if med > 0 else 0.0

        v = np.nan
        if np.isfinite(conf) and conf >= float(RADIAL_QC_MIN_CONF) and pk_val > 0:
            nb = int(max(0, RADIAL_QC_PEAK_NEIGHBOR_BINS))
            lo = max(0, pk_idx - nb)
            hi = min(self.Nd, pk_idx + nb + 1)
            w = p[lo:hi].astype(np.float64)
            vv = self.vel[lo:hi].astype(np.float64)
            sw = float(np.sum(w))
            v = float(np.sum(w * vv) / sw) if sw > 0 else float(self.vel[pk_idx])

        if np.isfinite(v):
            if self._ema is None:
                self._ema = float(v)
                self._ema_conf = float(conf)
            else:
                a = float(RADIAL_QC_EMA_ALPHA)
                self._ema = a * float(self._ema) + (1.0 - a) * float(v)
                self._ema_conf = a * float(self._ema_conf) + (1.0 - a) * float(conf)
            return {"v_rad_mps": float(self._ema), "conf": float(self._ema_conf), "dir": int(direction), "pk_bin": int(pk_idx)}
        return {"v_rad_mps": np.nan, "conf": float(conf) if np.isfinite(conf) else 0.0, "dir": int(direction), "pk_bin": int(pk_idx)}


class SingleTargetTracker2D:
    """Constant-velocity KF with multi-candidate association (single target)."""
    def __init__(self, speed_ema_alpha: float = TRACK_SPEED_EMA_ALPHA):
        self.x: Optional[np.ndarray] = None  # state [x,y,vx,vy]
        self.P: Optional[np.ndarray] = None
        self.last_t: Optional[float] = None
        self.last_meas_t: Optional[float] = None
        self._speed_ema: Optional[float] = None
        self._speed_alpha = float(speed_ema_alpha)

    def reset(self):
        self.x = None
        self.P = None
        self.last_t = None
        self.last_meas_t = None
        self._speed_ema = None

    def _predict(self, dt: float):
        dt = float(max(1e-3, dt))
        Fm = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]], dtype=np.float64)
        sa2 = float(TRACK_SIGMA_A) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = sa2 * np.array([[dt4/4, 0,     dt3/2, 0],
                            [0,     dt4/4, 0,     dt3/2],
                            [dt3/2, 0,     dt2,   0],
                            [0,     dt3/2, 0,     dt2]], dtype=np.float64)
        self.x = Fm @ self.x
        self.P = Fm @ self.P @ Fm.T + Q

    def _maha2(self, z: np.ndarray, R: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return (maha2, y, S_inv) for current predicted state."""
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        z = np.asarray(z, dtype=np.float64).reshape(2)
        R = np.asarray(R, dtype=np.float64).reshape(2, 2)
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf, y, np.zeros((2, 2), dtype=np.float64)
        maha2 = float(y.T @ S_inv @ y)
        return maha2, y, S_inv

    def _update(self, z: np.ndarray, R: np.ndarray, S_inv: np.ndarray, y: np.ndarray) -> None:
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float64)
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

    def step_candidates(self, t: float, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        candidates: list of dicts each with:
          - 'z' : np.array([x,y]) (float64)
          - 'R' : np.array([[..],[..]]) (float64)
          - optional metadata fields: 'score','range_m','az_deg','az_conf','r_bin'
        """
        t = float(t)
        if self.last_t is None:
            self.last_t = t

        dt = float(max(1e-3, t - self.last_t))
        self.last_t = t

        # Initialize track if needed
        if self.x is None or self.P is None:
            if not candidates:
                return {"valid": False, "n_cands": 0}
            # Init from best candidate by score (fallback to first)
            best = max(candidates, key=lambda c: float(c.get("score", 0.0)))
            z0 = np.asarray(best["z"], dtype=np.float64).reshape(2)
            self.x = np.array([z0[0], z0[1], 0.0, 0.0], dtype=np.float64)
            self.P = np.diag([TRACK_INIT_POS_STD**2, TRACK_INIT_POS_STD**2,
                              TRACK_INIT_VEL_STD**2, TRACK_INIT_VEL_STD**2]).astype(np.float64)
            self.last_meas_t = t
            out = self._state_dict(valid=True, maha2=0.0, meas_conf=1.0, meas=best, n_cands=len(candidates))
            self._update_speed_ema(out)
            return out

        # Predict
        self._predict(dt)

        # No candidates: propagate or drop if stale
        if not candidates:
            if self.last_meas_t is not None and (t - self.last_meas_t) > float(TRACK_MAX_MISS_S):
                self.reset()
                return {"valid": False, "n_cands": 0}
            out = self._state_dict(valid=True, maha2=np.nan, meas_conf=0.0, meas=None, n_cands=0)
            self._update_speed_ema(out)
            return out

        # Evaluate candidates by Mahalanobis distance; choose best
        best_cand = None
        best_maha2 = np.inf
        best_y = None
        best_S_inv = None

        for c in candidates:
            z = c.get("z", None)
            R = c.get("R", None)
            if z is None or R is None:
                continue
            maha2, y, S_inv = self._maha2(z, R)
            if not np.isfinite(maha2):
                continue
            if maha2 < best_maha2:
                best_maha2 = maha2
                best_cand = c
                best_y = y
                best_S_inv = S_inv
            elif maha2 == best_maha2 and best_cand is not None:
                # Tie-break by score
                if float(c.get("score", 0.0)) > float(best_cand.get("score", 0.0)):
                    best_cand = c
                    best_y = y
                    best_S_inv = S_inv

        if (best_cand is None) or (not np.isfinite(best_maha2)) or (best_maha2 > float(TRACK_MAHA_GATE2)):
            # Gate failed: keep predicted; drop if stale
            if self.last_meas_t is not None and (t - self.last_meas_t) > float(TRACK_MAX_MISS_S):
                self.reset()
                return {"valid": False, "n_cands": len(candidates)}
            out = self._state_dict(valid=True, maha2=float(best_maha2), meas_conf=0.0, meas=None, n_cands=len(candidates))
            self._update_speed_ema(out)
            return out

        # Update with best candidate
        self._update(best_cand["z"], best_cand["R"], best_S_inv, best_y)
        self.last_meas_t = t
        meas_conf = float(1.0 / (1.0 + best_maha2))
        out = self._state_dict(valid=True, maha2=float(best_maha2), meas_conf=meas_conf, meas=best_cand, n_cands=len(candidates))
        self._update_speed_ema(out)
        return out

    def _update_speed_ema(self, out: Dict[str, Any]) -> None:
        if not out.get("valid", False):
            return
        spd = out.get("speed_mps", np.nan)
        if not np.isfinite(spd):
            return
        if self._speed_ema is None:
            self._speed_ema = float(spd)
        else:
            a = float(self._speed_alpha)
            self._speed_ema = a * float(self._speed_ema) + (1.0 - a) * float(spd)

    def _state_dict(self, valid: bool, maha2: float, meas_conf: float,
                    meas: Optional[Dict[str, Any]], n_cands: int) -> Dict[str, Any]:
        if self.x is None:
            return {"valid": False, "n_cands": int(n_cands)}
        x, y, vx, vy = [float(v) for v in self.x]
        spd = float(np.hypot(vx, vy))
        r = float(np.hypot(x, y))
        v_rad = float((x * vx + y * vy) / r) if r > 1e-6 else np.nan

        out = {
            "valid": bool(valid),
            "x_m": x, "y_m": y,
            "vx_mps": vx, "vy_mps": vy,
            "speed_mps": spd,
            "speed_ema_mps": float(self._speed_ema) if (self._speed_ema is not None) else spd,
            "v_rad_pred_mps": v_rad,
            "maha2": float(maha2) if np.isfinite(maha2) else np.nan,
            "meas_conf": float(meas_conf),
            "n_cands": int(n_cands),
        }
        if meas is not None:
            out["meas"] = {
                "range_m": float(meas.get("range_m", np.nan)) if np.isfinite(meas.get("range_m", np.nan)) else None,
                "az_deg": float(meas.get("az_deg", np.nan)) if np.isfinite(meas.get("az_deg", np.nan)) else None,
                "az_conf": float(meas.get("az_conf", 0.0)),
                "r_bin": int(meas.get("r_bin", -1)),
                "score": float(meas.get("score", 0.0)),
            }
        else:
            out["meas"] = None
        return out


def _cluster_candidates_xy(cands: List[Dict[str, Any]], dist_m: float) -> List[Dict[str, Any]]:
    """
    Simple greedy clustering in XY:
      - sort by score desc
      - keep a new candidate if it's not within dist_m of any kept candidate
    This acts like NMS in XY and reduces near-duplicates.
    """
    if not cands:
        return []
    dist2 = float(dist_m) ** 2
    order = sorted(cands, key=lambda c: float(c.get("score", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    for c in order:
        z = np.asarray(c.get("z", [np.nan, np.nan]), dtype=np.float64).reshape(2)
        if not np.isfinite(z).all():
            continue
        ok = True
        for k in kept:
            zk = np.asarray(k.get("z", [np.nan, np.nan]), dtype=np.float64).reshape(2)
            if np.isfinite(zk).all():
                d2 = float(np.sum((z - zk) ** 2))
                if d2 < dist2:
                    ok = False
                    break
        if ok:
            kept.append(c)
    return kept


class MDViewer(QtWidgets.QMainWindow if QT_AVAILABLE else object):
    """Fast rolling spectrogram viewer (pyqtgraph). Used by combined_viewer_app.py."""
    def __init__(self, Nd_eff: int, hist: int = 512):
        if not QT_AVAILABLE:
            raise RuntimeError("MDViewer requires PySide6 + pyqtgraph.")
        super().__init__()
        pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder='row-major')
        self.setWindowTitle("Micro-Doppler (live)")
        self.Nd = int(Nd_eff)
        self.hist = int(hist)

        self.img_arr = np.zeros((self.Nd, self.hist), dtype=np.float32)

        w = pg.GraphicsLayoutWidget()
        self.setCentralWidget(w)
        p = w.addPlot()
        p.setLabel('bottom', 'Frame')
        p.setLabel('left', 'Doppler bin')

        self.image = pg.ImageItem(self.img_arr)
        self.image.setAutoDownsample(True)
        lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        self.image.setLookupTable(lut)
        p.addItem(self.image)

        self.image.setRect(QtCore.QRectF(0, 0, self.hist, self.Nd))

        vb = p.getViewBox()
        vb.setAspectLocked(False)
        vb.invertY(True)
        vb.disableAutoRange()
        vb.setRange(self.image.boundingRect(), padding=0.0)
        p.setLimits(xMin=0, xMax=self.hist, yMin=0, yMax=self.Nd)

        self._hud = pg.TextItem(color=(200, 200, 200))
        self._hud.setPos(2, max(1, self.Nd - 6))
        p.addItem(self._hud)
        self._fps_ema = None
        self._last_data_time = None

        self._refresh()

    def append_col(self, col_db: np.ndarray):
        col_db = np.asarray(col_db, dtype=np.float32).reshape(-1)
        if col_db.size != self.Nd:
            return
        self.img_arr = np.roll(self.img_arr, -1, axis=1)
        self.img_arr[:, -1] = col_db

        now = time.time()
        if self._last_data_time is None:
            self._last_data_time = now
        dt = now - self._last_data_time
        self._last_data_time = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)
        self._hud.setText(f"fps≈{self._fps_ema:.1f}")
        self._refresh()

    def _refresh(self):
        self.image.setImage(self.img_arr, autoLevels=True, autoRange=False)
        vb = self.image.getViewBox()
        if vb is not None:
            vb.setRange(self.image.boundingRect(), padding=0.0)


def _open_csv(path: str, header: list):
    """Open CSV for append; create header if file is new/empty. Returns (fp, writer) or (None, None)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        fp = open(path, "a", newline="")
        wr = csv.writer(fp)
        if write_header:
            wr.writerow(header)
        return fp, wr
    except Exception as e:
        print(f"[csv] WARNING: could not open {path}: {e}")
        return None, None


def _measurement_from_range_az(r_m: float, az_deg: float, az_conf: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (range, az) to (x,y) measurement and covariance R_xy.
    Uses confidence-adaptive angle std.
    """
    th = float(np.deg2rad(float(az_deg)))
    x = float(r_m) * float(np.cos(th))
    y = float(r_m) * float(np.sin(th))

    sr = float(TRACK_MEAS_RANGE_STD_M)

    # Confidence-adaptive sigma_theta (deg): higher conf => smaller sigma
    base = float(TRACK_MEAS_ANGLE_STD_DEG)
    conf_ratio = max(1.0, float(az_conf) / max(float(BF_MIN_CONF), 1e-6))
    sig_deg = base / np.sqrt(conf_ratio)
    sig_deg = float(np.clip(sig_deg, float(TRACK_MEAS_ANGLE_STD_MIN_DEG), float(TRACK_MEAS_ANGLE_STD_MAX_DEG)))
    sth = float(np.deg2rad(sig_deg))

    cth = float(np.cos(th))
    sth_sin = float(np.sin(th))

    # Jacobian-based covariance for x=r cosθ, y=r sinθ
    var_x = (cth * sr) ** 2 + (-float(r_m) * sth_sin * sth) ** 2
    var_y = (sth_sin * sr) ** 2 + (float(r_m) * cth * sth) ** 2
    cov_xy = (cth * sth_sin * (sr ** 2 - (float(r_m) ** 2) * (sth ** 2)))

    R = np.array([[var_x, cov_xy],
                  [cov_xy, var_y]], dtype=np.float64) + np.eye(2, dtype=np.float64) * 1e-6
    z = np.array([x, y], dtype=np.float64)
    return z, R


class GaitMetricsLogger:
    """Append-only tidy-row metrics logger with provenance and crash-safe finalization.

    Writes:
      - <filename>.tmp while running (append-only)
      - <filename> on clean close (atomic rename where possible)
      - run_meta.json (provenance / configuration) once at start

    Timestamps:
      - t_rel_s: monotonic seconds since start (smooth plots)
      - t_epoch_s: Unix epoch seconds (cross-device sync)
    """
    def __init__(self,
                 session_dir: str,
                 filename: str = "metrics.csv",
                 fieldnames: Optional[List[str]] = None,
                 meta: Optional[Dict[str, Any]] = None,
                 flush_every_n: int = 25,
                 flush_every_s: float = 1.0):
        self.session_dir = str(session_dir)
        os.makedirs(self.session_dir, exist_ok=True)

        self.filename = str(filename)
        self.final_path = os.path.join(self.session_dir, self.filename)
        self.tmp_path = self.final_path + ".tmp"

        self.fieldnames = list(fieldnames) if fieldnames else [
            "t_rel_s", "t_epoch_s", "frame_idx",
            "subject_id",
            "walking", "gait_speed_mean_mps", "cadence_spm", "asymmetry_ai",
            "track_valid", "x_m", "y_m", "vx_mps", "vy_mps",
            "torso_range_m", "az_deg", "az_conf", "r_bin",
            "quality", "flags"
        ]

        self.flush_every_n = int(max(1, flush_every_n))
        self.flush_every_s = float(max(0.0, flush_every_s))
        self._rows_since_flush = 0
        self._last_flush_mono = time.monotonic()

        self._t0_mono = time.monotonic()
        self._t0_epoch = time.time()

        self._fp = None
        self._wr: Optional[csv.DictWriter] = None

        # Ensure tmp exists and has header
        self._open_tmp()

        # Write provenance once
        if meta is not None:
            try:
                self.write_meta(meta)
            except Exception as e:
                print(f"[metrics] WARNING: failed to write run_meta.json: {e}")

    def _open_tmp(self) -> None:
        # Append mode allows resume after crash; ensure header if empty/new.
        new_file = (not os.path.exists(self.tmp_path)) or (os.path.getsize(self.tmp_path) == 0)
        self._fp = open(self.tmp_path, "a", newline="", encoding="utf-8")
        self._wr = csv.DictWriter(self._fp, fieldnames=self.fieldnames, extrasaction="ignore")
        if new_file:
            self._wr.writeheader()
            self._fp.flush()

    def _sanitize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in self.fieldnames:
            v = row.get(k, None)
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, float):
                if not np.isfinite(v):
                    v = ""
            out[k] = v
        return out

    def append(self, row: Dict[str, Any]) -> None:
        if self._wr is None or self._fp is None:
            return
        r = self._sanitize_row(row)
        self._wr.writerow(r)
        self._rows_since_flush += 1

        now_mono = time.monotonic()
        if (self._rows_since_flush >= self.flush_every_n) or (
            self.flush_every_s > 0.0 and (now_mono - self._last_flush_mono) >= self.flush_every_s
        ):
            try:
                self._fp.flush()
            except Exception:
                pass
            self._rows_since_flush = 0
            self._last_flush_mono = now_mono

    def t_rel_s(self) -> float:
        return float(time.monotonic() - self._t0_mono)

    def t_epoch_s(self) -> float:
        # Use wall clock for epoch; do not derive from monotonic.
        return float(time.time())

    def write_meta(self, meta: Dict[str, Any]) -> None:
        path = os.path.join(self.session_dir, "run_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def close(self, finalize: bool = True) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass
        self._fp = None
        self._wr = None

        if finalize:
            # Atomic-ish finalize: move tmp to final (overwrite final if exists).
            try:
                os.replace(self.tmp_path, self.final_path)
            except Exception:
                # Fallback: if replace fails, keep tmp but try copy semantics.
                try:
                    import shutil
                    shutil.copyfile(self.tmp_path, self.final_path)
                except Exception:
                    pass


def _safe_sha256(path: str) -> Optional[str]:
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
    except Exception:
        return None
    return None


def _try_git_commit() -> Optional[str]:
    """Best-effort git commit hash (if running inside a git repo)."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else None
    except Exception:
        return None


def _try_git_status_porcelain() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if s else ""
    except Exception:
        return None


class SpeedWindow:
    """Rolling-window robust walking speed estimator from track speed (radar-only)."""
    def __init__(self, window_s: float = 2.0, min_samples: int = 5, method: str = "median"):
        self.window_s = float(window_s)
        self.min_samples = int(max(1, min_samples))
        self.method = str(method).strip().lower()
        self.buf: Deque[Tuple[float, float]] = deque()

    def reset(self):
        self.buf.clear()

    def update(self, t_s: float, speed_mps: float, valid: bool) -> Dict[str, Any]:
        t_s = float(t_s)
        if valid and np.isfinite(speed_mps) and speed_mps >= 0:
            self.buf.append((t_s, float(speed_mps)))
        # trim
        while self.buf and (t_s - self.buf[0][0] > self.window_s):
            self.buf.popleft()
        n = len(self.buf)
        if n < self.min_samples:
            return {"speed_mps": np.nan, "n": int(n)}
        vals = np.array([v for _, v in self.buf], dtype=np.float64)
        if self.method == "mean":
            s = float(np.mean(vals))
        else:
            s = float(np.median(vals))
        return {"speed_mps": s if np.isfinite(s) else np.nan, "n": int(n)}


class WalkingBoutGate:
    """State machine to identify walking bouts and gate cadence/asymmetry."""
    def __init__(self,
                 start_speed_mps: float = 0.25,
                 stop_speed_mps: float = 0.15,
                 start_hold_s: float = 0.50,
                 stop_hold_s: float = 0.80):
        self.start_speed_mps = float(start_speed_mps)
        self.stop_speed_mps = float(stop_speed_mps)
        self.start_hold_s = float(max(0.0, start_hold_s))
        self.stop_hold_s = float(max(0.0, stop_hold_s))
        self.in_bout = False
        self.bout_id = 0
        self._t_above: Optional[float] = None
        self._t_below: Optional[float] = None
        self._t_start: Optional[float] = None

    def reset(self):
        self.in_bout = False
        self.bout_id = 0
        self._t_above = None
        self._t_below = None
        self._t_start = None

    def update(self, t_s: float, speed_mps: float, track_valid: bool) -> Dict[str, Any]:
        t_s = float(t_s)
        v_ok = np.isfinite(speed_mps)
        v = float(speed_mps) if v_ok else np.nan
        tv = bool(track_valid)

        above = tv and v_ok and (v >= self.start_speed_mps)
        below = (not tv) or (not v_ok) or (v <= self.stop_speed_mps)

        event = None
        if not self.in_bout:
            if above:
                if self._t_above is None:
                    self._t_above = t_s
                if (t_s - self._t_above) >= self.start_hold_s:
                    self.in_bout = True
                    self.bout_id += 1
                    self._t_start = t_s
                    self._t_below = None
                    event = "start"
            else:
                self._t_above = None
        else:
            if below:
                if self._t_below is None:
                    self._t_below = t_s
                if (t_s - self._t_below) >= self.stop_hold_s:
                    self.in_bout = False
                    self._t_above = None
                    self._t_below = None
                    self._t_start = None
                    event = "stop"
            else:
                self._t_below = None

        elapsed = (t_s - self._t_start) if (self.in_bout and self._t_start is not None) else 0.0
        return {"in_bout": bool(self.in_bout), "bout_id": int(self.bout_id), "event": event, "elapsed_s": float(elapsed)}


class GaitBoutSummaryCollector:
    """Accumulates per-walking-bout summary metrics and writes them on shutdown.

    Output columns (requested):
      - walking (0/1)
      - gait_speed_mean_mps
      - cadence_spm
      - asymmetry_ai

    A row is emitted per detected walking bout. If no bouts are detected, one row is emitted with walking=0.
    """

    def __init__(self):
        self._active_bout_id: Optional[int] = None
        self._t_start_s: Optional[float] = None
        self._speeds: List[float] = []
        self._cadences: List[float] = []
        self._asym: List[float] = []
        self.rows: List[Dict[str, Any]] = []

    @staticmethod
    def _finite_list_mean(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    @staticmethod
    def _finite_list_median(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else float("nan")

    def _start(self, bout_id: int, t_s: float) -> None:
        self._active_bout_id = int(bout_id)
        self._t_start_s = float(t_s)
        self._speeds = []
        self._cadences = []
        self._asym = []

    def _finish(self, t_end_s: float) -> None:
        if self._active_bout_id is None:
            return
        speed_mean = self._finite_list_mean(self._speeds)  # requested: mean speed
        cadence = self._finite_list_mean(self._cadences)
        asym = self._finite_list_mean(self._asym)

        self.rows.append({
            "walking": 1,
            "gait_speed_mean_mps": speed_mean,
            "cadence_spm": cadence,
            "asymmetry_ai": asym,
            "t_start_s": float(self._t_start_s) if self._t_start_s is not None else float("nan"),
            "t_end_s": float(t_end_s),
        })
        self._active_bout_id = None
        self._t_start_s = None
        self._speeds = []
        self._cadences = []
        self._asym = []

    def update(self,
               t_s: float,
               in_bout: bool,
               bout_id: int,
               event: Optional[str],
               walking_speed_mps: float,
               cadence_spm: float,
               asymmetry_ai: float) -> None:
        t_s = float(t_s)
        bout_id = int(bout_id) if bout_id is not None else 0

        # Recover if we enter a bout without seeing "start"
        if in_bout and self._active_bout_id is None:
            self._start(bout_id=bout_id, t_s=t_s)

        if event == "start":
            self._start(bout_id=bout_id, t_s=t_s)

        if in_bout and self._active_bout_id is not None:
            if np.isfinite(walking_speed_mps):
                self._speeds.append(float(walking_speed_mps))
            if np.isfinite(cadence_spm):
                self._cadences.append(float(cadence_spm))
            if np.isfinite(asymmetry_ai):
                self._asym.append(float(asymmetry_ai))

        if event == "stop":
            self._finish(t_end_s=t_s)

    def finalize(self, t_s: Optional[float] = None) -> None:
        if self._active_bout_id is None:
            return
        t_end = float(t_s) if (t_s is not None and np.isfinite(t_s)) else (
            float(time.time()) if self._t_start_s is None else float(self._t_start_s)
        )
        self._finish(t_end_s=t_end)

    def write_csv(self, session_dir: str, filename: str = "radar_gait_summary.csv") -> None:
        path = os.path.join(session_dir, filename)
        fieldnames = ["walking", "gait_speed_mean_mps", "cadence_spm", "asymmetry_ai"]

        rows = list(self.rows)
        if not rows:
            rows = [{
                "walking": 0,
                "gait_speed_mean_mps": float("nan"),
                "cadence_spm": float("nan"),
                "asymmetry_ai": float("nan"),
            }]

        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(fieldnames)
                for r in rows:
                    ws = int(r.get("walking", 0))
                    sp = r.get("gait_speed_mean_mps", float("nan"))
                    cd = r.get("cadence_spm", float("nan"))
                    ai = r.get("asymmetry_ai", float("nan"))
                    w.writerow([
                        ws,
                        f"{float(sp):.6f}" if np.isfinite(sp) else "",
                        f"{float(cd):.3f}" if np.isfinite(cd) else "",
                        f"{float(ai):.6f}" if np.isfinite(ai) else "",
                    ])
            print(f"[summary] Wrote gait summary: {path}")
        except Exception as e:
            print(f"[summary] WARNING: failed to write gait summary CSV: {e}")


def main():
    # Import hardware libs only when running as a program (keeps module import-safe for the viewer).
    try:
        from mmwave.dataloader import DCA1000
        from mmwave.dataloader.radars import TI
    except Exception as e:
        raise RuntimeError("mmwave package not available. Install mmwave on the radar-capture machine.") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", default=None, help="Directory to write outputs (default: sessions/radar_<timestamp>)")
    ap.add_argument("--auto-start", action="store_true", help="Skip ENTER prompt")
    ap.add_argument("--no-pub", action="store_true", help="Disable ZMQ publishing (still computes metrics)")
    ap.add_argument("--pub-addr", default=RADAR_PUB_ADDR, help="ZMQ PUB bind address")
    ap.add_argument("--cli", default="COM4")
    ap.add_argument("--data", default="COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--radar-cfg", default="configFiles/xWR6843_profile_3D.cfg")
    ap.add_argument("--dca-json", default="configFiles/cf.json")

    # Steps (5)-(8) controls
    ap.add_argument("--log-csv", action="store_true", help="Write one consolidated gait CSV (reduces clutter)")
    ap.add_argument("--csv-name", default="radar_gait_metrics.csv", help="CSV filename inside session dir")
    ap.add_argument("--speed-window-s", type=float, default=2.0)
    ap.add_argument("--speed-window-min-samples", type=int, default=5)
    ap.add_argument("--speed-window-method", choices=["median", "mean"], default="median")
    ap.add_argument("--bout-start-speed", type=float, default=0.25)
    ap.add_argument("--bout-stop-speed", type=float, default=0.15)
    ap.add_argument("--bout-start-hold-s", type=float, default=0.50)
    ap.add_argument("--bout-stop-hold-s", type=float, default=0.80)
    ap.add_argument("--gait-active-speed", type=float, default=0.15, help="Only run cadence/asym above this speed")
    ap.add_argument("--reset-gait-on-bout", action="store_true", help="Reset cadence/asym state on bout start/stop")
    args = ap.parse_args()

    # ---------- session directory ----------
    if args.session_dir:
        session_dir = args.session_dir
        base = os.path.basename(os.path.normpath(session_dir))
        if base.startswith("radar_"):
            session_id = base.split("radar_", 1)[1]
        else:
            session_id = base
    else:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join("sessions", f"radar_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    meta_path = os.path.join(session_dir, "session_meta.json")

    dca = None
    radar = None
    pub = None
    pub_ctx = None

    metrics_logger: Optional[GaitMetricsLogger] = None

    try:
        dca = DCA1000()
        dca.reset_radar()
        dca.reset_fpga()
        time.sleep(1)

        radar = TI(cli_loc=args.cli, data_loc=args.data, data_baud=args.baud,
                   config_file=args.radar_cfg, verbose=True)

        numLoops = 100000
        frameNumInBuf = 16
        numframes = 16
        radar.setFrameCfg(0)  # infinite frames
        if DEMO_MODE:
            frameNumInBuf = int(DEMO_FRAMEBUF)
            numframes = int(DEMO_NUMFRAMES)

        ADC_PARAMS_l, _ = dca.configure(args.dca_json, args.radar_cfg)

        # Save metadata (include key gait config + provenance)
        radar_cfg_hash = _safe_sha256(args.radar_cfg)
        dca_json_hash = _safe_sha256(args.dca_json)
        script_path = os.path.abspath(__file__) if "__file__" in globals() else None
        script_hash = _safe_sha256(script_path) if script_path else None
        git_commit = _try_git_commit()
        git_dirty = _try_git_status_porcelain()
        meta = {
            "session_id": session_id,
            "start_time_iso": datetime.now().isoformat(),
            "session_dir": session_dir,
            "timezone": time.tzname[0] if time.tzname else None,
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "git": {"commit": git_commit, "dirty_porcelain": git_dirty},
            "files": {
                "radar_config_file": args.radar_cfg,
                "radar_config_sha256": radar_cfg_hash,
                "dca_config_file": args.dca_json,
                "dca_config_sha256": dca_json_hash,
                "script_path": script_path,
                "script_sha256": script_hash,
            },
            "adc_params": ADC_PARAMS_l,
            "radar_pub": None if (args.no_pub or (not ZMQ_AVAILABLE)) else str(args.pub_addr),
            "speed_source": "range_azimuth_tracking",
            "steps_1_4": {"beamforming": BEAMFORM_ENABLE, "detection": DETECT_ENABLE, "tracking": TRACK_ENABLE},
            "steps_5_8": {
                "speed_window_s": float(args.speed_window_s),
                "speed_window_min_samples": int(args.speed_window_min_samples),
                "speed_window_method": str(args.speed_window_method),
                "bout_start_speed": float(args.bout_start_speed),
                "bout_stop_speed": float(args.bout_stop_speed),
                "bout_start_hold_s": float(args.bout_start_hold_s),
                "bout_stop_hold_s": float(args.bout_stop_hold_s),
                "gait_active_speed": float(args.gait_active_speed),
                "reset_gait_on_bout": bool(args.reset_gait_on_bout),
            },
        }
        # Write provenance (always) + also attach to metrics logger (if enabled)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
        if metrics_logger is not None:
            try:
                metrics_logger.write_meta(meta)
            except Exception:
                pass

        # ZMQ publisher
        if (not args.no_pub) and ZMQ_AVAILABLE:
            pub_ctx = zmq.Context.instance()
            pub = pub_ctx.socket(zmq.PUB)
            pub.bind(str(args.pub_addr))
            print(f"[zmq] PUB bound: {args.pub_addr}")
        elif not args.no_pub:
            print("[zmq] WARNING: pyzmq not installed; publishing disabled.")

        # Optional tidy-row metrics CSV (append-only + crash-safe finalize)
        if args.log_csv:
            metrics_fieldnames = [
                "t_rel_s", "t_epoch_s", "frame_idx",
                "subject_id",
                "walking", "gait_speed_mean_mps", "cadence_spm", "asymmetry_ai",
                "track_valid", "x_m", "y_m", "vx_mps", "vy_mps",
                "torso_range_m", "az_deg", "az_conf", "r_bin",
                "quality", "flags"
            ]
            metrics_logger = GaitMetricsLogger(
                session_dir=session_dir,
                filename=str(args.csv_name),
                fieldnames=metrics_fieldnames,
                meta=None,  # meta written below once constructed
                flush_every_n=25,
                flush_every_s=1.0,
            )
            print(f"[metrics] Tidy gait log (tmp): {os.path.join(session_dir, str(args.csv_name))}.tmp")
            try:
                metrics_logger.write_meta(meta)
            except Exception:
                pass

        # Frame period (for synthetic gait timeline)
        frame_period_s = _frame_period_s_from_params(ADC_PARAMS_l, GAIT_DETECT_DEFAULT_FRAME_PERIOD_S)

        # Gait detectors
        Nd_total = int(ADC_PARAMS_l['chirps'])
        Ntx = int(ADC_PARAMS_l['tx'])
        Nd_eff = Nd_total // max(Ntx, 1)

        gait_detector = None
        asym_est = None
        if GAIT_DETECT_ENABLE and SCIPY_AVAILABLE:
            gait_detector = MuDStepCycleDetector(Nd_eff)
            if ASYM_ENABLE:
                asym_est = StepAsymmetryEstimator()
        elif GAIT_DETECT_ENABLE and (not SCIPY_AVAILABLE):
            print("[gait] WARNING: SciPy not available; cadence/asymmetry disabled.")

        # Tracker and speed/bout gating
        tracker = SingleTargetTracker2D(speed_ema_alpha=TRACK_SPEED_EMA_ALPHA) if TRACK_ENABLE else None
        speed_win = SpeedWindow(window_s=float(args.speed_window_s),
                                min_samples=int(args.speed_window_min_samples),
                                method=str(args.speed_window_method))
        bout_gate = WalkingBoutGate(start_speed_mps=float(args.bout_start_speed),
                                    stop_speed_mps=float(args.bout_stop_speed),
                                    start_hold_s=float(args.bout_start_hold_s),
                                    stop_hold_s=float(args.bout_stop_hold_s))

        # Per-bout gait summary (written on shutdown)
        bout_summary = GaitBoutSummaryCollector()
        last_ts_gait = np.nan


        radial_qc = None
        if RADIAL_QC_ENABLE:
            # Build vel axis once for QC
            Nr_guess = int(ADC_PARAMS_l.get('samples', 256)) // 2
            _, vel_mps_axis = _compute_axes_from_cfg({**ADC_PARAMS_l}, Nr_guess, Nd_eff, Ntx)
            radial_qc = RadialSpeedEstimatorQC(vel_mps_axis)

        if not args.auto_start:
            input("press ENTER to start capture...")
        else:
            print("[auto] starting capture without ENTER")

        dca.stream_start()
        dca.fastRead_in_Cpp_thread_start(frameNumInBuf)
        radar.startSensor()

        t_synth0 = time.time()
        frame_idx = 0

        for loop_idx in range(numLoops):
            data_buf = dca.fastRead_in_Cpp_thread_get(numframes, verbose=True, sortInC=True)

            if (not DEMO_MODE) and (not DEMO_DISABLE_NPY_SAVES):
                try:
                    np.save(os.path.join(session_dir, f"adc_loop{loop_idx:06d}.npy"), data_buf)
                except Exception:
                    pass

            # Processing (keeps detection/association logic intact)
            E_fr, md_cols, det = postProc(data_buf, ADC_PARAMS_l)
            cand_lists = det.get("candidates", None)
            best = det.get("best", {})

            for k in range(md_cols.shape[0]):
                ts_wall = time.time()
                ts_gait = (t_synth0 + frame_idx * frame_period_s) if GAIT_DETECT_USE_SYNTH_TS else ts_wall

                col_lin = md_cols[k].astype(np.float32)
                col_lin = np.nan_to_num(col_lin, nan=0.0, posinf=0.0, neginf=0.0)
                col_db = (10.0 * np.log10(col_lin + 1e-12)).astype(np.float32)

                # Build candidate measurements for tracker (multi-candidate association preserved)
                meas_candidates: List[Dict[str, Any]] = []
                raw_cands = []
                if cand_lists is not None and k < len(cand_lists):
                    raw_cands = cand_lists[k] or []
                if (not raw_cands) and isinstance(best, dict):
                    try:
                        r_m = float(best["range_m"][k])
                        az = float(best["az_deg"][k])
                        conf = float(best["az_conf"][k])
                        rb = int(best["r_bin"][k])
                        if np.isfinite(r_m) and np.isfinite(az) and conf >= BF_MIN_CONF:
                            raw_cands = [{"range_m": r_m, "az_deg": az, "az_conf": conf, "r_bin": rb, "score": conf}]
                    except Exception:
                        raw_cands = []

                for c in raw_cands:
                    try:
                        r_m = float(c.get("range_m", np.nan))
                        az = float(c.get("az_deg", np.nan))
                        conf = float(c.get("az_conf", 0.0))
                        rb = int(c.get("r_bin", -1))
                        score = float(c.get("score", 0.0))
                        if (not np.isfinite(r_m)) or (not np.isfinite(az)) or (conf < float(BF_MIN_CONF)):
                            continue
                        z, R = _measurement_from_range_az(r_m, az, conf)
                        meas_candidates.append({
                            "z": z, "R": R,
                            "score": score,
                            "range_m": r_m, "az_deg": az, "az_conf": conf, "r_bin": rb,
                        })
                    except Exception:
                        continue

                if DETECT_CLUSTER_ENABLE and len(meas_candidates) > 1:
                    meas_candidates = _cluster_candidates_xy(meas_candidates, float(DETECT_CLUSTER_DIST_M))

                track_out = tracker.step_candidates(ts_gait, meas_candidates) if tracker is not None else {"valid": False, "n_cands": len(meas_candidates)}
                track_valid = bool(track_out.get("valid", False))
                track_speed = float(track_out.get("speed_mps", np.nan)) if track_valid else np.nan
                track_speed_ema = float(track_out.get("speed_ema_mps", np.nan)) if track_valid else np.nan

                # Robust walking speed from rolling window of track speed (uses EMA speed if available)
                speed_input = track_speed_ema if np.isfinite(track_speed_ema) else track_speed
                sw = speed_win.update(ts_gait, speed_input, valid=track_valid)
                walking_speed = float(sw.get("speed_mps", np.nan))
                walking_speed_n = int(sw.get("n", 0))

                # Bout gating
                bout = bout_gate.update(ts_gait, walking_speed, track_valid=track_valid)
                in_bout = bool(bout.get("in_bout", False))
                bout_id = int(bout.get("bout_id", 0))
                bout_elapsed = float(bout.get("elapsed_s", 0.0))

                # Reset gait state on bout transitions (without changing steps 1-4 core)
                if args.reset_gait_on_bout and bout.get("event") in ("start", "stop"):
                    if GAIT_DETECT_ENABLE and SCIPY_AVAILABLE:
                        gait_detector = MuDStepCycleDetector(Nd_eff)
                        asym_est = StepAsymmetryEstimator() if ASYM_ENABLE else None
                    else:
                        gait_detector = None
                        asym_est = None

                # Gate cadence/asymmetry to active walking
                gait_active = bool(in_bout and track_valid and np.isfinite(walking_speed) and (walking_speed >= float(args.gait_active_speed)))

                gait_metrics = None
                asym_metrics = None
                if gait_detector is not None and gait_active:
                    gait_metrics = gait_detector.update(ts_gait, col_lin)
                    if gait_metrics is not None and asym_est is not None:
                        asym_metrics = asym_est.update(gait_metrics.get("new_steps_interval_s", None))

                cadence_spm = float(gait_metrics.get("cadence_spm", np.nan)) if gait_metrics else np.nan
                step_count_window = int(gait_metrics.get("step_count_window", 0)) if gait_metrics else 0
                step_interval_mean_s = float(gait_metrics.get("step_interval_mean_s", np.nan)) if gait_metrics else np.nan
                step_interval_last_s = float(gait_metrics.get("step_interval_last_s", np.nan)) if gait_metrics else np.nan

                asym_ai = float(asym_metrics.get("asym_ai", np.nan)) if asym_metrics else np.nan
                mu_even = float(asym_metrics.get("mu_even_s", np.nan)) if asym_metrics else np.nan
                mu_odd = float(asym_metrics.get("mu_odd_s", np.nan)) if asym_metrics else np.nan
                asym_n = int(asym_metrics.get("n", 0)) if asym_metrics else 0
                # Update per-bout gait summary accumulator
                if bout_summary is not None:
                    bout_summary.update(
                        t_s=ts_gait,
                        in_bout=in_bout,
                        bout_id=bout_id,
                        event=bout.get("event", None) if isinstance(bout, dict) else None,
                        walking_speed_mps=walking_speed,
                        cadence_spm=cadence_spm,
                        asymmetry_ai=asym_ai,
                    )
                last_ts_gait = ts_gait

                qc_v = qc_conf = np.nan
                if radial_qc is not None:
                    qc = radial_qc.update(col_lin)
                    qc_v = float(qc.get("v_rad_mps", np.nan))
                    qc_conf = float(qc.get("conf", np.nan))

                # Measurement used by tracker (if any)
                meas = track_out.get("meas", None) if isinstance(track_out, dict) else None
                meas_range_m = meas.get("range_m") if isinstance(meas, dict) else None
                meas_az_deg = meas.get("az_deg") if isinstance(meas, dict) else None
                meas_az_conf = meas.get("az_conf") if isinstance(meas, dict) else None
                meas_r_bin = meas.get("r_bin") if isinstance(meas, dict) else None

                # Consolidated gait summary
                gait_summary = {
                    "t_gait_s": float(ts_gait),
                    "bout_id": bout_id,
                    "in_bout": in_bout,
                    "bout_elapsed_s": bout_elapsed,
                    "walking_speed_mps": float(walking_speed) if np.isfinite(walking_speed) else None,
                    "walking_speed_n": walking_speed_n,
                    "cadence_spm": float(cadence_spm) if np.isfinite(cadence_spm) else None,
                    "asym_ai": float(asym_ai) if np.isfinite(asym_ai) else None,
                    "asym_n": asym_n,
                    "track_valid": track_valid,
                }

                # Optional tidy metrics log (append-only)
                if metrics_logger is not None:
                    try:
                        # Define walking as the bout-gate state (stable) rather than instantaneous speed threshold.
                        walking_flag = 1 if in_bout else 0

                        # Torso range proxy: use tracker measurement range when available, else NaN.
                        torso_range_m = float(meas_range_m) if isinstance(meas_range_m, (float, int)) and np.isfinite(meas_range_m) else np.nan

                        # Basic quality score: 0..1 from tracker measurement confidence; set 0 if not valid.
                        quality = float(track_out.get("meas_conf", 0.0)) if track_valid else 0.0

                        # Flags: compact bitfield-style string for debugging (optional)
                        flags = []
                        if not track_valid:
                            flags.append("no_track")
                        if not gait_active:
                            flags.append("gait_inactive")
                        if gait_detector is None:
                            flags.append("no_gait_detector")
                        flags_s = "|".join(flags) if flags else ""

                        metrics_logger.append({
                            "t_rel_s": metrics_logger.t_rel_s(),
                            "t_epoch_s": float(ts_wall),
                            "frame_idx": int(frame_idx),
                            "subject_id": 0,
                            "walking": int(walking_flag),
                            "gait_speed_mean_mps": float(walking_speed) if np.isfinite(walking_speed) else "",
                            "cadence_spm": float(cadence_spm) if np.isfinite(cadence_spm) else "",
                            "asymmetry_ai": float(asym_ai) if np.isfinite(asym_ai) else "",
                            "track_valid": int(track_valid),
                            "x_m": float(track_out.get("x_m", np.nan)) if track_valid else "",
                            "y_m": float(track_out.get("y_m", np.nan)) if track_valid else "",
                            "vx_mps": float(track_out.get("vx_mps", np.nan)) if track_valid else "",
                            "vy_mps": float(track_out.get("vy_mps", np.nan)) if track_valid else "",
                            "torso_range_m": float(torso_range_m) if np.isfinite(torso_range_m) else "",
                            "az_deg": float(meas_az_deg) if isinstance(meas_az_deg, (float, int)) and np.isfinite(meas_az_deg) else "",
                            "az_conf": float(meas_az_conf) if isinstance(meas_az_conf, (float, int)) and np.isfinite(meas_az_conf) else "",
                            "r_bin": int(meas_r_bin) if isinstance(meas_r_bin, int) else "",
                            "quality": float(quality),
                            "flags": flags_s,
                        })
                    except Exception:
                        pass

                # Publish (keep md_db for viewer; include tracking + gait summary)
                if pub is not None:
                    try:
                        if DEMO_MODE and DEMO_PUB_THIN > 1 and ((frame_idx % int(DEMO_PUB_THIN)) != 0):
                            pass
                        else:
                            payload = {
                                "ts": float(ts_wall),
                                "md_db": col_db,
                                "range": E_fr[k].astype(np.float32) if (E_fr is not None and E_fr.size) else None,
                                "track": track_out if track_valid else {"valid": False},
                                "gait": gait_summary,
                            }
                            if radial_qc is not None:
                                payload["qc"] = {
                                    "v_rad_mps": float(qc_v) if np.isfinite(qc_v) else None,
                                    "conf": float(qc_conf) if np.isfinite(qc_conf) else None,
                                    "v_rad_pred_mps": float(track_out.get("v_rad_pred_mps", np.nan)) if track_valid and np.isfinite(track_out.get("v_rad_pred_mps", np.nan)) else None,
                                }
                            pub.send_pyobj(payload, flags=0)
                    except Exception:
                        pass

                frame_idx += 1

    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt; stopping.")
    except Exception:
        traceback.print_exc()
    finally:
        # Persist per-bout gait summary on shutdown
        try:
            if bout_summary is not None and session_dir is not None:
                bout_summary.finalize(t_s=(last_ts_gait if np.isfinite(last_ts_gait) else None))
                bout_summary.write_csv(session_dir)
        except Exception:
            traceback.print_exc()

        if metrics_logger is not None:
            try:
                metrics_logger.close(finalize=True)
            except Exception:
                pass

        if radar is not None:
            try:
                radar.stopSensor()
            except Exception:
                pass
        if dca is not None:
            try:
                dca.fastRead_in_Cpp_thread_stop()
            except Exception:
                pass
            try:
                dca.stream_stop()
            except Exception:
                pass
            try:
                dca.close()
            except Exception:
                pass

        if pub is not None:
            try:
                pub.close(linger=0)
            except Exception:
                pass


if __name__ == "__main__":
    main()
