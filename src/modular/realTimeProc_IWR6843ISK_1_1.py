# -*- coding: utf-8 -*-
# NOTE (2025-11-16): Presence-gated publishing.

# Smart Sensing and Computing for Dementia Care
# The goal of this project is to develop sensing systems to collect data regarding risk assessment and prevention.

# ---------- Imports ----------
import traceback
import time
from mmwave.dataloader import DCA1000
from mmwave.dataloader.radars import TI
import numpy as np
from datetime import datetime
import queue, threading
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
import os
import csv
import json
import argparse  # <-- existing import retained (needed for --session-dir / --auto-start)

# ========================= DEMO CONTROL SWITCHES =========================
# Turn this ON for the live demo. It:
#   - Skips heavy per-loop np.save() writes
#   - Uses smaller UDP batch sizes to reduce burstiness
#   - Enlarges the producer queue to avoid drops
#   - Skips Appendix-style extra computations (AoA/ghost/wall logs)
#   - Throttles ZMQ publishing (publish every k-th column)
DEMO_MODE: bool = True
DEMO_QUEUE_MAXSIZE: int = 8192          # headroom for bursts in demo
DEMO_FRAMEBUF: int = 8                  # frameNumInBuf in demo (smaller batch)
DEMO_NUMFRAMES: int = 8                 # numframes <= frameNumInBuf in demo
DEMO_PUB_THIN: int = 2                  # publish every k-th μD column during demo (1 = no thinning)

# ---------- Global handles ----------
dca = None
radar = None

# ---- OFFLINE PLOT (like draft.py) after the live run ----
PLOT_FINAL_AFTER_RUN = True
SAVE_FINAL_PNG = False
SAVE_FINAL_NPY  = False

_final_md_chunks = []        # will append per-loop md_cols (linear power)
_md_time_fp = None
_md_time_csv = None
SESSION_ID = None
SESSION_DIR = None

# ---- LIVE soft baseline (viewer shaping; no hard notch) ----
LIVE_SOFT_FLOOR = False
LIVE_BASELINE_TAU = 0.995  # higher = slower background update

# ------------------------- tunables for robustness -------------------------
# OPT: Removed unused CAL_* and ZERO_NOTCH_BINS; they had no effect on outputs.

# ------------------------- SNR-improvers (new tunables) -------------------------
# Demo focuses on stability, so enable a light MTI and keep other costs low.
MTI_CANCELLER = 2        # 0: off (mean-subtract), 1: 2-pulse, 2: 3-pulse (recommended)
NOISE_SUBTRACT_Q = 0.10  # robust per-Doppler noise floor (quantile across range), 0..1
DOPPLER_SMOOTH_K = 1     # moving-average across Doppler bins (odd int; 1 disables)
FRAME_EWMA = 0.00        # EWMA across frames inside this batch for the md column

EPS = 1e-12

# ---------- Small caches (avoid recomputing windows/axes per batch) ----------
_WIN_CACHE = {}   # keys: ('wr', Nr), ('wd', Nd_eff)
_AXES_CACHE = {}  # key: (id(adc_params), Nr_half, Nd_eff, Ntx)

_ZMQ_CTX = None
_ZMQ_PUB = None               # existing μD PUB tcp://127.0.0.1:5557

# ------------------------- Asymmetry engine global -------------------------


def _build_ula_steering_matrix(num_elems, wavelength, d_elem, angles_deg):
    """
    Uniform linear array steering matrix.
    Returns A: [num_angles, num_elems]
    """
    angles = np.deg2rad(angles_deg).astype(np.float32)
    elem_idx = np.arange(num_elems, dtype=np.float32)[None, :]  # [1, V]
    # phase = 2π d/λ * n * sin(theta)
    phase = 2.0 * np.pi * d_elem / wavelength * elem_idx * np.sin(angles[:, None])
    return np.exp(1j * phase).astype(np.complex64)


def _capon_beamform_frame(Xfrd, A, diag_load=1e-3):
    """
    Xfrd: [R, Nd, V] complex, single frame (from Xrd_virtual[f])
    A:    [Na, V] steering matrix
    Returns P_RA: [R, Na] real, range–azimuth power (Capon)
    """
    Rnum, Nd, V = Xfrd.shape
    Na = A.shape[0]
    P_RA = np.zeros((Rnum, Na), dtype=np.float32)

    # Use a small Doppler band around 0 Hz (hallway paper uses torso bin / walking speed)
    center = Nd // 2
    d_win = max(3, Nd // 16)
    d_lo = max(0, center - d_win)
    d_hi = min(Nd, center + d_win + 1)

    A_H = np.conjugate(A.T)  # [V, Na]

    for r in range(Rnum):
        # snapshots: [snapshots, V]
        Xsnap = Xfrd[r, d_lo:d_hi, :]           # [Nd_sel, V]
        if Xsnap.size == 0:
            continue

        # Sample covariance R = X^H X / N
        R = (Xsnap.conj().T @ Xsnap) / Xsnap.shape[0]   # [V, V]
        # Diagonal loading for robustness
        R.flat[::V+1] += diag_load * np.trace(R).real / V

        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            continue

        # Capon spectrum: P(θ) = 1 / (a^H R^{-1} a)
        # Compute in vectorized form: denom[Na] = diag(A R_inv A^H)
        ARinv = A @ R_inv        # [Na, V]
        denom = np.einsum('av,va->a', ARinv, A_H)  # [Na]
        denom = np.real(denom)
        denom[denom <= 0] = np.min(denom[denom > 0]) if np.any(denom > 0) else 1.0

        P_RA[r, :] = 1.0 / denom.astype(np.float32)

    # Normalize / convert to dB for visualization
    eps = 1e-9
    P_RA = 10.0 * np.log10(P_RA / (np.max(P_RA) + eps) + eps)
    return P_RA


def save_final_md_timealigned(md_chunks, session_dir, adc_params, fname="micro_doppler_time.png"):
    """
    Save a micro-Doppler spectrogram with a time axis aligned to radar_md_timestamps.csv.
    No UI is shown. Output is <session_dir>/<fname>.
    """
    import csv, numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime

    if not md_chunks:
        print("[final] No columns buffered; skipping spectrogram save.")
        return

    # Concatenate linear-power μD columns accumulated during the run
    md = np.concatenate(md_chunks, axis=0)   # [F_total, Nd]
    Nd_eff = md.shape[1]

    # Load per-column times recorded during the run
    ts_csv = os.path.join(session_dir, "radar_md_timestamps.csv")
    times = []
    try:
        with open(ts_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                times.append(float(row["t"]))
    except Exception as e:
        print(f"[final] WARN: cannot read timestamps from {ts_csv}: {e}")
        return

    if not times:
        print("[final] No timestamp rows found; skipping spectrogram save.")
        return

    # Truncate to the shorter length if needed
    n = min(len(times), md.shape[0])
    if len(times) != md.shape[0]:
        print(f"[final] NOTE: time/column count mismatch; using n={n} rows.")
    md = md[:n, :]
    t = np.asarray(times[:n], dtype=np.float64)

    # Time axis: seconds since first μD column for readability
    t0 = float(t[0])
    t_rel = t - t0

    # Compute velocity axis from ADC params (same helper used elsewhere in the file)
    Nr_half_guess = max(1, int(adc_params.get('samples', 0)) // 2)
    Ntx_guess = int(adc_params.get('tx') or adc_params.get('num_tx_like') or 1)
    _, vel_mps = _compute_axes_from_cfg(adc_params, Nr_half_guess, Nd_eff, Ntx_guess)

    # dB scaling and image save (no plt.show)
    md_db = 20.0 * np.log10(np.maximum(md, 1e-12)).astype(np.float32)
    plt.figure(figsize=(12, 4))
    plt.imshow(
        md_db.T, origin='lower', aspect='auto', cmap='jet',
        extent=[float(t_rel[0]), float(t_rel[-1]), float(vel_mps[0]), float(vel_mps[-1])]
    )
    plt.xlabel("Time since start (s)")
    plt.ylabel("Radial velocity (m/s)")
    plt.title("Micro-Doppler (time-aligned)")
    plt.tight_layout()

    out_png = os.path.join(session_dir, fname)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[final] saved spectrogram: {out_png}")


# #------------------------- Appendix-like controls (removed) -------------------------
# CLUTTER_WIDTH_BINS = 2
# GHOST_SUPPRESS = False
# GHOST_BOX_R = 2
# GHOST_BOX_D = 3
# LOG_WALL_EST_EVERY = 0 if DEMO_MODE else 30  # silence appendix logs in demo
#
# # ------------------------- Appendix-like helpers (removed) -------------------------
# def angle_process_rdvec(rd_vec_v, n_fft=128):
#     ang_fft = np.fft.fft(rd_vec_v, n=n_fft)
#     ang_fft = np.fft.fftshift(ang_fft)
#     angle_peak_idx = int(np.argmax(np.abs(ang_fft)))
#     s = 2.0 * (angle_peak_idx - (ang_fft.shape[0] // 2)) / float(ang_fft.shape[0])
#     s = np.clip(s, -1.0, 1.0)
#     angle = np.rad2deg(np.arcsin(s))
#     return ang_fft, float(angle)
#
# def find_peak(range_velocity_db, clutter_width_bins, meters_per_bin):
#     rv_mod = range_velocity_db.copy()
#     nv = rv_mod.shape[1]
#     c = nv // 2
#     w = int(clutter_width_bins)
#     w = max(0, w)
#     rv_mod[:, max(0, c-w):min(nv, c+w)] = -np.inf
#     idx = np.unravel_index(np.nanargmax(rv_mod), rv_mod.shape)
#     power_db = rv_mod[idx]
#     distance_m = float(idx[0]) * float(meters_per_bin)
#     return rv_mod, float(power_db), distance_m, idx
#
# def find_second_peak(range_velocity_db, main_idx, meters_per_bin):
#     rv_mod = range_velocity_db.copy()
#     r, d = main_idx
#     rv_mod[max(0, r-2):r+3, max(0, d-3):d+4] = -np.inf
#     rv_mod[:r+1, :] = -np.inf
#     rv_mod[:, d:] = -np.inf
#     idx2 = np.unravel_index(np.nanargmax(rv_mod), rv_mod.shape)
#     distance_ghost_m = float(idx2[0]) * float(meters_per_bin)
#     power2_db = rv_mod[idx2]
#     return idx2, distance_ghost_m, float(power2_db)
#
# def inverse_wall_non_radial(d_direct_idx, v_real_idx, d_ghost_idx, v_ghost_idx,
#                             theta_ghost_deg, theta_direct_deg=0.0, meters_per_bin=0.3125):
#     import math
#     theta_rad = math.radians(theta_direct_deg)
#     theta_ghost_rad = math.radians(theta_ghost_deg)
#     b = 4.0 * d_direct_idx * math.sin(theta_rad)
#     d_wall_d = meters_per_bin/8.0 * (b + math.sqrt(max(b*b - 16.0*(d_direct_idx**2 - d_ghost_idx**2), 0.0)))
#     arg = math.acos(max(min(v_ghost_idx/float(v_real_idx + (1e-9)), 1.0), -1.0)) - theta_rad
#     d_wall_v = meters_per_bin/2.0 * d_direct_idx * (math.cos(theta_rad)*math.tan(arg) + math.sin(theta_rad))
#     d_wall_angle = meters_per_bin/2.0 * d_direct_idx * (math.cos(theta_rad)*math.tan(theta_ghost_rad) + math.sin(theta_rad))
#     return float(d_wall_d), float(d_wall_v), float(d_wall_angle)
#
# # ------------------------- Optional appendix callsite (removed) -------------------------
# if (not DEMO_MODE) and (do_ghost_suppress or log_wall_every):
#     for fidx in range(Fnum):
#         RD_amp = np.sqrt(np.maximum(P_frd_work[fidx], 0.0) + EPS)
#         RD_db = 20.0 * _safe_log10(RD_amp)
#         cwidth = max(CLUTTER_WIDTH_BINS, Nd // 64)
#         rv_mod, power_db, distance_m, idx = find_peak(RD_db, cwidth, meters_per_bin)
#         idx2, distance_ghost_m, power2_db = find_second_peak(rv_mod, idx, meters_per_bin)
#         r0, d0 = int(idx[0]),  int(idx[1])
#         r1, d1 = int(idx2[0]), int(idx2[1])
#         try:
#             _, angle_direct = angle_process_rdvec(Xrd_virtual[fidx, r0, d0, :], n=128)
#         except Exception:
#             angle_direct = 0.0
#         try:
#             _, angle_ghost = angle_process_rdvec(Xrd_virtual[fidx, r1, d1, :], n=128)
#         except Exception:
#             angle_ghost = 0.0
#         d_wall_d, d_wall_v, d_wall_angle = inverse_wall_non_radial(
#             d_direct_idx=r0, v_real_idx=max(d0, 1),
#             d_ghost_idx=r1,  v_ghost_idx=max(d1, 1),
#             theta_ghost_deg=angle_ghost, theta_direct_deg=angle_direct,
#             meters_per_bin=meters_per_bin
#         )
#         if log_wall_every and (fidx % log_wall_every == 0):
#             print(f"[appendix] f={fidx} main@({r0},{d0}) ghost@({r1},{d1}) "
#                   f"AoA(main,ghost)=({angle_direct:.1f},{angle_ghost:.1f}) "
#                   f"a_est(m): d={d_wall_d:.2f} v={d_wall_v:.2f} ang={d_wall_angle:.2f}")
#         if do_ghost_suppress:
#             r_lo = max(0, r1 - GHOST_BOX_R); r_hi = min(Rnum, r1 + GHOST_BOX_R + 1)
#             d_lo = max(0, d1 - GHOST_BOX_D); d_hi = min(Nd,   d1 + GHOST_BOX_D + 1)
#             P_frd_work[fidx, r_lo:r_hi, d_lo:d_hi] = 0.0

# ------------------------- Axes computation -------------------------
def _compute_axes_from_cfg(params: dict, Nr_half: int, Nd_eff: int, Ntx_hint: int):
    c = 299_792_458.0
    rng_m = np.arange(Nr_half, dtype=np.float32)
    vel_mps = (np.arange(Nd_eff, dtype=np.float32) - (Nd_eff // 2)).astype(np.float32)
    try:
        slope_mhz_per_us = (params.get('slope_mhz_per_us') or
                            params.get('freqSlopeConst') or
                            params.get('slopeMhzPerUs') or
                            params.get('slope'))
        start_freq_ghz   = (params.get('start_freq_ghz') or
                            params.get('startFreq') or
                            params.get('startFreqGHz') or
                            params.get('start_freq'))
        ramp_end_time_us = (params.get('ramp_end_time_us') or
                            params.get('rampEndTime') or
                            params.get('rampEndTime_us') or
                            params.get('ramp_end_time'))
        idle_time_us     = (params.get('idle_time_us') or
                            params.get('idleTime') or
                            params.get('idle_time'))
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

        B  = slope_Hz_per_s * Tr
        dR = c / (2.0 * B)
        rng_m = (dR * np.arange(Nr_half, dtype=np.float32)).astype(np.float32)

        # ---- LOCK TD TO SAME-TX INTERVAL, IGNORE FRAME-DERIVED PATH ----
        TD = Ntx_eff * (Ti + Tr)

        fd = np.fft.fftshift(np.fft.fftfreq(Nd_eff, d=TD)).astype(np.float32)
        vel_mps = (fd * lam / 2.0).astype(np.float32)

        vmax = float(max(abs(vel_mps[0]), abs(vel_mps[-1])))
        print(f"[axes] f0={f0_GHz:.1f} GHz  λ={lam:.4f} m  "
              f"Tchirp={((Ti+Tr)*1e6):.2f} µs  TD={TD*1e6:.2f} µs  "
              f"Ntx={Ntx_eff}  vmax≈{vmax:.2f} m/s  dR≈{dR:.3f} m")

    except Exception:
        rng_m = np.arange(Nr_half, dtype=np.float32)
        vel_mps = (np.arange(Nd_eff, dtype=np.float32) - (Nd_eff // 2)).astype(np.float32)

    return rng_m, vel_mps


def _safe_log10(x):
    return np.log10(np.maximum(x, EPS)).astype(np.float32)


def _apply_mti(x: np.ndarray, order: int) -> np.ndarray:
    if order <= 0:
        return x
    y = np.zeros_like(x)
    if order == 1:
        y[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
    else:
        y[:, 2:, :, :] = x[:, 2:, :, :] - 2.0 * x[:, 1:-1, :, :] + x[:, :-2, :, :]
    return y

# ------------------------- main processing -------------------------
def postProc(adc_data, ADC_PARAMS_l):
    """
    Post-processes raw ADC radar data into range–Doppler (RD) and micro-Doppler (μD) products.

    Returns:
      E_fr (F, R)          : per-frame range energy (linear)
      md_cols (F, Nd_eff)  : micro-Doppler columns (linear)

    This function:
      - Performs Range FFT and Doppler FFT
      - Optionally applies MTI filtering and noise-floor subtraction
      - Collapses range dimension using Gaussian range weighting
      - Publishes μD columns via ZMQ (tcp://127.0.0.1:5557)
    """
    global _ZMQ_CTX, _ZMQ_PUB

    # ---------- Setup sockets ----------
    try:
        _ZMQ_CTX
    except NameError:
        _ZMQ_CTX = None
        _ZMQ_PUB = None

    if _ZMQ_CTX is None or _ZMQ_PUB is None:
        try:
            import zmq
            _ZMQ_CTX = zmq.Context.instance()
            _ZMQ_PUB = _ZMQ_CTX.socket(zmq.PUB)
            _ZMQ_PUB.bind("tcp://127.0.0.1:5557")
        except Exception:
            _ZMQ_CTX = None
            _ZMQ_PUB = None

    # ---------- Reshape ADC data ----------
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

    # ---------- Range FFT ----------
    win_key = ('wr', Nr)
    if win_key not in _WIN_CACHE:
        _WIN_CACHE[win_key] = np.hanning(Nr).astype(np.float32)[None, None, None, None, :]
    wr = _WIN_CACHE[win_key]
    Xr = np.fft.fft(adc_cplx * wr, n=Nr, axis=-1)

    # Keep positive beat half
    Nfft = Xr.shape[-1]; N2 = Nfft // 2
    pos = Xr[..., :N2]
    neg = Xr[..., N2:]
    Xr = neg[..., ::-1] if (np.abs(neg)**2).sum() > (np.abs(pos)**2).sum() else pos
    Nr_half = Xr.shape[-1]

    # ---------- Doppler FFT ----------
    Nd_eff = Nd_total // max(Ntx, 1)
    dwin_key = ('wd', Nd_eff)
    if dwin_key not in _WIN_CACHE:
        _WIN_CACHE[dwin_key] = np.hanning(Nd_eff).astype(np.float32)
    wd = _WIN_CACHE[dwin_key]

    Xrd_virtual_list = []

    mti_order = MTI_CANCELLER
    q_noise = NOISE_SUBTRACT_Q
    frame_ewma = FRAME_EWMA

    for m in range(Ntx):
        Xr_sel = Xr[:, m:Nd_total:Ntx, m, :, :]
        if mti_order > 0:
            Xr_sel = _apply_mti(Xr_sel, mti_order)
        else:
            Xr_sel = Xr_sel - Xr_sel.mean(axis=1, keepdims=True)

        # Glitch repair
        E = np.sum(np.abs(Xr_sel) ** 2, axis=(2, 3)).astype(np.float32)
        med = np.median(E, axis=1, keepdims=True)
        mad = np.median(np.abs(E - med), axis=1, keepdims=True) + 1e-9
        bad = E > (med + 8.0 * mad)
        Ff, Nn = np.where(bad)
        for f, n in zip(Ff, Nn):
            if 0 < n < (Nd_eff - 1):
                Xr_sel[f, n, :, :] = 0.5 * (Xr_sel[f, n - 1, :, :] + Xr_sel[f, n + 1, :, :])
            elif n == 0 and Nd_eff > 1:
                Xr_sel[f, n, :, :] = Xr_sel[f, 1, :, :]
            elif n == (Nd_eff - 1) and Nd_eff > 1:
                Xr_sel[f, n, :, :] = Xr_sel[f, Nd_eff - 2, :, :]

        # Doppler window + FFT
        Xr_sel = Xr_sel * wd[None, :, None, None]
        Xd = np.fft.fft(Xr_sel, n=Nd_eff, axis=1)
        Xd = np.fft.fftshift(Xd, axes=1)
        Xd = np.transpose(Xd, (0, 3, 1, 2))
        Xrd_virtual_list.append(Xd)

    # ---------- Concatenate TXs (Virtual Array) ----------
    if not Xrd_virtual_list:
        raise ValueError("Xrd_virtual_list is empty; no valid TX channels processed.")
    Xrd_virtual = np.concatenate(Xrd_virtual_list, axis=3)
    Xrd_virtual = Xrd_virtual - Xrd_virtual.mean(axis=3, keepdims=True)
    Fnum, Rnum, Nd, V = Xrd_virtual.shape

    # ---------- Axes ----------
    ax_key = (id(ADC_PARAMS_l), Rnum, Nd, Ntx)
    if ax_key in _AXES_CACHE:
        rng_m, vel_mps = _AXES_CACHE[ax_key]
    else:
        rng_m, vel_mps = _compute_axes_from_cfg({**ADC_PARAMS_l}, Rnum, Nd, Ntx)
        _AXES_CACHE[ax_key] = (rng_m, vel_mps)
    meters_per_bin = float(np.abs(rng_m[1] - rng_m[0])) if Rnum > 1 else 0.1

    # ---------- Power cube ----------
    P_frd = (np.abs(Xrd_virtual) ** 2).sum(axis=3).astype(np.float32)
    P_frd_work = P_frd.copy()

    # Noise subtraction
    if 0.0 < q_noise < 1.0:
        nf_all = np.quantile(P_frd_work, q_noise, axis=1).astype(np.float32)
        P_frd_work = np.maximum(P_frd_work - nf_all[:, None, :], 0.0)

    # ---------- Range Energy ----------
    center = Nd // 2
    zero_w = max(3, Nd // 64)
    edge_w = max(center - 1, int(0.45 * Nd))
    d_idx = np.arange(Nd)
    dop_mask = (np.abs(d_idx - center) >= zero_w) & (np.abs(d_idx - center) <= edge_w)
    E_fr = P_frd_work[:, :, dop_mask].sum(axis=2)

    # Smooth range energy
    if E_fr.size:
        kern = np.ones(5, dtype=np.float32) / 5.0
        E_fr = np.apply_along_axis(lambda x: np.convolve(x, kern, mode='same'), 1, E_fr)

    # ---------- Range Center Tracking ----------
    if E_fr.shape[0] > 0:
        r_center = np.zeros(Fnum, dtype=np.int32)
        r_center[0] = int(np.argmax(E_fr[0]))
        alpha = 0.8
        for fidx in range(1, Fnum):
            r_raw = int(np.argmax(E_fr[fidx]))
            r_center[fidx] = int(round(alpha * r_center[fidx - 1] + (1.0 - alpha) * r_raw))
    else:
        r_center = np.zeros(Fnum, dtype=np.int32)

    # ---------- Gaussian Range Weights ----------
    dR = float(np.abs(rng_m[1] - rng_m[0])) if Rnum > 1 else 0.1
    sigma_m = 0.8
    sigma_bins = max(2.0, sigma_m / max(dR, 1e-6))
    r_grid = np.arange(Rnum, dtype=np.float32)
    W = np.exp(-0.5 * ((r_grid[None, :] - r_center[:, None]) / sigma_bins) ** 2).astype(np.float32)
    W /= (W.sum(axis=1, keepdims=True) + EPS)

    # ---------- μD Columns ----------
    power_adapt = (P_frd_work * W[:, :, None]).sum(axis=1).astype(np.float32)
    if DOPPLER_SMOOTH_K > 1:
        k = int(max(1, DOPPLER_SMOOTH_K | 1))
        ker = np.ones(k, dtype=np.float32) / float(k)
        power_adapt = np.apply_along_axis(lambda r: np.convolve(r, ker, mode='same'), 1, power_adapt)

    if 0.0 < frame_ewma < 1.0 and power_adapt.shape[0] > 1:
        for fidx in range(1, power_adapt.shape[0]):
            power_adapt[fidx] = frame_ewma * power_adapt[fidx - 1] + (1.0 - frame_ewma) * power_adapt[fidx]

    # ---------- Publish ----------
    if _ZMQ_PUB is not None:
        try:
            md_db_all = 10.0 * np.log10(np.maximum(power_adapt, 1e-12)).astype(np.float32)
            for fidx in range(power_adapt.shape[0]):
                ts = float(time.time())
                payload = {
                    "ts": ts,
                    "md_db": md_db_all[fidx].astype(np.float32),
                    "range": E_fr[fidx].astype(np.float32) if E_fr.size else np.zeros((Rnum,), np.float32),
                }
                if not (DEMO_MODE and DEMO_PUB_THIN > 1 and (fidx % DEMO_PUB_THIN) != 0):
                    _ZMQ_PUB.send_pyobj(payload, flags=0)
        except Exception:
            pass

    return E_fr.astype(np.float32), power_adapt.astype(np.float32)


class MDViewer(QtWidgets.QMainWindow):
    """
    Micro-Doppler viewer with a small HUD:
      - Top line: fps
      - Second line: WiGait metrics text (from radar), e.g.
        "WiGait: v=1.12 m/s f=1.02 Hz cad=122.4 spm SL=0.71 m"
    """
    def __init__(self, Nd_eff, hist=512):
        super().__init__()
        pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder='row-major')
        self.setWindowTitle("Micro-Doppler (live)")
        self.Nd = int(Nd_eff)
        self.hist = int(hist)

        # Image buffer (Nd x hist), newest column on the right
        self.img_arr = np.zeros((self.Nd, self.hist), dtype=np.float32)

        # Plot setup
        w = pg.GraphicsLayoutWidget()
        self.setCentralWidget(w)
        p = w.addPlot()
        p.setLabel('bottom', 'Frame')
        p.setLabel('left',   'Doppler bin')
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
        p.getAxis('bottom').enableAutoSIPrefix(False)
        p.getAxis('left').enableAutoSIPrefix(False)

        # ---- HUD overlays (match depth viewer style) ----
        # Approximate depth text size: use a small 10-pt font.
        hud_font = QtGui.QFont()
        hud_font.setPointSize(8)

        # FPS: green, top-left
        self._hud_fps = pg.TextItem(color=(0, 255, 0))
        self._hud_fps.setFont(hud_font)
        # y ~ 8 is near the top since Y is inverted
        self._hud_fps.setPos(2, -200)
        p.addItem(self._hud_fps)

        # WiGait metrics: same color as depth WiGait overlay.
        # Depth uses BGR (0, 200, 255); in RGB that is (255, 200, 0).
        self._hud_gait = pg.TextItem(color=(0, 200, 255))
        self._hud_gait.setFont(hud_font)
        # Place just below FPS line
        self._hud_gait.setPos(2, -6)
        p.addItem(self._hud_gait)

        # State
        self._fps_ema = None
        self._gait_text = ""

        self._refresh()

    def append_col(self, col_db: np.ndarray):
        """Append one μD column (in dB) to the scrolling image."""
        col_db = np.asarray(col_db, dtype=np.float32)
        if col_db.shape[0] != self.Nd:
            # caller must recreate MDViewer for Nd change
            return

        # Shift left and add newest column on the right
        self.img_arr = np.roll(self.img_arr, -1, axis=1)
        self.img_arr[:, -1] = col_db

        # FPS HUD (column rate)
        now = time.time()
        if not hasattr(self, "_last_data_time"):
            self._last_data_time = now
        dt = now - self._last_data_time
        self._last_data_time = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)
        self._hud_fps.setText(f"FPS≈{self._fps_ema:.1f}")

        # Keep the gait text where it is
        if self._gait_text:
            self._hud_gait.setText(self._gait_text)

        self._refresh()

    def set_gait_text(self, text: str):
        """
        Set the WiGait metrics text (exactly the depth format).
        Example: "WiGait: v=1.12 m/s f=1.02 Hz cad=122.4 spm SL=0.71 m"
        """
        self._gait_text = str(text) if text else ""
        self._hud_gait.setText(self._gait_text)

    def set_gait_metrics(self, mean_speed_mps: float, stride_freq_hz: float,
                         cadence_spm: float, mean_stride_len_m: float):
        """
        Convenience method to format and set the gait text in the depth-style format.
        """
        try:
            v = float(mean_speed_mps)
            f = float(stride_freq_hz)
            cad = float(cadence_spm)
            sl = float(mean_stride_len_m)
        except Exception:
            self.set_gait_text("")
            return

        txt = (f"Gait Metrics : v = {v:4.2f} m/s f = {f:4.2f} Hz "
               f"cad = {cad:5.1f} spm Step Length = {sl:4.2f} m")
        self.set_gait_text(txt)

    def _refresh(self):
        self.image.setImage(self.img_arr, autoLevels=True, autoRange=False)
        vb = self.image.getViewBox()
        if vb is not None:
            vb.setRange(self.image.boundingRect(), padding=0.0)


# -----------------------------------------------------------------------------------
try:
    dca = DCA1000()
    dca.reset_radar()
    dca.reset_fpga()
    print("wait for reset")
    time.sleep(1)

    dca_config_file = "../modular/configFiles/cf.json"
    radar_config_file = "../modular/configFiles/xWR6843_profile_3D.cfg"

    radar = TI(cli_loc='COM4', data_loc='COM3', data_baud=921600, config_file=radar_config_file, verbose=True)
    numLoops = 100000
    frameNumInBuf = 16
    numframes = 16  # must be <= frameNumInBuf
    radar.setFrameCfg(0)

    if DEMO_MODE:
        frameNumInBuf = int(DEMO_FRAMEBUF)
        numframes = int(DEMO_NUMFRAMES)

    ADC_PARAMS_l, _ = dca.configure(dca_config_file, radar_config_file)

    try:
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--session-dir", default=None)
        ap.add_argument("--auto-start", action="store_true")
        _args, _ = ap.parse_known_args()
        if _args.session_dir:
            SESSION_DIR = _args.session_dir
            base = os.path.basename(os.path.normpath(SESSION_DIR))
            SESSION_ID = base.split("radar_", 1)[1] if base.startswith("radar_") else base
        else:
            SESSION_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
            SESSION_DIR = os.path.join("../modular/sessions", f"radar_{SESSION_ID}")
    except Exception:
        SESSION_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
        SESSION_DIR = os.path.join("../modular/sessions", f"radar_{SESSION_ID}")
    os.makedirs(SESSION_DIR, exist_ok=True)
    meta = {
        "session_id": SESSION_ID,
        "start_time_iso": datetime.now().isoformat(),
        "dca_config_file": dca_config_file,
        "radar_config_file": radar_config_file,
        "adc_params": ADC_PARAMS_l
    }
    with open(os.path.join(SESSION_DIR, "session_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if not _args.auto_start:
        input("press ENTER to start capture...")
    else:
        print("[auto] starting capture without ENTER")
    dca.stream_start()
    dca.fastRead_in_Cpp_thread_start(frameNumInBuf)
    radar.startSensor()

    Nd_total = int(ADC_PARAMS_l['chirps'])
    Ntx = int(ADC_PARAMS_l['tx'])
    Nd_eff = Nd_total // max(Ntx, 1)

    q_max = DEMO_QUEUE_MAXSIZE if DEMO_MODE else 2048
    q = queue.Queue(maxsize=q_max)
    _live_floor = None

    try:
        ts_csv_path = os.path.join(SESSION_DIR, "radar_md_timestamps.csv")
        write_header = (not os.path.exists(ts_csv_path)) or (os.path.getsize(ts_csv_path) == 0)
        _md_time_fp = open(ts_csv_path, "a", newline="")
        _md_time_csv = csv.writer(_md_time_fp)
        if write_header:
            _md_time_csv.writerow(["t","loop_idx","col_idx"])
        print(f"[timestamp] Logging μD column timestamps to: {ts_csv_path}")
    except Exception as _e:
        print(f"[timestamp] WARNING: could not open timestamp CSV: {_e}")

    def _live_soft_floor(col_lin: np.ndarray) -> np.ndarray:
        global _live_floor
        if not LIVE_SOFT_FLOOR:
            return col_lin
        if _live_floor is None:
            _live_floor = col_lin.copy()
        else:
            _live_floor = LIVE_BASELINE_TAU * _live_floor + (1.0 - LIVE_BASELINE_TAU) * col_lin
        out = col_lin - _live_floor
        out[out < 0.0] = 0.0
        return out

    def producer():
        try:
            for loop_idx in range(numLoops):
                data_buf = dca.fastRead_in_Cpp_thread_get(numframes, verbose=True, sortInC=True)

                if not DEMO_MODE:
                    try:
                        np.save(os.path.join(SESSION_DIR, f"adc_loop{loop_idx:06d}.npy"), data_buf)
                    except Exception as _e:
                        print(f"[save] WARNING: raw ADC save failed: {_e}")

                _, md_cols = postProc(data_buf, ADC_PARAMS_l)
                _final_md_chunks.append(md_cols.copy())

                if not DEMO_MODE:
                    try:
                        np.save(os.path.join(SESSION_DIR, f"md_cols_loop{loop_idx:06d}.npy"), md_cols)
                    except Exception as _e:
                        print(f"[save] WARNING: md_cols save failed: {_e}")

                batch = md_cols.reshape(-1)
                bmin, bmax, bmean = float(np.min(batch)), float(np.max(batch)), float(np.mean(batch))
                bnnz = int(np.count_nonzero(batch))
                print(f"[producer] batch frames={md_cols.shape[0]} bins={md_cols.shape[1]} "
                      f"min={bmin:.3e} max={bmax:.3e} mean={bmean:.3e} nnz={bnnz}")

                for k in range(md_cols.shape[0]):
                    ts = time.time()
                    col_lin = md_cols[k].astype(np.float32)
                    col_lin = np.nan_to_num(col_lin, nan=0.0, posinf=0.0, neginf=0.0)
                    col_lin = _live_soft_floor(col_lin)
                    col_db = 10.0 * np.log10(col_lin + 1e-12).astype(np.float32)

                    if (k % max(1, md_cols.shape[0] // 4)) == 0:
                        print(f"[producer] col stats: min={float(col_db.min()):.2f} "
                              f"p5={float(np.percentile(col_db,5)):.2f} "
                              f"p95={float(np.percentile(col_db,95)):.2f} "
                              f"max={float(col_db.max()):.2f}")

                    try:
                        q.put_nowait(col_db)
                    except queue.Full:
                        print("[producer] WARNING: queue full, dropping column")

                    try:
                        if _md_time_csv is not None:
                            _md_time_csv.writerow([f"{ts:.6f}", int(loop_idx), int(k)])
                    except Exception:
                        pass

                    try:
                        adc_csv_path = os.path.join(SESSION_DIR, "adc_loop_timestamps.csv")
                        write_header = not os.path.exists(adc_csv_path)
                        with open(adc_csv_path, "a", newline="") as f_adc:
                            writer = csv.writer(f_adc)
                            if write_header:
                                writer.writerow(["t", "loop_idx"])
                            writer.writerow([f"{time.time():.6f}", loop_idx])
                    except Exception as _e:
                        print(f"[timestamp] WARNING: could not log adc loop timestamp: {_e}")

        finally:
            q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            break

    if PLOT_FINAL_AFTER_RUN:
        try:
            # existing quick-look (frame-index x-axis)
            # render_final_md(_final_md_chunks, ADC_PARAMS_l)

            # new, time-aligned spectrogram saved to disk only
            save_final_md_timealigned(_final_md_chunks, SESSION_DIR, ADC_PARAMS_l,
                                      fname="micro_doppler_time.png")
        except Exception:
            traceback.print_exc()


except Exception as e:
    traceback.print_exc()
finally:
    if _md_time_fp is not None:
        try:
            _md_time_fp.flush()
            _md_time_fp.close()
        except Exception:
            pass
    if radar is not None:
        radar.stopSensor()
    if dca is not None:
        dca.fastRead_in_Cpp_thread_stop()
        dca.stream_stop()
        dca.close()
