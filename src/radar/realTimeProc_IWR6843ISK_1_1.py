import traceback
import time
from mmwave.dataloader import DCA1000
from mmwave.dataloader.radars import TI
import numpy as np
import datetime
# Smart Sensing and Computing for Dementia Care
# The goal of this project is to develop sensing
# systems to collect data regarding risk assessment and prevention.

# --- NEW: imports for producer/consumer + fast live plotting ---
import queue, threading
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from collections import deque  # NEW
import math  # NEW

'''
# AWR2243采集原始数据的一般流程
...
"packetDelay_us":  5 (us)   ~   706 (Mbps)
"packetDelay_us": 10 (us)   ~   545 (Mbps)
"packetDelay_us": 25 (us)   ~   325 (Mbps)
"packetDelay_us": 50 (us)   ~   193 (Mbps)
'''
dca = None
radar = None

# ---- OFFLINE PLOT (like draft.py) after the live run ----
PLOT_FINAL_AFTER_RUN = True   # set False to disable
SAVE_FINAL_PNG = False
SAVE_FINAL_NPY  = False       # also save the linear-power matrix if you want

_final_md_chunks = []         # we’ll append md_cols (linear power) batches here

# ---- LIVE soft baseline (viewer shaping; no hard notch) ----
LIVE_SOFT_FLOOR = False
LIVE_BASELINE_TAU = 0.995     # 0.99–0.999; higher = slower background update

# ------------------------- tunables for robustness -------------------------
CAL_DURATION_S = 5.0       # warm-up time before freezing color limits
CAL_MIN_COLS = 48          # or at least this many columns before freeze

# ------------------------- SNR-improvers (new tunables) -------------------------
MTI_CANCELLER = 0      # 0: off (use mean-subtract), 1: 2-pulse, 2: 3-pulse (recommended)
ZERO_NOTCH_BINS = 0     # notch ±N Doppler bins around 0 to kill residual static
NOISE_SUBTRACT_Q = 0.10  # robust per-Doppler noise floor (quantile across range), 0..1
DOPPLER_SMOOTH_K = 1   # moving-average across Doppler bins (odd int; 1 disables)
FRAME_EWMA = 0.00        # EWMA across frames inside this batch for the md column


# NEW: thesis-appendix processing controls
CLUTTER_WIDTH_BINS = 2         # corresponds to 'cl_width' in appendix code
GHOST_SUPPRESS = False          # zero a small box around detected ghost before MD collapse
GHOST_BOX_R = 2                # +/- rows (range) around ghost idx
GHOST_BOX_D = 3                # +/- cols (doppler) around ghost idx
LOG_WALL_EST_EVERY = 30        # print a wall-distance estimate every N frames (None to disable)
EPS = 1e-12

# ------------------------- helpers -------------------------
def _hann(n: int) -> np.ndarray:
    return np.hanning(int(n)).astype(np.float32)

def _moving_avg_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k)
    k = max(1, k | 1)  # odd
    pad = k // 2
    xpad = np.pad(x, ((0, 0), (pad, pad)), mode='edge')
    ker = np.ones((1, k), dtype=np.float32) / float(k)
    return np.apply_along_axis(lambda r: np.convolve(r, ker.flatten(), mode='valid'), 1, xpad)

def _compute_axes_from_cfg(params: dict, Nr_half: int, Nd_eff: int, Ntx_hint: int):
    """
    Compute range (m) and velocity (m/s) axes from TI cfg-style params.
    Robust to common unit/key slips (e.g., 7.7->77 GHz, rampEndTime parsed as 7 instead of 78).
    Uses chirp timing first; falls back to frame timing if the result is implausible.
    """
    import numpy as np
    c = 299792458.0

    # Defaults if anything goes wrong
    rng_m = np.arange(Nr_half, dtype=np.float32)
    vel_mps = (np.arange(Nd_eff, dtype=np.float32) - (Nd_eff // 2)).astype(np.float32)

    try:
        # ---- pull fields (tolerant to different key names) ----
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

        # Frame-derived fields (for fallback / cross-check)
        frame_ms         = (params.get('framePeriodicity_ms') or params.get('framePeriodicity') or
                            params.get('frame_periodicity_ms') or params.get('frame_periodicity'))
        chirp_start_idx  = params.get('chirpStartIdx')
        chirp_end_idx    = params.get('chirpEndIdx')
        num_loops        = params.get('numLoops')
        Ntx_cfg          = params.get('tx')

        # ---- basic units / derived constants ----
        slope_Hz_per_s = float(slope_mhz_per_us) * 1e12                  # (MHz/µs) → Hz/s
        f0_GHz = float(start_freq_ghz)
        if f0_GHz < 10.0:                                                # guard 7.7 → 77
            f0_GHz *= 10.0
        lam = c / (f0_GHz * 1e9)                                         # wavelength (m)

        Tr = float(ramp_end_time_us) * 1e-6                              # s
        Ti = float(idle_time_us) * 1e-6 if idle_time_us is not None else 0.0
        Ntx_eff = int(Ntx_hint) if int(Ntx_hint) > 0 else int(Ntx_cfg or 1)

        # ---- RANGE axis from instantaneous bandwidth ----
        B  = slope_Hz_per_s * Tr                                         # Hz
        dR = c / (2.0 * B)                                               # m
        rng_m = (dR * np.arange(Nr_half, dtype=np.float32)).astype(np.float32)

        # ---- DOPPLER axis (two candidates) ----
        # A) from chirp timing (correct for PRF unless cfg was mis-parsed)
        TD_chirp = Ntx_eff * (Ti + Tr)                                   # s (time between same-TX chirps)
        # B) fallback from frame timing (robust if ramp/idle are wrong)
        TD_frame = None
        if (frame_ms is not None) and (chirp_start_idx is not None) and (chirp_end_idx is not None) and (num_loops is not None):
            T_frame_s = float(frame_ms) * 1e-3
            chirps_per_loop = int(chirp_end_idx) - int(chirp_start_idx) + 1
            Nd_total = int(num_loops) * chirps_per_loop                  # total chirps (all TX) per frame
            T_chirp_est = T_frame_s / max(Nd_total, 1)                   # s per chirp (any TX)
            TD_frame = Ntx_eff * T_chirp_est                              # s between same-TX chirps

        # Choose the TD that yields a human-credible vmax (favor chirp timing when sane)
        def vmax_from_TD(TD): return lam / (4.0 * TD)
        TD = TD_chirp
        # If chirp-derived TD is clearly too small/large (vmax outside ~1..15 m/s), prefer frame-derived
        if TD_frame is not None:
            vA = vmax_from_TD(TD_chirp)
            vB = vmax_from_TD(TD_frame)
            if not (1.0 <= vA <= 15.0):
                TD = TD_frame
            else:
                # both plausible → pick the one nearer to ~5 m/s pedestrian band
                TD = TD_chirp if abs(vA - 5.0) <= abs(vB - 5.0) else TD_frame

        # Final Doppler/velocity axis
        fd = np.fft.fftshift(np.fft.fftfreq(Nd_eff, d=TD)).astype(np.float32)
        vel_mps = (fd * lam / 2.0).astype(np.float32)

        # Optional: one-line debug so you can verify once and forget
        vmax = float(max(abs(vel_mps[0]), abs(vel_mps[-1])))
        print(f"[axes] f0={f0_GHz:.1f} GHz  λ={lam:.4f} m  "
              f"Tchirp={((Ti+Tr)*1e6):.2f} µs  TD={TD*1e6:.2f} µs  "
              f"Ntx={Ntx_eff}  vmax≈{vmax:.2f} m/s  dR≈{dR:.3f} m")

    except Exception:
        # fall back to indices if anything fails
        rng_m = np.arange(Nr_half, dtype=np.float32)
        vel_mps = (np.arange(Nd_eff, dtype=np.float32) - (Nd_eff // 2)).astype(np.float32)

    return rng_m, vel_mps

# ------------------------- Appendix A methods (adapted) -------------------------
def angle_process_rdvec(rd_vec_v, n_fft=128):
    ang_fft = np.fft.fft(rd_vec_v, n=n_fft)
    ang_fft = np.fft.fftshift(ang_fft)
    angle_peak_idx = int(np.argmax(np.abs(ang_fft)))
    s = 2.0 * (angle_peak_idx - (ang_fft.shape[0] // 2)) / float(ang_fft.shape[0])
    s = np.clip(s, -1.0, 1.0)
    angle = np.rad2deg(np.arcsin(s))
    return ang_fft, float(angle)

def find_peak(range_velocity_db, clutter_width_bins, meters_per_bin):
    rv_mod = range_velocity_db.copy()
    nv = rv_mod.shape[1]
    c = nv // 2
    w = int(clutter_width_bins)
    w = max(0, w)
    rv_mod[:, max(0, c-w):min(nv, c+w)] = -np.inf
    idx = np.unravel_index(np.nanargmax(rv_mod), rv_mod.shape)
    power_db = rv_mod[idx]
    distance_m = float(idx[0]) * float(meters_per_bin)
    return rv_mod, float(power_db), distance_m, idx

def find_second_peak(range_velocity_db, main_idx, meters_per_bin):
    rv_mod = range_velocity_db.copy()
    r, d = main_idx
    rv_mod[max(0, r-2):r+3, max(0, d-3):d+4] = -np.inf
    rv_mod[:r+1, :] = -np.inf
    rv_mod[:, d:] = -np.inf
    idx2 = np.unravel_index(np.nanargmax(rv_mod), rv_mod.shape)
    distance_ghost_m = float(idx2[0]) * float(meters_per_bin)
    power2_db = rv_mod[idx2]
    return idx2, distance_ghost_m, float(power2_db)

def inverse_wall_non_radial(d_direct_idx, v_real_idx, d_ghost_idx, v_ghost_idx,
                            theta_ghost_deg, theta_direct_deg=0.0, meters_per_bin=0.3125):
    theta_rad = math.radians(theta_direct_deg)
    theta_ghost_rad = math.radians(theta_ghost_deg)
    b = 4.0 * d_direct_idx * math.sin(theta_rad)
    d_wall_d = meters_per_bin/8.0 * (b + math.sqrt(max(b*b - 16.0*(d_direct_idx**2 - d_ghost_idx**2), 0.0)))
    arg = math.acos(max(min(v_ghost_idx/float(v_real_idx + (1e-9)), 1.0), -1.0)) - theta_rad
    d_wall_v = meters_per_bin/2.0 * d_direct_idx * (math.cos(theta_rad)*math.tan(arg) + math.sin(theta_rad))
    d_wall_angle = meters_per_bin/2.0 * d_direct_idx * (math.cos(theta_rad)*math.tan(theta_ghost_rad) + math.sin(theta_rad))
    return float(d_wall_d), float(d_wall_v), float(d_wall_angle)

def find_ghost_AoA_non_radial(d_direct_m, d_wall_m=2.5, theta_direct_deg=0.0):
    theta = math.radians(theta_direct_deg)
    arg = (2.0*d_wall_m - d_direct_m*math.sin(theta)) / max(d_direct_m*math.cos(theta), 1e-9)
    angle_deg = math.degrees(math.atan(arg))
    return float(angle_deg)

def _safe_log10(x):
    return np.log10(np.maximum(x, EPS)).astype(np.float32)

def render_final_md(md_chunks, adc_params, title="Micro-Doppler (full recording)"):
    if not md_chunks:
        print("[final] No columns buffered; skipping.")
        return

    import matplotlib.pyplot as plt
    md = np.concatenate(md_chunks, axis=0)           # [F_total, Nd]
    Nd_eff = md.shape[1]

    Nr_half_guess = max(1, int(adc_params.get('samples', 0)) // 2)
    Ntx_guess = int(adc_params.get('tx') or adc_params.get('num_tx_like') or 1)
    _, vel_mps = _compute_axes_from_cfg(adc_params, Nr_half_guess, Nd_eff, Ntx_guess)

    md_db = 20.0 * np.log10(np.maximum(md, 1e-12)).astype(np.float32)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4))
    plt.imshow(
        md_db.T, origin='lower', aspect='auto', cmap='jet',
        extent=[0, md_db.shape[0]-1, float(vel_mps[0]), float(vel_mps[-1])]
    )
    plt.title(title)
    plt.xlabel('Frame'); plt.ylabel('Radial velocity (m/s)')
    plt.tight_layout()

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if SAVE_FINAL_PNG:
        out_png = f"micro_doppler_full_{ts}.png"
        plt.savefig(out_png, dpi=150)
        print(f"[final] saved: {out_png}  shape={md_db.shape}")
    if SAVE_FINAL_NPY:
        out_npy = f"micro_doppler_full_{ts}.npy"
        np.save(out_npy, md)
        print(f"[final] saved: {out_npy}")

    plt.show()

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
    Micro-Doppler processing pipeline with optional Appendix-A steps BEFORE range-weighted collapse.
    Returns:
      range_doppler (None placeholder), md_cols (F, Nd_eff) power (linear)
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
    wr = np.hanning(Nr).astype(np.float32)[None, None, None, None, :]
    Xr = np.fft.fft(adc_cplx * wr, n=Nr, axis=-1)

    # 2) Keep the beat-frequency half with higher energy
    Nfft = Xr.shape[-1]; N2 = Nfft // 2
    pos = Xr[..., :N2]
    neg = Xr[..., N2:]
    if np.sum(np.abs(neg)**2) > np.sum(np.abs(pos)**2):
        Xr = neg[..., ::-1]
    else:
        Xr = pos
    Nr_half = Xr.shape[-1]

    # 3) Doppler per-TX (TDM)
    Nd_eff = Nd_total // max(Ntx, 1)
    wd = np.hanning(Nd_eff).astype(np.float32)

    Xrd_virtual_list = []

    for m in range(Ntx):
        # select chirps for TX m: [F, Nd_eff, Nrx, Nr_half]
        Xr_sel = Xr[:, m:Nd_total:Ntx, m, :, :]

        # MTI or mean-subtraction
        if MTI_CANCELLER > 0:
            Xr_sel = _apply_mti(Xr_sel, MTI_CANCELLER)
        else:
            Xr_sel = Xr_sel - Xr_sel.mean(axis=1, keepdims=True)

        # --- Glitch guard: detect/repair bad chirps (impulses) in slow-time ---
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

        # Doppler window + FFT + shift
        Xr_sel = Xr_sel * wd[None, :, None, None]
        Xd = np.fft.fft(Xr_sel, n=Nd_eff, axis=1)
        Xd = np.fft.fftshift(Xd, axes=1)

        # Reorder to [F, R, Nd_eff, Nrx]
        Xd = np.transpose(Xd, (0, 3, 1, 2))
        Xrd_virtual_list.append(Xd)

    # Concatenate TX across channels -> virtual array V=TX*RX
    Xrd_virtual = np.concatenate(Xrd_virtual_list, axis=3)  # [F, R, Nd, V]

    # Spatial common-mode (mutual-coupling) cancellation across V
    Xrd_virtual = Xrd_virtual - Xrd_virtual.mean(axis=3, keepdims=True)

    Fnum, Rnum, Nd, V = Xrd_virtual.shape

    # Appendix-A steps (optional, leave as-is)
    rng_m, vel_mps = _compute_axes_from_cfg({**ADC_PARAMS_l}, Rnum, Nd, Ntx)
    meters_per_bin = float(np.abs(rng_m[1] - rng_m[0])) if Rnum > 1 else 0.1

    # Power cube
    P_frd = (np.abs(Xrd_virtual) ** 2).sum(axis=3).astype(np.float32)  # [F, R, Nd]
    P_frd_work = P_frd.copy()

    # (duplicate power lines kept intentionally minimal—no functional change)
    P_frd = (np.abs(Xrd_virtual) ** 2).sum(axis=3).astype(np.float32)
    P_frd_work = P_frd.copy()

    if 0.0 < NOISE_SUBTRACT_Q < 1.0:
        for fidx in range(P_frd_work.shape[0]):
            nf = np.quantile(P_frd_work[fidx], NOISE_SUBTRACT_Q, axis=0).astype(np.float32)
            P_frd_work[fidx] = np.maximum(P_frd_work[fidx] - nf[None, :], 0.0)

    for fidx in range(Fnum):
        RD_amp = np.sqrt(np.maximum(P_frd_work[fidx], 0.0) + EPS)
        RD_db = 20.0 * _safe_log10(RD_amp)
        cwidth = max(CLUTTER_WIDTH_BINS, Nd // 64)
        rv_mod, power_db, distance_m, idx = find_peak(RD_db, cwidth, meters_per_bin)
        idx2, distance_ghost_m, power2_db = find_second_peak(rv_mod, idx, meters_per_bin)

        r0, d0 = int(idx[0]), int(idx[1])
        r1, d1 = int(idx2[0]), int(idx2[1])

        try:
            _, angle_direct = angle_process_rdvec(Xrd_virtual[fidx, r0, d0, :], n=128)
        except Exception:
            angle_direct = 0.0
        try:
            _, angle_ghost = angle_process_rdvec(Xrd_virtual[fidx, r1, d1, :], n=128)
        except Exception:
            angle_ghost = 0.0

        d_wall_d, d_wall_v, d_wall_angle = inverse_wall_non_radial(
            d_direct_idx=r0, v_real_idx=max(d0, 1),
            d_ghost_idx=r1,  v_ghost_idx=max(d1, 1),
            theta_ghost_deg=angle_ghost, theta_direct_deg=angle_direct,
            meters_per_bin=meters_per_bin
        )

        if LOG_WALL_EST_EVERY and (fidx % LOG_WALL_EST_EVERY == 0):
            print(f"[appendix] f={fidx} main@({r0},{d0}) ghost@({r1},{d1}) "
                  f"AoA(main,ghost)=({angle_direct:.1f},{angle_ghost:.1f}) "
                  f"a_est(m): d={d_wall_d:.2f} v={d_wall_v:.2f} ang={d_wall_angle:.2f}")

        if GHOST_SUPPRESS:
            r_lo = max(0, r1 - GHOST_BOX_R); r_hi = min(Rnum, r1 + GHOST_BOX_R + 1)
            d_lo = max(0, d1 - GHOST_BOX_D); d_hi = min(Nd,   d1 + GHOST_BOX_D + 1)
            P_frd_work[fidx, r_lo:r_hi, d_lo:d_hi] = 0.0

    # 4) Energy vs range using Doppler mask
    center = Nd // 2
    zero_w = max(3, Nd // 64)
    edge_w = max(center - 1, int(0.45 * Nd))
    d_idx = np.arange(Nd)
    dop_mask = (np.abs(d_idx - center) >= zero_w) & (np.abs(d_idx - center) <= edge_w)

    E_fr = P_frd_work[:, :, dop_mask].sum(axis=2)

    # Smooth across range (5-bin moving average)
    kern = np.ones(5, dtype=np.float32) / 5.0
    E_fr = np.apply_along_axis(lambda x: np.convolve(x, kern, mode='same'), 1, E_fr)

    # EWMA range center
    r_center = np.zeros(Fnum, dtype=np.int32)
    r_center[0] = int(np.argmax(E_fr[0]))
    alpha = 0.8
    for fidx in range(1, Fnum):
        r_raw = int(np.argmax(E_fr[fidx]))
        r_center[fidx] = int(round(alpha * r_center[fidx - 1] + (1.0 - alpha) * r_raw))

    # 5) Gaussian range weights
    dR = float(np.abs(rng_m[1] - rng_m[0])) if Rnum > 1 else 0.1
    sigma_m = 0.8
    sigma_bins = max(2.0, sigma_m / max(dR, 1e-6))
    r_grid = np.arange(Rnum, dtype=np.float32)
    W = np.exp(-0.5 * ((r_grid[None, :] - r_center[:, None]) / sigma_bins) ** 2).astype(np.float32)
    W /= (W.sum(axis=1, keepdims=True) + EPS)

    # 6) Collapse range with weights -> micro-Doppler columns
    power_adapt = (P_frd_work * W[:, :, None]).sum(axis=1).astype(np.float32)  # [F, Nd]
    power_adapt = (P_frd_work * W[:, :, None]).sum(axis=1).astype(np.float32)  # duplicate kept

    if DOPPLER_SMOOTH_K > 1:
        k = int(max(1, DOPPLER_SMOOTH_K | 1))
        ker = np.ones(k, dtype=np.float32) / float(k)
        for fidx in range(power_adapt.shape[0]):
            power_adapt[fidx] = np.convolve(power_adapt[fidx], ker, mode='same')

    if 0.0 < FRAME_EWMA < 1.0 and power_adapt.shape[0] > 1:
        for fidx in range(1, power_adapt.shape[0]):
            power_adapt[fidx] = FRAME_EWMA * power_adapt[fidx - 1] + (1.0 - FRAME_EWMA) * power_adapt[fidx]

    range_doppler = None
    md_cols = power_adapt
    return range_doppler, md_cols

    range_doppler = None
    md_cols = power_adapt
    return range_doppler, md_cols

class MDViewer(QtWidgets.QMainWindow):
    """Fast rolling spectrogram viewer (pyqtgraph) — no level freezing, always autoscale."""
    def __init__(self, Nd_eff, hist=512):
        super().__init__()
        # Disable OpenGL to avoid blank-image issues on some Windows drivers
        pg.setConfigOptions(antialias=False, useOpenGL=False, imageAxisOrder='row-major')
        self.setWindowTitle("Micro-Doppler (live)")
        self.Nd = Nd_eff
        self.hist = hist

        # Image buffer (we'll autoscale every draw; levels are not frozen)
        self.img_arr = np.zeros((Nd_eff, hist), dtype=np.float32)

        w = pg.GraphicsLayoutWidget()
        self.setCentralWidget(w)
        p = w.addPlot()
        p.setLabel('bottom', 'Frame')
        p.setLabel('left', 'Doppler bin')

        self.image = pg.ImageItem(self.img_arr)
        lut = pg.colormap.get('viridis').getLookupTable(0.0, 1.0, 256)
        self.image.setLookupTable(lut)
        p.addItem(self.image)

        # Map image to (x: 0..hist, y: 0..Nd)
        self.image.setRect(QtCore.QRectF(0, 0, self.hist, self.Nd))

        # ViewBox
        vb = p.getViewBox()
        vb.setAspectLocked(False)
        vb.invertY(True)
        vb.disableAutoRange()
        vb.setRange(self.image.boundingRect(), padding=0.0)
        p.setLimits(xMin=0, xMax=self.hist, yMin=0, yMax=self.Nd)
        p.getAxis('bottom').enableAutoSIPrefix(False)
        p.getAxis('left').enableAutoSIPrefix(False)

        # Simple FPS HUD (levels freeze removed)
        self._hud = pg.TextItem(color=(200, 200, 200))
        self._hud.setPos(2, max(1, self.Nd - 6))
        p.addItem(self._hud)
        self._fps_ema = None

        self._refresh()

    def append_col(self, col_db: np.ndarray):
        # shift and append
        self.img_arr = np.roll(self.img_arr, -1, axis=1)
        self.img_arr[:, -1] = col_db

        # FPS
        now = time.time()
        if not hasattr(self, "_last_data_time"):
            self._last_data_time = now
        dt = now - self._last_data_time
        self._last_data_time = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_ema = inst_fps if self._fps_ema is None else (0.9*self._fps_ema + 0.1*inst_fps)

        self._hud.setText(f"fps≈{self._fps_ema:.1f}")

        self._refresh()

    def _refresh(self):
        # Autoscale levels every draw; no fixed levels
        self.image.setImage(self.img_arr, autoLevels=True, autoRange=False)
        vb = self.image.getViewBox()
        if vb is not None:
            vb.setRange(self.image.boundingRect(), padding=0.0)
# -----------------------------------------------------------------------------------

try:
    dca = DCA1000()

    # 1. 重置雷达与DCA1000
    dca.reset_radar()
    dca.reset_fpga()
    print("wait for reset")
    time.sleep(1)

    # 2. 通过UART初始化雷达并配置相应参数
    dca_config_file = "configFiles/cf.json"  # 记得将cf.json中的lvdsMode设为2，xWR1843只支持2路LVDS lanes
    radar_config_file = "configFiles/xWR1843_profile_3D.cfg"  # 记得将lvdsStreamCfg的第三个参数设置为1开启LVDS数据传输
    # 记得改端口号,verbose=True会显示向毫米波雷达板子发送的所有串口指令及响应
    radar = TI(cli_loc='COM4', data_loc='COM3',data_baud=921600,config_file=radar_config_file,verbose=True)
    numLoops=100
    frameNumInBuf=16
    numframes=16 # numframes必须<=frameNumInBuf
    radar.setFrameCfg(0) # 0 for infinite frames

    # 3. 通过网口UDP发送配置FPGA指令
    # 4. 通过网口UDP发送配置record数据包指令
    '''
    dca.sys_alive_check()             # 检查FPGA是否连通正常工作
    dca.config_fpga(dca_config_file)  # 配置FPGA参数
    dca.config_record(dca_config_file)# 配置record参数
    '''
    ADC_PARAMS_l,_=dca.configure(dca_config_file,radar_config_file)  # 此函数完成上述所有操作

    # 按回车开始采集
    input("press ENTER to start capture...")

    # 5. 通过网口udp发送开始采集指令
    dca.stream_start()
    dca.fastRead_in_Cpp_thread_start(frameNumInBuf) # 启动udp采集线程，方法一（推荐）

    # 6. 通过UART启动雷达
    radar.startSensor()

    # -------------------- Producer (worker) + Consumer (Qt) --------------------
    Nd_total = int(ADC_PARAMS_l['chirps'])
    Ntx = int(ADC_PARAMS_l['tx'])
    Nd_eff = Nd_total // max(Ntx, 1)    # Doppler bins per single-TX stream
    q = queue.Queue(maxsize=2048)
    _live_floor = None


    def _live_soft_floor(col_lin: np.ndarray) -> np.ndarray:
        """Subtract a slowly-updating per-Doppler floor (EWMA)."""
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
                # Use the RD pipeline (with appendix steps) to get micro-Doppler columns
                _, md_cols = postProc(data_buf, ADC_PARAMS_l)     # md_cols: [F, Nd_eff]
                _final_md_chunks.append(md_cols.copy())  # keep linear-power columns

                # --- VERIFICATION LOGGING (batch) ---
                batch = md_cols.reshape(-1)
                bmin, bmax, bmean = float(np.min(batch)), float(np.max(batch)), float(np.mean(batch))
                bnnz = int(np.count_nonzero(batch))
                print(f"[producer] batch frames={md_cols.shape[0]} bins={md_cols.shape[1]} "
                      f"min={bmin:.3e} max={bmax:.3e} mean={bmean:.3e} nnz={bnnz}")

                for k in range(md_cols.shape[0]):
                    col_lin = md_cols[k].astype(np.float32)
                    col_lin = np.nan_to_num(col_lin, nan=0.0, posinf=0.0, neginf=0.0)
                    col_lin = _live_soft_floor(col_lin)

                    # power -> dB (no per-column normalization)
                    col_db = 10.0 * np.log10(col_lin + 1e-12).astype(np.float32)

                    # --- VERIFICATION LOGGING (per-column, sparse) ---
                    if (k % max(1, md_cols.shape[0] // 4)) == 0:
                        print(f"[producer] col stats: min={float(col_db.min()):.2f} "
                              f"p5={float(np.percentile(col_db,5)):.2f} "
                              f"p95={float(np.percentile(col_db,95)):.2f} "
                              f"max={float(col_db.max()):.2f}")

                    try:
                        q.put_nowait(col_db)
                    except queue.Full:
                        print("[producer] WARNING: queue full, dropping column")

        finally:
            q.put(None)  # signal consumer to exit

    # start producer thread
    t = threading.Thread(target=producer, daemon=True)
    t.start()

    # consumer / viewer on Qt main thread
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    viewer = MDViewer(Nd_eff=Nd_eff, hist=512)
    viewer.resize(900, 500)
    viewer.show()

    # module-level watchdog timestamp
    _last_gui_tick = time.time()

    def tick():
        global _last_gui_tick
        drained = False
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                timer.stop()
                viewer.close()
                QtWidgets.QApplication.quit()
                return
            viewer.append_col(item)
            drained = True
        # watchdog: warn if no data has arrived in a while
        now = time.time()
        if not drained and now - _last_gui_tick > 1.5:
            print("[viewer] WARNING: no new columns in >1.5 s (check capture/processing)")
            _last_gui_tick = now
        if drained:
            _last_gui_tick = now
            viewer.repaint()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(10)  # ~100 Hz UI polls

    app.exec()

    # Create an offline “full-period” micro-Doppler like draft.py
    if PLOT_FINAL_AFTER_RUN:
        try:
            render_final_md(_final_md_chunks, ADC_PARAMS_l)
        except Exception:
            traceback.print_exc()


except Exception as e:
    traceback.print_exc()
finally:
    if radar is not None:
        # 8. 通过UART停止雷达
        radar.stopSensor()
    if dca is not None:
        # 9. 通过网口udp发送停止采集指令
        dca.fastRead_in_Cpp_thread_stop() # 停止udp采集线程(必须先于stream_stop调用，即UDP接收时不能同时发送)
        dca.stream_stop()  # DCA停止采集
        dca.close()
q