import numpy as np
from matplotlib import pyplot as plt
from mmwave.dataloader import parse_raw_adc
from pathlib import Path
import gc  # to allow freeing memory promptly
from scipy.signal import medfilt  # for light "noise reduced" curve

# --- USER PATHS ---
source_fp = r"C:\\TIRadarADCData\\dca1000Data\\iqData_Raw_0.bin"
dest_fp   = r"C:\\TIRadarADCData\\dca1000Data\\iqData_Cooked_0.bin"
# Optional: TI mmWave config used for this capture (highly recommended)
cfg_fp    = r"C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\xwr68xx_profile_2025_06_08.cfg.txt"

# --- FIGURE OUTPUT CONTROL ---
SHOW_FIGS = True                         # set False to avoid blocking at plt.show()
SAVE_FIGS = False                        # set True to save all figures instead of (or in addition to) showing
OUT_DIR   = r"C:\\Users\\kaden\\Documents\\OpenRadars\\figs"  # used if SAVE_FIGS=True

# --- TUNABLES ---
ANGLE_NFFT = 128           # zero-padding for angle FFT
SNR_DB_MIN = 20.0          # min RD SNR (dB) for using peaks in wall inference
CALIB_USE_PHASE = True     # apply per-(TX,RX) complex phase calibration if present in cfg
VIRT_SPACING_LAMBDA = 0.5  # assume λ/2 spacing for ULA

# --- VISUALIZATION OPTIONS ---
RD_FRAME = 445   # which frame to visualize in RD & AoA (0 .. num_frames-1)

# --- ADDITIONS to track the paper ---
DEVIATION_THR_DEG = 15.0   # |θ'_meas - θ'_theory| threshold used to "noise-reduce" ghost AoA
TRACK_MAX_FRAMES  = 300    # cap for per-frame trajectory extraction to limit runtime


def bitcount(x: int) -> int:
    return int(bin(int(x) & 0xFFFFFFFF).count("1"))

# -------------------------------
# CFG parsing helpers
# -------------------------------
def parse_ti_cfg(cfg_path: str):
    cfg_text = Path(cfg_path).read_text()
    rx_en = tx_en = None
    num_adc_samples = None
    chirp_start = chirp_end = None
    num_loops = None
    for line in cfg_text.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        tok = line.split()
        if tok[0] == 'channelCfg':
            rx_en = int(tok[1]); tx_en = int(tok[2])
        elif tok[0] == 'profileCfg':
            num_adc_samples = int(tok[10])
        elif tok[0] == 'frameCfg':
            chirp_start = int(tok[1]); chirp_end = int(tok[2])
            num_loops = int(tok[3])
    assert rx_en is not None and tx_en is not None, 'channelCfg not found in cfg.'
    assert num_adc_samples is not None, 'profileCfg (numAdcSamples) not found in cfg.'
    assert chirp_start is not None and chirp_end is not None and num_loops is not None, 'frameCfg not found in cfg.'
    num_rx = bitcount(rx_en)
    num_tx_like = (chirp_end - chirp_start + 1)
    return {
        'num_rx': num_rx,
        'num_tx_like': num_tx_like,
        'num_adc_samples': num_adc_samples,
        'num_loops': num_loops,
    }

def parse_ti_cfg_details(cfg_path: str):
    start_freq_ghz = idle_time_us = adc_start_time_us = ramp_end_time_us = None
    slope_mhz_per_us = fs_ksps = None
    range_bias_m = 0.0
    chan_phase = None
    txt = Path(cfg_path).read_text()
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        tok = line.split()
        if tok[0] == 'profileCfg':
            start_freq_ghz    = float(tok[2])
            idle_time_us      = float(tok[3])
            adc_start_time_us = float(tok[4])
            ramp_end_time_us  = float(tok[5])
            slope_mhz_per_us  = float(tok[8])
            fs_ksps           = float(tok[11])
        elif tok[0] == 'compRangeBiasAndRxChanPhase':
            try:
                range_bias_m = float(tok[1])
                vals = list(map(float, tok[2:]))
                c = np.array(vals[0:32], dtype=np.float32)
                if c.size >= 2:
                    chan_phase = c[0::2] + 1j * c[1::2]
            except Exception:
                chan_phase = None
    if None in (start_freq_ghz, idle_time_us, adc_start_time_us, ramp_end_time_us, slope_mhz_per_us, fs_ksps):
        raise ValueError('profileCfg not found or malformed in cfg file')
    out = {
        'start_freq_ghz': start_freq_ghz,
        'idle_time_us': idle_time_us,
        'adc_start_time_us': adc_start_time_us,
        'ramp_end_time_us': ramp_end_time_us,
        'slope_mhz_per_us': slope_mhz_per_us,
        'fs_ksps': fs_ksps,
        'range_bias_m': range_bias_m,
    }
    if chan_phase is not None:
        out['chan_phase'] = chan_phase.astype(np.complex64)
    return out

# -------------------------------
# Reshape and FFTs
# -------------------------------
def reshape_clean_adc(clean_bin_path: str, cfg_params: dict):
    num_rx          = cfg_params['num_rx']
    num_tx_like     = cfg_params['num_tx_like']
    num_adc_samples = cfg_params['num_adc_samples']
    num_loops       = cfg_params['num_loops']

    raw_i16 = np.fromfile(clean_bin_path, dtype=np.int16)
    if raw_i16.size % 2 != 0:
        raw_i16 = np.ascontiguousarray(raw_i16[:-1])

    iq = raw_i16.reshape(-1, 2)
    cplx = iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)

    vals_per_chirp = num_rx * num_adc_samples
    total_chirps = cplx.size // vals_per_chirp
    if total_chirps == 0:
        raise ValueError('No full chirps found. Check cfg vs. data file.')

    used = total_chirps * vals_per_chirp
    cplx = cplx[:used]

    data = cplx.reshape(total_chirps, num_rx, num_adc_samples)

    chirps_per_frame = num_loops * num_tx_like
    num_frames = total_chirps // chirps_per_frame
    if num_frames == 0:
        raise ValueError('Not enough chirps for a full frame. Capture may be truncated.')

    used_chirps = num_frames * chirps_per_frame
    data = data[:used_chirps]

    data = data.reshape(num_frames, num_loops, num_tx_like, num_rx, num_adc_samples)
    return data

def _apply_window(x: np.ndarray, win: np.ndarray, axis: int) -> np.ndarray:
    x_move = np.moveaxis(x, axis, -1)
    y = x_move * win
    return np.moveaxis(y, -1, axis)

def range_doppler_with_tx_separation(data: np.ndarray, _params: dict,
                                     n_range: int | None = None,
                                     n_doppler: int | None = None,
                                     clutter_mode: str = "mti"):
    """Range FFT -> keep one half (reverse if negative half chosen) -> Doppler FFT.
    Returns:
      Xr          (f, l, t, r, N_range_half)
      Xrd_txrx    (f, N_range_half, N_dopp, t, r)
      Xrd_virtual (f, N_range_half, N_dopp, t*r)
    """
    f, l, t, r, n = data.shape
    if n_range is None:
        n_range = n
    if n_doppler is None:
        n_doppler = l

    win_r = np.hanning(n).astype(np.float32)
    Xr = _apply_window(data, win_r, axis=4)
    Xr = np.fft.fft(Xr, n=n_range, axis=4).astype(np.complex64, copy=False)

    # keep one half; reverse if we keep the negative-beat half
    N  = Xr.shape[-1]
    N2 = N // 2
    pos = Xr[..., :N2]
    neg = Xr[..., N2:]
    Epos = np.sum(np.abs(pos)**2)
    Eneg = np.sum(np.abs(neg)**2)
    Xr = neg[..., ::-1] if Eneg > Epos else pos  # now index 0 is the smallest |f_b|

    # --- clutter removal along slow-time (per range bin / per TX-RX) ---
    if clutter_mode == "mti":
        Xr = Xr - Xr.mean(axis=1, keepdims=True)   # first-order MTI (DC kill)
    elif clutter_mode == "off":
        pass
    else:
        raise ValueError(f"unknown clutter_mode: {clutter_mode}")

    # Doppler FFT along loops (axis=1), keeping TXs separated
    win_d = np.hanning(l).astype(np.float32)
    Xrd = _apply_window(Xr, win_d, axis=1)
    Xrd = np.fft.fft(Xrd, n=n_doppler, axis=1).astype(np.complex64, copy=False)
    Xrd = np.fft.fftshift(Xrd, axes=1)

    # (f, N_range_half, N_dopp, t, r)
    Xrd_txrx = np.transpose(Xrd, (0, 4, 1, 2, 3))

    # virtual array after Doppler
    n_range_eff = Xrd_txrx.shape[1]
    Xrd_virtual = Xrd_txrx.reshape(f, n_range_eff, n_doppler, t * r)

    return Xr, Xrd_txrx, Xrd_virtual

# -------------------------------
# Axes
# -------------------------------
def make_axes(params: dict, n_range: int, n_dopp: int, n_angle: int):
    c = 299_792_458.0
    fs = float(params.get('fs_ksps', 0.0)) * 1e3
    n = int(params['num_adc_samples'])
    slope_MHz_per_us = float(params['slope_mhz_per_us'])
    slope_Hz_per_s = slope_MHz_per_us * 1e12
    T_samp = n / fs if fs > 0 else float('nan')
    B = slope_Hz_per_s * T_samp
    dR = c / (2.0 * B) if B > 0 else float('nan')
    rng_bins_m = dR * np.arange(n_range)

    Ti = float(params['idle_time_us']) * 1e-6
    Tr = float(params['ramp_end_time_us']) * 1e-6
    Tchirp = Ti + Tr
    t = int(params['num_tx_like'])
    TD = t * Tchirp
    fd = np.fft.fftshift(np.fft.fftfreq(n_dopp, d=TD))
    lam = c / (float(params.get('start_freq_ghz', 60.0)) * 1e9)
    vel_mps = fd * lam / 2.0

    # Angle axis for λ/2 spacing
    k = np.arange(n_angle) - (n_angle // 2)
    sin_theta = np.clip(2.0 * k / float(n_angle), -1.0, 1.0)
    ang_deg = np.degrees(np.arcsin(sin_theta))
    return rng_bins_m, fd, vel_mps, ang_deg

# -------------------------------
# Angle FFT — PER-CELL
# -------------------------------
def angle_spectrum_cell(Xrd_virtual: np.ndarray, params: dict,
                        frame_idx: int, r_idx: int, d_idx: int,
                        n_angle: int = ANGLE_NFFT) -> np.ndarray:
    t = int(params['num_tx_like'])
    r = int(params['num_rx'])
    x = Xrd_virtual[frame_idx, r_idx, d_idx, :].reshape(t, r)

    calib = params.get('chan_phase', None)
    if CALIB_USE_PHASE and (calib is not None) and (calib.size >= t*r):
        calib_v = calib[:t*r].reshape(t, r)
        w = np.conj(calib_v / (np.abs(calib_v) + 1e-12))
        x = x * w

    if t > 1:
        Ti = float(params['idle_time_us']) * 1e-6
        Tr = float(params['ramp_end_time_us']) * 1e-6
        Tchirp = Ti + Tr
        N_dopp = Xrd_virtual.shape[2]
        fd = np.fft.fftshift(np.fft.fftfreq(N_dopp, d=t*Tchirp))
        f_d = fd[d_idx]
        m_idx = np.arange(t, dtype=int).reshape(t, 1)
        phase = np.exp(-1j * 2 * np.pi * f_d * m_idx * Tchirp)
        x = x * phase

    xv = x.reshape(t * r)
    spec = np.fft.fftshift(np.fft.fft(xv, n=n_angle))
    return spec

# -------------------------------
# Helpers
# -------------------------------
def _rd_power_db_from_txrx(Xrd_txrx, frame_idx: int = 0):
    """Avg TX×RX power in dB for a given frame."""
    rd = np.abs(Xrd_txrx[frame_idx]).mean(axis=(2, 3))
    return 20 * np.log10(np.maximum(rd, 1e-12)), rd

def _rti_from_Xr(Xr):
    """Range–Time Intensity (dB): collapse loops×TX×RX per frame."""
    p = (np.abs(Xr) ** 2).mean(axis=(1, 2, 3))  # (frames, Nrange_half)
    return 20 * np.log10(np.maximum(p, 1e-12))

# -------------------------------
# AoA curves
# -------------------------------
def extract_angles_vs_range(Xrd_virtual, params, doppler_idx, ang_deg, rng_bins_m,
                            main_mask_deg=10.0, snr_thresh_db=6.0, frame_idx=0):
    R = Xrd_virtual.shape[1]
    ranges_m = rng_bins_m - params.get('range_bias_m', 0.0)
    th_main = np.full(R, np.nan, dtype=np.float32)
    th_ghost = np.full(R, np.nan, dtype=np.float32)

    if len(ang_deg) > 1:
        deg_per_bin = abs(ang_deg[1] - ang_deg[0])
    else:
        deg_per_bin = 1.0
    mask_bins = max(1, int(round(main_mask_deg / max(deg_per_bin, 1e-6))))

    for r in range(R):
        spec = angle_spectrum_cell(Xrd_virtual, params, frame_idx, r, doppler_idx, n_angle=ANGLE_NFFT)
        p_db = 20*np.log10(np.maximum(np.abs(spec), 1e-12))

        noise = np.median(p_db)
        k_main = int(np.argmax(p_db))
        if p_db[k_main] - noise < snr_thresh_db:
            continue

        th_main[r] = ang_deg[k_main]

        p_db_masked = p_db.copy()
        p_db_masked[max(0, k_main - mask_bins): k_main + mask_bins + 1] = -1e9
        k_ghost = int(np.argmax(p_db_masked))
        if p_db[k_ghost] - noise < snr_thresh_db:
            continue

        th_ghost[r] = ang_deg[k_ghost]

    return ranges_m, th_main, th_ghost

def ghost_angle_theory(a_m, d_m, theta_main_rad):
    cos_t = np.cos(theta_main_rad)
    cos_t = np.where(np.abs(cos_t) < 1e-6, np.nan, cos_t)
    tan_theta_p = (2.0 * a_m / np.maximum(d_m, 1e-6) - np.sin(theta_main_rad)) / cos_t
    return np.degrees(np.arctan(tan_theta_p))

# -------------------------------
# Derived metrics & report
# -------------------------------
def _derived_metrics(p: dict):
    c = 299_792_458.0
    fc_hz = float(p.get('start_freq_ghz', 60.0)) * 1e9
    lam = c / fc_hz
    n = int(p['num_adc_samples'])
    Fs = float(p.get('fs_ksps', 0.0)) * 1e3
    T_samp = (n / Fs) if Fs > 0 else float('nan')
    S_Hz_per_s = float(p['slope_mhz_per_us']) * 1e12
    B_eff = S_Hz_per_s * T_samp
    dR = c / (2.0 * B_eff) if B_eff > 0 else float('nan')
    Ti = float(p['idle_time_us']) * 1e-6
    Tramp = float(p['ramp_end_time_us']) * 1e-6
    Tchirp = Ti + Tramp
    numTx = int(p['num_tx_like'])
    M = int(p['num_loops'])
    TD = numTx * Tchirp
    vmax = lam / (4.0 * TD) if TD > 0 else float('nan')
    dv = lam / (2.0 * M * TD) if TD > 0 else float('nan')
    available_win = max(0.0, Tramp - float(p['adc_start_time_us']) * 1e-6)
    margin = available_win - T_samp
    return {
        'lambda_m': lam,
        'T_samp_s': T_samp,
        'B_eff_Hz': B_eff,
        'range_res_m': dR,
        'T_chirp_s': Tchirp,
        'T_between_same_tx_s': TD,
        'vmax_mps': vmax,
        'doppler_res_mps': dv,
        'available_sample_window_s': available_win,
        'window_margin_s': margin,
        'virt_ant': int(p['num_tx_like']) * int(p['num_rx']),
    }

def print_cfg_report(p: dict):
    m = _derived_metrics(p)
    def us(x): return f"{x*1e6:.2f} µs"
    print("\n=== Radar CFG Summary ===")
    print(f"fc: {p.get('start_freq_ghz', 60.0)} GHz  (λ={m['lambda_m']*1e3:.2f} mm)")
    print(f"Idle: {p['idle_time_us']} µs   Ramp: {p['ramp_end_time_us']} µs   ADC start: {p['adc_start_time_us']} µs")
    print(f"Slope: {p['slope_mhz_per_us']} MHz/µs   Fs: {p['fs_ksps']} ksps   n: {p['num_adc_samples']}")
    print(f"RX: {p['num_rx']}   TX-like: {p['num_tx_like']}   Virtual elements: {m['virt_ant']}")
    print(f"Loops per frame (per TX): {p['num_loops']}")
    print(f"Chirp time: {us(m['T_chirp_s'])}   Time between same-TX chirps: {us(m['T_between_same_tx_s'])}")
    print(f"Sampled ramp time: {us(m['T_samp_s'])}   Available window: {us(m['available_sample_window_s'])}")
    if m['window_margin_s'] < 0:
        print(f"WARNING: Samples don't fit on the ramp by {-m['window_margin_s']*1e6:.2f} µs. Increase rampEndTime or reduce n/Fs.")
    else:
        print(f"Margin on ramp: {us(m['window_margin_s'])}")
    print(f"Effective bandwidth: {m['B_eff_Hz']/1e9:.3f} GHz   Range resolution: {m['range_res_m']:.3f} m")
    print(f"v_max (TDM-aware): {m['vmax_mps']:.3f} m/s   Doppler bin: {m['doppler_res_mps']:.3f} m/s")
    if 'range_bias_m' in p:
        print(f"Range bias (cfg): {p['range_bias_m']:.3f} m")
    print("========================")

# -------------------------------
# Geometry: wall inference (2.22, 2.23, 2.24)
# -------------------------------
def infer_wall_a_from_d_dprime_theta(d, dprime, theta_rad):
    A = 4.0 * d * np.sin(theta_rad)
    disc = A*A - 16.0*(d*d - dprime*dprime)
    if disc < 0:
        return np.nan
    return (A + np.sqrt(disc)) / 8.0

def infer_wall_a_from_angles(d, theta_rad, theta_prime_rad):
    return 0.5 * d * (np.cos(theta_rad) * np.tan(theta_prime_rad) + np.sin(theta_rad))

def infer_wall_a_from_vel(d, theta_rad, vr_main, vr_ghost):
    ratio = np.clip(vr_ghost / (vr_main + 1e-12), -1.0, 1.0)
    arg = np.arccos(ratio) - theta_rad
    return 0.5 * d * (np.cos(theta_rad) * np.tan(arg) + np.sin(theta_rad))

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parse_raw_adc(source_fp, dest_fp)

    try:
        params = parse_ti_cfg(cfg_fp)
        params.update(parse_ti_cfg_details(cfg_fp))
    except Exception as e:
        raise SystemExit(f"CFG parse failed: {e}")

    print_cfg_report(params)

    data = reshape_clean_adc(dest_fp, params)
    print('Reshaped data shape:', data.shape)

    # --- compute BOTH before and after clutter removal ---
    Xr_no, Xrd_txrx_no, Xrd_virtual_no = range_doppler_with_tx_separation(data, params, clutter_mode="off")
    Xr,    Xrd_txrx,    Xrd_virtual    = range_doppler_with_tx_separation(data, params, clutter_mode="mti")
    print('After Range FFT (no clutter) :', Xr_no.shape)
    print('After Range FFT (with MTI)   :', Xr.shape)
    print('After Doppler (TX,RX)        :', Xrd_txrx.shape)
    print('Virtual array cube           :', Xrd_virtual.shape)

    # RTI before freeing memory
    _rti_no_db  = _rti_from_Xr(Xr_no)
    _rti_mti_db = _rti_from_Xr(Xr)

    del data
    del Xr
    gc.collect()

    # Range–Doppler quicklook (chosen frame, AFTER)
    rd0 = np.abs(Xrd_txrx[RD_FRAME]).mean(axis=(2, 3))
    rd0_db = 20 * np.log10(np.maximum(rd0, 1e-12))
    noise_db = np.median(rd0_db)
    N_dopp = rd0_db.shape[1]
    clutter_w = max(1, N_dopp // 64)

    rd_mod = rd0_db.copy()
    rd_mod[:, N_dopp // 2 - clutter_w : N_dopp // 2 + clutter_w] = -1e9
    peak_idx = np.unravel_index(np.argmax(rd_mod), rd_mod.shape)
    snr_main_db = rd0_db[peak_idx] - noise_db

    rd_mod2 = rd_mod.copy()
    i, j = peak_idx
    rd_mod2[max(0, i - 2): i + 3, max(0, j - 3): j + 4] = -1e9
    rd_mod2[: i + 1, :] = -1e9
    rd_mod2[:, j:] = -1e9
    ghost_idx = np.unravel_index(np.argmax(rd_mod2), rd_mod2.shape)
    snr_ghost_db = rd0_db[ghost_idx] - noise_db
    print(f"SNR main ~ {snr_main_db:.1f} dB, SNR ghost ~ {snr_ghost_db:.1f} dB")

    rng_bins_m, fd_hz, vel_mps, ang_deg = make_axes(
        params,
        n_range=Xrd_virtual.shape[1],
        n_dopp=Xrd_virtual.shape[2],
        n_angle=ANGLE_NFFT
    )

    # Angle spectra at the two RD peaks (AFTER)
    spec_main  = angle_spectrum_cell(Xrd_virtual, params, RD_FRAME, peak_idx[0],  peak_idx[1],  n_angle=ANGLE_NFFT)
    spec_ghost = angle_spectrum_cell(Xrd_virtual, params, RD_FRAME, ghost_idx[0], ghost_idx[1], n_angle=ANGLE_NFFT)
    k_main  = int(np.argmax(np.abs(spec_main)));  ang_main_deg  = ang_deg[k_main]
    k_ghost = int(np.argmax(np.abs(spec_ghost))); ang_ghost_deg = ang_deg[k_ghost]

    # Wall inference (scalars) if SNR is adequate
    a_22 = a_23 = a_24 = np.nan
    if (snr_main_db >= SNR_DB_MIN) and (snr_ghost_db >= SNR_DB_MIN):
        d  = rng_bins_m[peak_idx[0]]  - params.get('range_bias_m', 0.0)
        dp = rng_bins_m[ghost_idx[0]] - params.get('range_bias_m', 0.0)
        th  = np.radians(ang_main_deg)
        thp = np.radians(ang_ghost_deg)
        vr  = vel_mps[peak_idx[1]]
        vrp = vel_mps[ghost_idx[1]]
        a_22 = infer_wall_a_from_d_dprime_theta(d, dp, th)
        a_23 = infer_wall_a_from_angles(d, th, thp)
        a_24 = infer_wall_a_from_vel(d, th, vr, vrp)
    print(f"Estimated wall distance a (m): 2.22={a_22:.2f}, 2.23={a_23:.2f}, 2.24={a_24:.2f}")

    # ----------------- Plots -----------------
    rng_disp = rng_bins_m - params.get('range_bias_m', 0.0)

    # 1) Single Range–Doppler (AFTER)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        rd0_db, origin='lower', aspect='auto', cmap='jet',
        extent=[vel_mps[0], vel_mps[-1], rng_disp[0], rng_disp[-1]]
    )
    plt.colorbar(label='dB')
    plt.scatter(vel_mps[peak_idx[1]], rng_disp[peak_idx[0]], s=150, c='orange', marker='x', label=f'main ({snr_main_db:.0f} dB)')
    plt.scatter(vel_mps[ghost_idx[1]], rng_disp[ghost_idx[0]], s=150, c='lime', marker='x', label=f'ghost ({snr_ghost_db:.0f} dB)')
    plt.xlabel('Radial velocity (m/s)'); plt.ylabel('Range (m)')
    plt.title(f'Range–Doppler (avg over TX×RX) — frame {RD_FRAME}')
    plt.legend(loc='upper right')

    # 2) Target distance vs AoA (ADD: estimator selection + deviation-thresholded ghost curve)
    ranges_m, theta_main_curve, theta_ghost_curve = extract_angles_vs_range(
        Xrd_virtual, params, doppler_idx=peak_idx[1], ang_deg=ang_deg, rng_bins_m=rng_bins_m,
        main_mask_deg=10.0, snr_thresh_db=6.0, frame_idx=RD_FRAME
    )

    # ---- ADDED: choose the best 'a' by minimizing AoA deviation to theory ----
    def _aoa_dev_cost(a_val, th_main_deg, th_ghost_deg, d_m):
        if not np.isfinite(a_val) or a_val <= 0:
            return np.inf
        th_theory = ghost_angle_theory(a_val, d_m, np.radians(th_main_deg))
        mask = np.isfinite(th_theory) & np.isfinite(th_ghost_deg)
        if mask.sum() < 5:
            return np.inf
        return np.nanmedian(np.abs(th_theory[mask] - th_ghost_deg[mask]))

    costs = {}
    for key, aval in (('2.22', a_22), ('2.23', a_23), ('2.24', a_24)):
        costs[key] = _aoa_dev_cost(aval, theta_main_curve, theta_ghost_curve, ranges_m)

    # pick the estimator with smallest median deviation
    best_key = min(costs, key=lambda k: costs[k]) if len(costs) else '2.23'
    a_use = {'2.22': a_22, '2.23': a_23, '2.24': a_24}.get(best_key, np.nan)

    theta_theory = None
    if np.isfinite(a_use) and a_use > 0:
        theta_theory = ghost_angle_theory(a_use, ranges_m, np.radians(theta_main_curve))

    # ---- ADDED: deviation-thresholded "noise-reduced" ghost AoA curve ----
    theta_ghost_clean = np.full_like(theta_ghost_curve, np.nan)
    if theta_theory is not None and np.any(np.isfinite(theta_theory)):
        diff = np.abs(theta_ghost_curve - theta_theory)
        keep = np.isfinite(diff) & (diff <= DEVIATION_THR_DEG)
        theta_ghost_clean[keep] = theta_ghost_curve[keep]
        # light median filtering on the retained samples
        tmp = np.where(np.isfinite(theta_ghost_clean), theta_ghost_clean, 0.0)
        sm = medfilt(tmp, kernel_size=5)
        sm[~np.isfinite(theta_ghost_clean)] = np.nan
        theta_ghost_clean = sm

    plt.figure(figsize=(9, 6))
    if np.any(np.isfinite(theta_ghost_curve)):
        plt.plot(theta_ghost_curve, ranges_m, label='FFT ghost angle')
    if theta_theory is not None and np.any(np.isfinite(theta_theory)):
        plt.plot(theta_theory, ranges_m, 'r:', label=f'theoretical angle (a from {best_key})')
    if np.any(np.isfinite(theta_ghost_clean)):
        plt.plot(theta_ghost_clean, ranges_m, 'y--', label=f'noise reduced (|Δ|≤{DEVIATION_THR_DEG:.0f}°)')
    if np.any(np.isfinite(theta_main_curve)):
        plt.plot(theta_main_curve, ranges_m, 'g-.', label='FFT main angle')
    plt.gca().invert_yaxis()
    plt.xlabel('AoA (degree)')
    plt.ylabel('Target distance (m)')
    plt.title(f'Target distance vs AoA (frame {RD_FRAME} @ main Doppler)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3) Micro-Doppler (time × velocity) — global (unchanged)
    power = (np.abs(Xrd_virtual) ** 2).sum(axis=(1, 3))
    F, Nd = power.shape
    plt.figure(figsize=(11, 4))
    plt.imshow(20 * np.log10(np.maximum(power.T, 1e-12)), origin='lower', aspect='auto',
               extent=[0, F-1, vel_mps[0], vel_mps[-1]], cmap='jet')
    plt.title('Micro-Doppler (time × velocity)')
    plt.xlabel('Frame'); plt.ylabel('Radial velocity (m/s)')
    plt.tight_layout()

    # ----------------- A/B RD panel (Before vs After) with peak markers + Metrics -----------------
    rd_no_db, rd_no_lin = _rd_power_db_from_txrx(Xrd_txrx_no, RD_FRAME)  # BEFORE (chosen frame)
    rd_mti_db = rd0_db                                                   # AFTER  (chosen frame)
    vmin_ab, vmax_ab = np.percentile(rd_no_db, [5, 99])

    rd_after_lin = np.abs(Xrd_txrx[RD_FRAME]).mean(axis=(2, 3))
    zero_col = rd_no_db.shape[1] // 2
    clutter_att_db = 10 * np.log10(
        np.maximum(rd_no_lin[:, zero_col], 1e-12) /
        np.maximum(rd_after_lin[:, zero_col], 1e-12)
    ).mean()

    peak_no = rd_no_db.max()
    peak_mti = rd_mti_db.max()
    noise_no  = float(np.median(rd_no_db))
    noise_mti = float(np.median(rd_mti_db))

    # -------- BEFORE peak: NAÏVE (no zero-Doppler mask) ----------
    N_dopp_pre = rd_no_db.shape[1]
    peak_idx_pre_naive = np.unravel_index(np.argmax(rd_no_db), rd_no_db.shape)
    is_clutter_pick = (abs(peak_idx_pre_naive[1] - N_dopp_pre // 2) <= max(1, N_dopp_pre // 64))

    fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    extent = [vel_mps[0], vel_mps[-1], rng_disp[0], rng_disp[-1]]

    im0 = ax[0, 0].imshow(rd_no_db, origin='lower', aspect='auto', cmap='jet',
                           extent=extent, vmin=vmin_ab, vmax=vmax_ab)
    ax[0, 0].set_title(f'RD — BEFORE clutter removal (frame {RD_FRAME})')
    ax[0, 0].set_xlabel('Radial velocity (m/s)'); ax[0, 0].set_ylabel('Range (m)')

    im1 = ax[0, 1].imshow(rd_mti_db, origin='lower', aspect='auto', cmap='jet',
                           extent=extent, vmin=vmin_ab, vmax=vmax_ab)
    ax[0, 1].set_title(f'RD — AFTER clutter removal (frame {RD_FRAME})')
    ax[0, 1].set_xlabel('Radial velocity (m/s)'); ax[0, 1].set_ylabel('Range (m)')

    # Bottom-left: BEFORE — NAÏVE peak only (no masking)
    ax[1, 0].imshow(rd_no_db, origin='lower', aspect='auto', cmap='jet',
                    extent=extent, vmin=vmin_ab, vmax=vmax_ab)
    ax[1, 0].scatter(vel_mps[peak_idx_pre_naive[1]], rng_disp[peak_idx_pre_naive[0]],
                     facecolors='none', edgecolors='r', s=90, marker='o', linewidths=1.7, label='naïve peak')
    ax[1, 0].set_title('RD BEFORE — naïve peak (no v≈0 masking)')
    ax[1, 0].set_xlabel('Radial velocity (m/s)'); ax[1, 0].set_ylabel('Range (m)')
    ann = "CLUTTER" if is_clutter_pick else "target?"
    ax[1, 0].text(0.02, 0.02, f"Naïve picks {ann}",
                  transform=ax[1, 0].transAxes, fontsize=9,
                  bbox=dict(facecolor='k', alpha=0.3, pad=6), color='w')
    ax[1, 0].legend(loc='upper right', fontsize=8)

    # Bottom-right: AFTER + peak marker
    ax[1, 1].imshow(rd_mti_db, origin='lower', aspect='auto', cmap='jet',
                    extent=extent, vmin=vmin_ab, vmax=vmax_ab)
    ax[1, 1].set_title('RD AFTER + peak marker')
    ax[1, 1].set_xlabel('Radial velocity (m/s)'); ax[1, 1].set_ylabel('Range (m)')
    ax[1, 1].scatter(vel_mps[peak_idx[1]], rng_disp[peak_idx[0]], c='w', s=60, marker='x')

    fig.colorbar(im0, ax=[ax[0,0], ax[0,1]], label='dB')
    fig.colorbar(im0, ax=[ax[1,0]], label='dB')

    txt = (f"Noise: {noise_no:.1f}→{noise_mti:.1f} dB  |  Peak: {peak_no:.1f}→{peak_mti:.1f} dB\n"
           f"Mean clutter attenuation @ v≈0: {clutter_att_db:.1f} dB")
    ax[1, 1].text(0.02, 0.02, txt, transform=ax[1, 1].transAxes, fontsize=9,
                  bbox=dict(facecolor='k', alpha=0.3, pad=6), color='w')

    # ----------------- RTI (Range–Time) Before/After -----------------
    vmin_rti, vmax_rti = np.percentile(_rti_no_db, [5, 99])

    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.imshow(_rti_no_db.T, origin='lower', aspect='auto', cmap='jet',
               extent=[0, _rti_no_db.shape[0]-1, rng_disp[0], rng_disp[-1]],
               vmin=vmin_rti, vmax=vmax_rti)
    plt.title('RTI — BEFORE clutter removal'); plt.ylabel('Range (m)')

    plt.subplot(2, 1, 2)
    plt.imshow(_rti_mti_db.T, origin='lower', aspect='auto', cmap='jet',
               extent=[0, _rti_mti_db.shape[0]-1, rng_disp[0], rng_disp[-1]],
               vmin=vmin_rti, vmax=vmax_rti)
    plt.title('RTI — AFTER clutter removal'); plt.xlabel('Frame'); plt.ylabel('Range (m)')
    plt.tight_layout()

    # ----------------- Adaptive range-weighted Micro-Doppler (keeps full walk) -----------------
    v_abs = np.abs(vel_mps)
    dop_mask = (v_abs >= 0.10) & (v_abs <= 3.00)

    rd_power = (np.abs(Xrd_txrx)**2)              # (F, R, Nd, T, Rx)
    E_fr = rd_power.sum(axis=(3,4))               # (F, R, Nd)
    E_fr = E_fr[:, :, dop_mask].sum(axis=2)       # (F, R)

    E_fr = np.apply_along_axis(lambda x: np.convolve(x, np.ones(5)/5.0, mode='same'), 1, E_fr)

    Fnum, Rnum = E_fr.shape
    r_center = np.zeros(Fnum, dtype=int)
    r_center[0] = int(np.argmax(E_fr[0]))
    alpha = 0.8
    for fidx in range(1, Fnum):
        r_raw = int(np.argmax(E_fr[fidx]))
        r_center[fidx] = int(round(alpha * r_center[fidx-1] + (1.0 - alpha) * r_raw))

    dR = float(np.abs(rng_disp[1] - rng_disp[0])) if Rnum > 1 else 0.1
    sigma_m = 0.8
    sigma_bins = max(2.0, sigma_m / max(dR, 1e-6))
    r_grid = np.arange(Rnum, dtype=np.float32)
    W = np.exp(-0.5 * ((r_grid[None, :] - r_center[:, None]) / sigma_bins) ** 2)  # (F, R)
    W /= (W.sum(axis=1, keepdims=True) + 1e-12)

    P_frd = (np.abs(Xrd_virtual) ** 2).sum(axis=3)     # (F, R, Nd)
    power_adapt = (P_frd * W[:, :, None]).sum(axis=1)  # (F, Nd)

    plt.figure(figsize=(11, 4))
    plt.imshow(20 * np.log10(np.maximum(power_adapt.T, 1e-12)), origin='lower', aspect='auto',
               extent=[0, power_adapt.shape[0]-1, vel_mps[0], vel_mps[-1]], cmap='jet')
    plt.title('Micro-Doppler (adaptive range-weighted)')
    plt.xlabel('Frame'); plt.ylabel('Radial velocity (m/s)')
    plt.tight_layout()

    # ----------------- ADDED: Cartesian trajectories (x–y) over time -----------------
    # Derive per-frame main & ghost positions using AFTER-cube peak/ghost AoA and range.
    F_all = Xrd_virtual.shape[0]
    F_use = min(F_all, TRACK_MAX_FRAMES)
    x_main = np.full(F_use, np.nan, dtype=np.float32)
    y_main = np.full(F_use, np.nan, dtype=np.float32)
    x_ghost = np.full(F_use, np.nan, dtype=np.float32)
    y_ghost = np.full(F_use, np.nan, dtype=np.float32)

    for fidx in range(F_use):
        rd_f = np.abs(Xrd_txrx[fidx]).mean(axis=(2,3))   # (R, Nd)
        N_d = rd_f.shape[1]
        cw = max(1, N_d // 64)
        rd_f_mod = rd_f.copy()
        rd_f_mod[:, N_d//2 - cw : N_d//2 + cw] = -1e9
        pk_r, pk_d = np.unravel_index(np.argmax(rd_f_mod), rd_f_mod.shape)

        # ghost candidate (suppress vicinity of main + bias to lower-left like demo)
        rd_g = 20*np.log10(np.maximum(rd_f_mod, 1e-12))
        rd_g[max(0, pk_r - 2): pk_r + 3, max(0, pk_d - 3): pk_d + 4] = -1e9
        rd_g[: pk_r + 1, :] = -1e9
        rd_g[:, pk_d:] = -1e9
        gh_r, gh_d = np.unravel_index(np.argmax(rd_g), rd_g.shape)

        # AoA for main/ghost
        spec_m = angle_spectrum_cell(Xrd_virtual, params, fidx, pk_r, pk_d, n_angle=ANGLE_NFFT)
        spec_g = angle_spectrum_cell(Xrd_virtual, params, fidx, gh_r, gh_d, n_angle=ANGLE_NFFT)
        km = int(np.argmax(np.abs(spec_m))); kg = int(np.argmax(np.abs(spec_g)))
        thm = np.radians(ang_deg[km]); thg = np.radians(ang_deg[kg])

        rm = rng_disp[pk_r]; rg = rng_disp[gh_r]
        if np.isfinite(rm) and np.isfinite(thm):
            x_main[fidx] = rm * np.cos(thm); y_main[fidx] = rm * np.sin(thm)
        if np.isfinite(rg) and np.isfinite(thg):
            x_ghost[fidx] = rg * np.cos(thg); y_ghost[fidx] = rg * np.sin(thg)

    plt.figure(figsize=(7.5, 6))
    if np.any(np.isfinite(x_main)):
        plt.plot(x_main, y_main, label='main track')
    if np.any(np.isfinite(x_ghost)):
        plt.plot(x_ghost, y_ghost, label='ghost track')
    plt.axhline(0, linewidth=0.5, alpha=0.4)
    plt.axvline(0, linewidth=0.5, alpha=0.4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Cartesian trajectories over time (AFTER clutter removal)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # ---------- SHOW / SAVE ----------
    if SAVE_FIGS:
        from pathlib import Path as _P
        outdir = _P(OUT_DIR)
        outdir.mkdir(parents=True, exist_ok=True)
        for fid in plt.get_fignums():
            plt.figure(fid)
            plt.savefig(outdir / f"fig_{fid}.png", dpi=200, bbox_inches='tight')

    if SHOW_FIGS:
        try:
            plt.show()
        except KeyboardInterrupt:
            # Swallow Ctrl+C during interactive viewing to avoid noisy exit.
            pass
    else:
        plt.close('all')
