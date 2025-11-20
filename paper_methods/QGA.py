import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

C = 3e8  # speed of light in m/s


def butter_lowpass_zero_phase(x, cutoff_hz, fs_hz, order=4):
    """
    Zero-phase Butterworth low-pass filter along a 1D signal.
    """
    if x.ndim != 1:
        x = np.asarray(x).ravel()
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        # No filtering if cutoff is invalid for the given sampling rate
        return x.copy()
    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)
    return filtfilt(b, a, x)


def detect_zero_crossings(x):
    """
    Return indices where the signal crosses zero (sign change).
    Index returned is the first sample AFTER the crossing.
    """
    x = np.asarray(x)
    # Avoid zeros exactly at 0 by nudging
    eps = np.finfo(float).eps
    x = np.where(x >= 0, x + eps, x - eps)
    signs = np.sign(x)
    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings + 1


def detect_heel_strikes_from_trunk_accel(acc, t, min_interval_s=0.3, threshold=0.0):
    """
    Heel-strike detection from trunk acceleration signal using zero-crossings.
    Based on zero-crossings of low-pass filtered trunk acceleration.
    """
    acc = np.asarray(acc)
    t = np.asarray(t)
    if acc.shape != t.shape:
        raise ValueError("acc and t must have the same shape")

    # Optionally offset by threshold
    acc_thr = acc - threshold
    candidate_idx = detect_zero_crossings(acc_thr)
    heel_idx = []
    last_t = -np.inf
    for idx in candidate_idx:
        if idx <= 0 or idx >= len(t):
            continue
        ti = t[idx]
        if ti - last_t >= min_interval_s:
            heel_idx.append(idx)
            last_t = ti
    return np.array(heel_idx, dtype=int)


def compute_gait_from_trunk_velocity(trunk_velocity, t,
                                     v_lp_cutoff_hz=5.0,
                                     acc_lp_cutoff_hz=2.0,
                                     min_step_interval_s=0.3):
    """
    Common gait-parameter extraction from trunk radial velocity.

    Parameters
    ----------
    trunk_velocity : array_like
        Radial velocity of the trunk vs time [m/s].
    t : array_like
        Time stamps [s], same length as trunk_velocity.
    v_lp_cutoff_hz : float
        Low-pass cutoff for trunk velocity (default 5 Hz).
    acc_lp_cutoff_hz : float
        Low-pass cutoff for acceleration before heel-strike detection (default 2 Hz).
    min_step_interval_s : float
        Minimum allowed time between consecutive heel strikes [s].

    Returns
    -------
    results : dict
        step_times : array [s]
        step_lengths : array [m]
        ST : mean step time [s]
        SL : mean step length [m]
        STV : step time variability (CV, %)
        SLV : step length variability (CV, %)
        trunk_velocity_filt : filtered trunk velocity [m/s]
        trunk_acceleration_filt : filtered trunk acceleration [m/s^2]
        heel_indices : indices of detected heel strikes
    """
    v = np.asarray(trunk_velocity).astype(float)
    t = np.asarray(t).astype(float)
    if v.shape != t.shape:
        raise ValueError("trunk_velocity and t must have the same shape")

    # Sampling frequency inferred from time axis
    dt = np.mean(np.diff(t))
    if dt <= 0:
        raise ValueError("Time vector t must be strictly increasing")
    fs = 1.0 / dt

    # 1) Low-pass filter trunk velocity (5 Hz as in the paper)
    v_filt = butter_lowpass_zero_phase(v, cutoff_hz=v_lp_cutoff_hz, fs_hz=fs, order=4)

    # 2) Acceleration as first derivative (central finite difference)
    acc = np.gradient(v_filt, dt)

    # 3) Low-pass filter acceleration (2 Hz as in the paper)
    acc_filt = butter_lowpass_zero_phase(acc, cutoff_hz=acc_lp_cutoff_hz, fs_hz=fs, order=4)

    # 4) Heel strikes from zero-crossings of acceleration
    heel_idx = detect_heel_strikes_from_trunk_accel(acc_filt, t, min_interval_s=min_step_interval_s)

    if heel_idx.size < 2:
        # Not enough steps to compute variability
        return {
            "step_times": np.array([]),
            "step_lengths": np.array([]),
            "ST": np.nan,
            "SL": np.nan,
            "STV": np.nan,
            "SLV": np.nan,
            "trunk_velocity_filt": v_filt,
            "trunk_acceleration_filt": acc_filt,
            "heel_indices": heel_idx,
        }

    heel_times = t[heel_idx]

    # 5) Step times: time between consecutive heel strikes
    step_times = np.diff(heel_times)

    # 6) Step lengths: integral of absolute trunk speed over each step
    abs_v = np.abs(v_filt)
    step_lengths = []
    for i0, i1 in zip(heel_idx[:-1], heel_idx[1:]):
        if i1 <= i0:
            continue
        step_len = np.trapz(abs_v[i0:i1 + 1], t[i0:i1 + 1])
        step_lengths.append(step_len)
    step_lengths = np.array(step_lengths)

    if step_times.size == 0 or step_lengths.size == 0:
        ST = SL = STV = SLV = np.nan
    else:
        ST = float(np.mean(step_times))
        SL = float(np.mean(step_lengths))
        STV = float(100.0 * np.std(step_times, ddof=1) / ST) if step_times.size > 1 else np.nan
        SLV = float(100.0 * np.std(step_lengths, ddof=1) / SL) if step_lengths.size > 1 else np.nan

    return {
        "step_times": step_times,
        "step_lengths": step_lengths,
        "ST": ST,
        "SL": SL,
        "STV": STV,
        "SLV": SLV,
        "trunk_velocity_filt": v_filt,
        "trunk_acceleration_filt": acc_filt,
        "heel_indices": heel_idx,
    }


# ----------------------------------------------------------------------
# A. QGA Using RF Micro-Doppler Signatures
# ----------------------------------------------------------------------

def qga_using_micro_doppler(mu_doppler_mag,
                            freq_axis_hz,
                            time_axis_s,
                            carrier_freq_hz,
                            v_lp_cutoff_hz=5.0,
                            acc_lp_cutoff_hz=2.0,
                            min_step_interval_s=0.3):
    """
    Quantitative gait analysis using RF micro-Doppler signatures.

    This function assumes that a micro-Doppler (µD) time–frequency
    representation has already been computed from radar slow-time data
    (e.g. via STFT).

    Parameters
    ----------
    mu_doppler_mag : ndarray, shape (F, T)
        Magnitude of the µD spectrogram (linear scale, not dB).
    freq_axis_hz : ndarray, shape (F,)
        Doppler frequency axis corresponding to rows of mu_doppler_mag [Hz].
    time_axis_s : ndarray, shape (T,)
        Time axis corresponding to columns of mu_doppler_mag [s].
    carrier_freq_hz : float
        Radar carrier frequency f0 [Hz], used to convert Doppler to radial velocity.
    v_lp_cutoff_hz, acc_lp_cutoff_hz, min_step_interval_s :
        Parameters passed through to `compute_gait_from_trunk_velocity`.

    Returns
    -------
    results : dict
        Same keys as `compute_gait_from_trunk_velocity`.
    """
    mu_doppler_mag = np.asarray(mu_doppler_mag)
    freq_axis_hz = np.asarray(freq_axis_hz).astype(float)
    time_axis_s = np.asarray(time_axis_s).astype(float)

    if mu_doppler_mag.ndim != 2:
        raise ValueError("mu_doppler_mag must be 2D (F x T)")
    F, T = mu_doppler_mag.shape
    if freq_axis_hz.shape[0] != F:
        raise ValueError("freq_axis_hz length must match number of rows in mu_doppler_mag")
    if time_axis_s.shape[0] != T:
        raise ValueError("time_axis_s length must match number of columns in mu_doppler_mag")

    wavelength_m = C / carrier_freq_hz
    vel_axis = (wavelength_m / 2.0) * freq_axis_hz  # v = (λ/2) f_D

    # For each time slice, pick the velocity bin with maximum return,
    # which corresponds to the torso.
    torso_idx = np.argmax(mu_doppler_mag, axis=0)
    trunk_velocity = vel_axis[torso_idx]

    return compute_gait_from_trunk_velocity(
        trunk_velocity=trunk_velocity,
        t=time_axis_s,
        v_lp_cutoff_hz=v_lp_cutoff_hz,
        acc_lp_cutoff_hz=acc_lp_cutoff_hz,
        min_step_interval_s=min_step_interval_s,
    )


# ----------------------------------------------------------------------
# B. QGA Using Joint RF Data Representations
# ----------------------------------------------------------------------

def compute_gai(max_foot_vel_left, max_foot_vel_right):
    """
    Compute the Gait Asymmetry Indicator (GAI):

        GAI = | MFVR / MFVL - 1 |

    where MFVL and MFVR are the maximum foot velocities of
    the left and right legs, respectively.
    """
    MFVL = float(np.abs(max_foot_vel_left))
    MFVR = float(np.abs(max_foot_vel_right))
    eps = 1e-6
    if MFVL < eps or MFVR < eps:
        return np.nan
    return abs(MFVR / MFVL - 1.0)


def qga_using_joint_rf(trunk_velocity,
                       time_axis_s,
                       leg_velocity_left=None,
                       leg_velocity_right=None,
                       v_lp_cutoff_hz=5.0,
                       acc_lp_cutoff_hz=2.0,
                       min_step_interval_s=0.3):
    """
    Quantitative gait analysis using joint RF domain representations
    (range/Doppler/angle).

    This function operates on radial velocity time-series that have
    already been extracted from the joint RF data (e.g., torso and
    per-leg velocities derived from range-Doppler and range-angle maps).

    Parameters
    ----------
    trunk_velocity : array_like
        Radial velocity of the torso [m/s] as function of time.
    time_axis_s : array_like
        Time stamps [s] corresponding to trunk_velocity.
    leg_velocity_left : array_like or None
        Radial velocity of the left foot/leg [m/s] (optional).
    leg_velocity_right : array_like or None
        Radial velocity of the right foot/leg [m/s] (optional).
    v_lp_cutoff_hz, acc_lp_cutoff_hz, min_step_interval_s :
        Parameters forwarded to `compute_gait_from_trunk_velocity`.

    Returns
    -------
    results : dict
        All keys from `compute_gait_from_trunk_velocity`, plus:
        - 'MFVL', 'MFVR' : maximum absolute foot velocities [m/s] (if leg signals provided)
        - 'GAI' : gait asymmetry indicator, or np.nan if not computable.
    """
    base = compute_gait_from_trunk_velocity(
        trunk_velocity=np.asarray(trunk_velocity),
        t=np.asarray(time_axis_s),
        v_lp_cutoff_hz=v_lp_cutoff_hz,
        acc_lp_cutoff_hz=acc_lp_cutoff_hz,
        min_step_interval_s=min_step_interval_s,
    )

    MFVL = MFVR = None
    gai = np.nan

    if leg_velocity_left is not None and leg_velocity_right is not None:
        vL = np.asarray(leg_velocity_left).astype(float)
        vR = np.asarray(leg_velocity_right).astype(float)
        MFVL = float(np.max(np.abs(vL))) if vL.size > 0 else np.nan
        MFVR = float(np.max(np.abs(vR))) if vR.size > 0 else np.nan
        gai = compute_gai(MFVL, MFVR)

    base.update({
        "MFVL": MFVL,
        "MFVR": MFVR,
        "GAI": gai,
    })
    return base


# ----------------------------------------------------------------------
# C. QGA Using RF Skeletons
# ----------------------------------------------------------------------

def qga_using_rf_skeleton(joint_positions,
                          fs_hz,
                          left_ankle_idx,
                          right_ankle_idx,
                          left_hip_idx,
                          right_hip_idx,
                          ap_axis=0,
                          lp_cutoff_hz=10.0,
                          min_step_interval_s=0.3):
    """
    Quantitative gait analysis using RF-derived skeletons.

    This function assumes that a 3D skeleton (e.g., 14 joints) has
    already been estimated from RF data for each time frame, for example
    using a CNN+LSTM model. The method follows the description in the
    pilot study:

    - Low-pass filter joint trajectories at 10 Hz (4th-order Butterworth,
      zero lag).
    - Heel strikes: farthest anterior position of the ankles relative to
      the midpoint of the hips.
    - Step time: time interval between consecutive heel strikes.
    - Step length: absolute AP distance travelled by the ankle between
      consecutive heel strikes of the same leg.

    Parameters
    ----------
    joint_positions : ndarray, shape (T, J, 3)
        3D joint coordinates over time [m].
    fs_hz : float
        Sampling frequency of the skeleton time-series [Hz].
    left_ankle_idx, right_ankle_idx : int
        Indices of left/right ankle joints in the second dimension.
    left_hip_idx, right_hip_idx : int
        Indices of left/right hip joints in the second dimension.
    ap_axis : int
        Index of the antero-posterior (AP) axis in the 3D coordinates
        (0, 1 or 2).
    lp_cutoff_hz : float
        Low-pass cutoff for position filtering (default 10 Hz).
    min_step_interval_s : float
        Minimum time between heel strikes (used in peak distance).

    Returns
    -------
    results : dict
        step_times : array [s]
        step_lengths : array [m]
        ST, SL, STV, SLV (as in other paper_methods)
        heel_indices_L : indices of left heel strikes
        heel_indices_R : indices of right heel strikes
    """
    pos = np.asarray(joint_positions).astype(float)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError("joint_positions must have shape (T, J, 3)")

    T, J, _ = pos.shape
    if not (0 <= left_ankle_idx < J and 0 <= right_ankle_idx < J
            and 0 <= left_hip_idx < J and 0 <= right_hip_idx < J):
        raise ValueError("Joint indices out of range")

    if ap_axis not in (0, 1, 2):
        raise ValueError("ap_axis must be 0, 1 or 2")

    # Time vector
    t = np.arange(T) / float(fs_hz)

    # Extract AP coordinates
    la = pos[:, left_ankle_idx, ap_axis]
    ra = pos[:, right_ankle_idx, ap_axis]
    lh = pos[:, left_hip_idx, ap_axis]
    rh = pos[:, right_hip_idx, ap_axis]

    # Low-pass filter ankle and hip AP trajectories
    la_f = butter_lowpass_zero_phase(la, cutoff_hz=lp_cutoff_hz, fs_hz=fs_hz, order=4)
    ra_f = butter_lowpass_zero_phase(ra, cutoff_hz=lp_cutoff_hz, fs_hz=fs_hz, order=4)
    lh_f = butter_lowpass_zero_phase(lh, cutoff_hz=lp_cutoff_hz, fs_hz=fs_hz, order=4)
    rh_f = butter_lowpass_zero_phase(rh, cutoff_hz=lp_cutoff_hz, fs_hz=fs_hz, order=4)

    # Midpoint of hips
    mid_hips_ap = 0.5 * (lh_f + rh_f)

    # Relative AP positions (ankle - mid-hip)
    rel_la = la_f - mid_hips_ap
    rel_ra = ra_f - mid_hips_ap

    # Heel strikes as local maxima of relative AP displacement
    min_samples = int(round(min_step_interval_s * fs_hz))
    if min_samples < 1:
        min_samples = 1

    peaks_L, _ = find_peaks(rel_la, distance=min_samples)
    peaks_R, _ = find_peaks(rel_ra, distance=min_samples)

    # Build combined event list sorted in time
    events = [(idx, "L") for idx in peaks_L] + [(idx, "R") for idx in peaks_R]
    events.sort(key=lambda x: x[0])

    if len(events) < 2:
        return {
            "step_times": np.array([]),
            "step_lengths": np.array([]),
            "ST": np.nan,
            "SL": np.nan,
            "STV": np.nan,
            "SLV": np.nan,
            "heel_indices_L": np.array(peaks_L, dtype=int),
            "heel_indices_R": np.array(peaks_R, dtype=int),
        }

    # Step times: between consecutive heel strikes (irrespective of leg)
    event_indices = np.array([e[0] for e in events], dtype=int)
    event_times = t[event_indices]
    step_times = np.diff(event_times)

    # Step lengths: AP displacement of the ankle between consecutive
    # heel strikes of the SAME leg.
    step_lengths = []
    last_idx_for_leg = {"L": None, "R": None}
    for idx, leg in events:
        prev_idx_same_leg = last_idx_for_leg[leg]
        if prev_idx_same_leg is not None and idx > prev_idx_same_leg:
            if leg == "L":
                curr_pos = la_f[idx]
                prev_pos = la_f[prev_idx_same_leg]
            else:
                curr_pos = ra_f[idx]
                prev_pos = ra_f[prev_idx_same_leg]
            step_lengths.append(abs(curr_pos - prev_pos))
        last_idx_for_leg[leg] = idx

    step_lengths = np.asarray(step_lengths, dtype=float)

    if step_times.size == 0 or step_lengths.size == 0:
        ST = SL = STV = SLV = np.nan
    else:
        ST = float(np.mean(step_times))
        SL = float(np.mean(step_lengths))
        STV = float(100.0 * np.std(step_times, ddof=1) / ST) if step_times.size > 1 else np.nan
        SLV = float(100.0 * np.std(step_lengths, ddof=1) / SL) if step_lengths.size > 1 else np.nan

    return {
        "step_times": step_times,
        "step_lengths": step_lengths,
        "ST": ST,
        "SL": SL,
        "STV": STV,
        "SLV": SLV,
        "heel_indices_L": np.array(peaks_L, dtype=int),
        "heel_indices_R": np.array(peaks_R, dtype=int),
    }
