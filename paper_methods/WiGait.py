import numpy as np
from collections import deque
from bisect import bisect_left, insort


def _gaussian_smooth_1d(x, sigma_samples):
    """
    Simple 1D Gaussian smoothing implemented with NumPy only.
    """
    if sigma_samples is None or sigma_samples <= 0:
        return np.asarray(x, dtype=float)

    x = np.asarray(x, dtype=float)
    radius = int(3 * sigma_samples)
    if radius < 1:
        return x

    positions = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (positions / sigma_samples) ** 2)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def _compute_step_displacements(x, y):
    """
    Compute per-sample step displacements sqrt(dx^2 + dy^2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx * dx + dy * dy)


def compute_velocity(x, y, dt, T=0.2, smooth_sigma_seconds=0.05):
    """
    Compute velocity as in WiGait (eq. 1) using a sliding window:

        v(t) = sum_{i=n}^{n+m-1} ||p_{i+1} - p_i|| / T
        with T = m * dt

    Parameters
    ----------
    x, y : array-like
        Position coordinates.
    dt : float
        Sampling interval in seconds.
    T : float, optional
        Temporal window size in seconds (default 0.2s).
    smooth_sigma_seconds : float, optional
        Gaussian smoothing sigma on position before velocity
        calculation (must be smaller than typical step period).

    Returns
    -------
    v : np.ndarray
        Velocity estimate for each interval (length len(x) - 1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if smooth_sigma_seconds is not None and smooth_sigma_seconds > 0:
        sigma_samples = smooth_sigma_seconds / dt
        x = _gaussian_smooth_1d(x, sigma_samples)
        y = _gaussian_smooth_1d(y, sigma_samples)

    displacements = _compute_step_displacements(x, y)
    m = max(int(round(T / dt)), 1)
    if m == 1:
        return displacements / dt

    kernel = np.ones(m, dtype=float) / (m * dt)
    v = np.convolve(displacements, kernel, mode="same")
    return v


def _approximate_diameter_sliding(x, y, window_size_samples, k_dirs=6):
    """
    Streaming approximate diameter of locations in a sliding
    window, using projections onto k fixed directions (Algorithm 1
    in the paper). :contentReference[oaicite:2]{index=2}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])

    # Directions uniformly spaced between 0 and pi.
    angles = np.linspace(0.0, np.pi, k_dirs, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    # For each direction keep a sliding queue and a sorted list.
    queues = [deque() for _ in range(k_dirs)]
    sorted_lists = [[] for _ in range(k_dirs)]

    diameters = np.zeros(n, dtype=float)

    for i in range(n):
        max_diff = 0.0
        for k in range(k_dirs):
            p = x[i] * cos_t[k] + y[i] * sin_t[k]
            q = queues[k]
            arr = sorted_lists[k]

            q.append(p)
            insort(arr, p)

            if len(q) > window_size_samples:
                old = q.popleft()
                # remove 'old' from sorted list
                idx = bisect_left(arr, old)
                if idx < len(arr) and arr[idx] == old:
                    del arr[idx]
                else:
                    # Fallback linear search (numerical duplicates)
                    for j in range(len(arr)):
                        if arr[j] == old:
                            del arr[j]
                            break

            if arr:
                diff = arr[-1] - arr[0]
                if diff > max_diff:
                    max_diff = diff

        diameters[i] = max_diff

    return diameters


def _approximate_diameter_segment(x, y, start, end, k_dirs=6):
    """
    Approximate diameter for a fixed segment [start, end].
    """
    x_seg = np.asarray(x[start : end + 1], dtype=float)
    y_seg = np.asarray(y[start : end + 1], dtype=float)
    if x_seg.size == 0:
        return 0.0

    angles = np.linspace(0.0, np.pi, k_dirs, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    max_diff = 0.0
    for k in range(k_dirs):
        proj = x_seg * cos_t[k] + y_seg * sin_t[k]
        diff = proj.max() - proj.min()
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _has_walking_periodicity(
    x,
    y,
    dt,
    min_freq=0.5,
    max_freq=3.0,
    peak_ratio_threshold=0.1,
):
    """
    Periodicity test based on the FFT of velocity to determine
    whether a moving segment is pure walking (Section “Walking
    Period Identification”). :contentReference[oaicite:3]{index=3}
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 8:
        return False

    v = compute_velocity(x, y, dt, T=0.2, smooth_sigma_seconds=0.05)
    v = v - np.mean(v)

    n = len(v)
    if n < 16:
        return False

    spec = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(n, d=dt)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return False

    band = spec[mask]
    if band.size == 0:
        return False

    peak = band.max()
    total = band.sum()
    if total <= 0:
        return False

    return (peak / total) >= peak_ratio_threshold


def identify_walking_periods(
    t,
    x,
    y,
    window_seconds=4.0,
    diameter_threshold_stationary=1.6,
    min_path_length=4.0,
    k_dirs=6,
    periodicity_peak_ratio=0.1,
):
    """
    Step 1: Identify walking periods from raw location data.

    Implements the two-stage method on pages 3–4:
      (a) Sliding-window diameter to remove stationary / in-place
          motion using threshold B.
      (b) From remaining moving periods, keep only segments whose
          diameter exceeds 'min_path_length' and whose velocity
          shows strong periodicity (steps). :contentReference[oaicite:4]{index=4}

    Parameters
    ----------
    t : array-like
        Time stamps (seconds).
    x, y : array-like
        Position coordinates.
    window_seconds : float
        Sliding window size for stationary detection (default 4 s).
    diameter_threshold_stationary : float
        Threshold B (default 1.6 m).
    min_path_length : float
        Minimum spatial extent for a walking period (default 4 m).
    k_dirs : int
        Number of projection directions.
    periodicity_peak_ratio : float
        Minimum (peak / sum) ratio in the velocity FFT band.

    Returns
    -------
    list of (start_idx, end_idx)
        Walking periods as inclusive indices into the time series.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return []

    dt = float(np.median(np.diff(t)))
    window_size_samples = max(int(round(window_seconds / dt)), 1)

    diameters = _approximate_diameter_sliding(x, y, window_size_samples, k_dirs=k_dirs)

    moving_mask = diameters >= diameter_threshold_stationary

    walking_periods = []
    n = len(moving_mask)
    i = 0
    while i < n:
        if not moving_mask[i]:
            i += 1
            continue
        start = i
        while i < n and moving_mask[i]:
            i += 1
        end = i - 1

        # Check spatial extent of this moving segment.
        d_seg = _approximate_diameter_segment(x, y, start, end, k_dirs=k_dirs)
        if d_seg < min_path_length:
            continue

        # Periodicity test on velocity.
        if not _has_walking_periodicity(
            x[start : end + 1],
            y[start : end + 1],
            dt,
            peak_ratio_threshold=periodicity_peak_ratio,
        ):
            continue

        walking_periods.append((start, end))

    return walking_periods


def extract_stable_phase(
    t,
    x,
    y,
    period,
    dv=0.45,
    delta=1e-3,
    T=0.2,
    smooth_sigma_seconds=0.05,
    max_iterations=20,
):
    """
    Step 2: Extract the stable walking phase inside a walking
    period using the iterative algorithm on page 5. :contentReference[oaicite:5]{index=5}

    Parameters
    ----------
    t : array-like
        Time stamps (seconds).
    x, y : array-like
        Position coordinates.
    period : (int, int)
        (start_idx, end_idx) of a walking period (inclusive).
    dv : float
        Half-width of the stable velocity band (default 0.45 m/s).
    delta : float
        Convergence threshold on median velocity between
        iterations (default 0.001 m/s).
    T : float
        Window size for velocity computation (seconds).
    smooth_sigma_seconds : float
        Smoothing sigma for position (seconds).
    max_iterations : int
        Maximum number of refinement iterations.

    Returns
    -------
    stable_start, stable_end, v_stable : (int, int, float)
        Global indices (inclusive) for the stable phase and
        the final stable velocity estimate. On failure, returns
        (None, None, np.nan).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    start, end = period
    if end <= start:
        return None, None, np.nan

    dt = float(np.median(np.diff(t)))
    x_seg = x[start : end + 1]
    y_seg = y[start : end + 1]

    v = compute_velocity(x_seg, y_seg, dt, T=T, smooth_sigma_seconds=smooth_sigma_seconds)
    if v.size == 0:
        return None, None, np.nan

    # Initial stable-phase estimate: whole period.
    s0 = 0
    e0 = v.size - 1
    v_prev = None
    v_curr = np.median(v[s0 : e0 + 1])

    for _ in range(max_iterations):
        if v_prev is not None and abs(v_curr - v_prev) < delta:
            break

        v_prev = v_curr

        # Longest consecutive region where v >= v_curr - dv.
        threshold = v_curr - dv
        local_mask = v[s0 : e0 + 1] >= threshold

        best_len = 0
        best_start_local = None
        curr_len = 0
        curr_start_local = None

        for idx, val in enumerate(local_mask):
            if val:
                if curr_start_local is None:
                    curr_start_local = idx
                    curr_len = 1
                else:
                    curr_len += 1
            else:
                if curr_len > best_len:
                    best_len = curr_len
                    best_start_local = curr_start_local
                curr_start_local = None
                curr_len = 0

        if curr_len > best_len:
            best_len = curr_len
            best_start_local = curr_start_local

        if best_len <= 0 or best_start_local is None:
            return None, None, np.nan

        s0 = s0 + best_start_local
        e0 = s0 + best_len - 1

        v_curr = np.median(v[s0 : e0 + 1])

    stable_start = start + s0
    stable_end = start + e0
    return stable_start, stable_end, float(v_curr)


def _detrend(signal):
    """
    Remove a linear trend from a 1D signal using least squares.
    """
    y = np.asarray(signal, dtype=float)
    n = y.size
    if n <= 1:
        return y - y.mean() if n > 0 else y
    t = np.arange(n, dtype=float)
    p = np.polyfit(t, y, 1)
    trend = np.polyval(p, t)
    return y - trend


def estimate_stride_frequency_and_length(
    t,
    x,
    y,
    z,
    stable_period,
    T=0.2,
    smooth_sigma_seconds=0.05,
    freq_min=0.5,
    freq_max=3.0,
):
    """
    Step 3: Compute gait velocity and stride length from the
    stable phase (pages 5–6). :contentReference[oaicite:6]{index=6}

    - Gait velocity: median velocity in the stable phase.
    - Stride frequency: dominant frequency in the combined,
      normalized FFT of velocity and elevation.
    - Stride length: L = v_gait / f_stride.

    Parameters
    ----------
    t : array-like
        Time stamps (seconds).
    x, y, z : array-like
        Position coordinates; z is elevation.
    stable_period : (int, int)
        (start_idx, end_idx) of the stable phase (inclusive).
    T : float
        Window size for velocity computation (seconds).
    smooth_sigma_seconds : float
        Smoothing sigma for position (seconds).
    freq_min, freq_max : float
        Search band for stride frequency (Hz).

    Returns
    -------
    gait_velocity : float
    stride_frequency : float
    stride_length : float
        If stride_frequency cannot be estimated, stride_length
        is np.nan.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    start, end = stable_period
    if end <= start:
        return np.nan, np.nan, np.nan

    dt = float(np.median(np.diff(t)))

    sl = slice(start, end + 1)
    x_seg = x[sl]
    y_seg = y[sl]
    z_seg = z[sl]

    v = compute_velocity(x_seg, y_seg, dt, T=T, smooth_sigma_seconds=smooth_sigma_seconds)
    if v.size == 0:
        return np.nan, np.nan, np.nan

    gait_velocity = float(np.median(v))

    # Align z with velocity intervals.
    n = v.size
    if z_seg.size < n:
        return gait_velocity, np.nan, np.nan
    z_aligned = z_seg[:n]

    v_d = _detrend(v)
    z_d = _detrend(z_aligned)

    V = np.fft.rfft(v_d)
    Z = np.fft.rfft(z_d)
    freqs = np.fft.rfftfreq(n, d=dt)

    V_amp = np.abs(V)
    Z_amp = np.abs(Z)

    norm_v = np.linalg.norm(V_amp)
    norm_z = np.linalg.norm(Z_amp)
    if norm_v > 0:
        V_amp = V_amp / norm_v
    if norm_z > 0:
        Z_amp = Z_amp / norm_z

    combined = V_amp + Z_amp

    band_mask = (freqs >= freq_min) & (freqs <= freq_max)
    band_mask &= freqs > 0.0  # exclude DC

    if not np.any(band_mask):
        return gait_velocity, np.nan, np.nan

    band_indices = np.where(band_mask)[0]
    band_values = combined[band_mask]
    if band_values.size == 0:
        return gait_velocity, np.nan, np.nan

    peak_idx_in_band = np.argmax(band_values)
    stride_frequency = float(freqs[band_indices[peak_idx_in_band]])

    if stride_frequency <= 0:
        return gait_velocity, stride_frequency, np.nan

    stride_length = gait_velocity / stride_frequency
    return gait_velocity, stride_frequency, stride_length
