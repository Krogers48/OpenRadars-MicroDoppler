import numpy as np

# ============================================================
# A. Parsing and Signal Processing
#    - Range FFT
#    - Range–Doppler Map
# ============================================================

def compute_range_fft(adc_data, axis=-1, nfft=None, window=True):
    """
    First FFT (range-FFT) for FMCW radar (Section II.A).

    Parameters
    ----------
    adc_data : ndarray
        Complex ADC samples. The dimension specified by `axis` is the
        fast-time / range-sample axis.
    axis : int
        Axis corresponding to fast-time / range.
    nfft : int or None
        FFT size. If None, use adc_data.shape[axis].
    window : bool
        If True, apply a Hann window along `axis` before FFT.

    Returns
    -------
    range_fft : ndarray
        Complex range spectrum with same shape as `adc_data`, except the
        size of `axis` becomes `nfft`.
    """
    adc_data = np.asarray(adc_data, dtype=complex)
    if nfft is None:
        nfft = adc_data.shape[axis]

    if window:
        N = adc_data.shape[axis]
        w = np.hanning(N)
        shape = [1] * adc_data.ndim
        shape[axis] = N
        adc_data = adc_data * w.reshape(shape)

    range_fft = np.fft.fft(adc_data, n=nfft, axis=axis)
    return range_fft


def compute_range_doppler(adc_data, axis_range=-1, axis_doppler=-2,
                          nfft_range=None, nfft_doppler=None,
                          window_range=True, window_doppler=True):
    """
    Compute range–Doppler map (second FFT along Doppler axis) (Section II.A).

    Parameters
    ----------
    adc_data : ndarray
        Complex ADC samples with at least two dimensions:
        [..., num_chirps_per_frame, num_samples_per_chirp]
        where `axis_doppler` is chirp (slow-time within frame),
        and `axis_range` is fast-time.
    axis_range : int
        Axis for fast-time / range samples.
    axis_doppler : int
        Axis for Doppler / slow-time (chirps).
    nfft_range, nfft_doppler : int or None
        FFT sizes for range and Doppler. If None, use axis sizes.
    window_range, window_doppler : bool
        Apply Hann windows along the corresponding axes.

    Returns
    -------
    rd_map : ndarray
        Range–Doppler map with FFTs applied along both specified axes.
        Doppler axis is FFT-shifted to be centered at zero.
    """
    # Range FFT
    tmp = compute_range_fft(adc_data, axis=axis_range, nfft=nfft_range,
                            window=window_range)

    # Doppler FFT
    if nfft_doppler is None:
        nfft_doppler = tmp.shape[axis_doppler]

    tmp = np.asarray(tmp, dtype=complex)
    if window_doppler:
        Nd = tmp.shape[axis_doppler]
        w_d = np.hanning(Nd)
        shape_d = [1] * tmp.ndim
        shape_d[axis_doppler] = Nd
        tmp = tmp * w_d.reshape(shape_d)

    rd_map = np.fft.fftshift(
        np.fft.fft(tmp, n=nfft_doppler, axis=axis_doppler),
        axes=axis_doppler
    )
    return rd_map


# ============================================================
# B. Clutter Removal (Static/Passive Clutter)
# ============================================================

def remove_static_clutter(data, axis=0):
    """
    Stationary clutter removal by subtracting the mean over time
    (Section II.B).

    Parameters
    ----------
    data : ndarray
        Input radar data. The dimension specified by `axis` is treated
        as slow-time (frames). Everything else is left unchanged.
    axis : int
        Axis corresponding to time / slow-time.

    Returns
    -------
    cleaned : ndarray
        Data with mean over `axis` removed, suppressing static scatterers.
    """
    data = np.asarray(data)
    mean_over_time = np.mean(data, axis=axis, keepdims=True)
    return data - mean_over_time


# ============================================================
# C. 2D-CFAR on Range–Azimuth Heatmaps
# ============================================================

def ca_cfar_2d(ra_map, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-3):
    """
    2D CA-CFAR detector for range-azimuth heatmaps (Section II.C).

    Parameters
    ----------
    ra_map : ndarray, shape (num_range_bins, num_azimuth_bins)
        Power or magnitude-squared of range-azimuth map (linear scale, not dB).
    guard_cells : tuple of int
        Number of guard cells (range, azimuth) around CUT on each side.
    training_cells : tuple of int
        Number of training cells (range, azimuth) on each side.
    pfa : float
        Desired probability of false alarm.

    Returns
    -------
    detection_mask : bool ndarray, same shape as ra_map
        True where a detection is declared.
    threshold_map : ndarray, same shape as ra_map
        CFAR threshold used at each cell.
    """
    ra_map = np.asarray(ra_map, dtype=float)
    num_r, num_a = ra_map.shape
    g_r, g_a = guard_cells
    t_r, t_a = training_cells

    # Half window sizes (training + guard)
    w_r = g_r + t_r
    w_a = g_a + t_a

    detection_mask = np.zeros_like(ra_map, dtype=bool)
    threshold_map = np.zeros_like(ra_map, dtype=float)

    # Number of training cells in full window minus guard band + CUT
    total_cells = (2 * w_r + 1) * (2 * w_a + 1)
    guard_cells_count = (2 * g_r + 1) * (2 * g_a + 1)
    num_train = total_cells - guard_cells_count

    if num_train <= 0:
        raise ValueError("Not enough training cells, adjust guard_cells/training_cells.")

    # CFAR scaling factor alpha (CA-CFAR)
    alpha = num_train * (pfa ** (-1.0 / num_train) - 1.0)

    # Loop over valid CUT positions (avoid edges where window would spill)
    for r in range(w_r, num_r - w_r):
        r_start = r - w_r
        r_end = r + w_r + 1
        for a in range(w_a, num_a - w_a):
            a_start = a - w_a
            a_end = a + w_a + 1

            window = ra_map[r_start:r_end, a_start:a_end]

            # Mask out guard cells + CUT in the center
            g_r_start = t_r
            g_r_end = t_r + 2 * g_r + 1
            g_a_start = t_a
            g_a_end = t_a + 2 * g_a + 1

            train_window = window.copy()
            train_window[g_r_start:g_r_end, g_a_start:g_a_end] = 0.0

            noise_sum = np.sum(train_window)
            noise_level = noise_sum / num_train

            threshold = alpha * noise_level
            threshold_map[r, a] = threshold

            if ra_map[r, a] > threshold:
                detection_mask[r, a] = True

    return detection_mask, threshold_map


def build_cfar_detections(ra_cube, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-3):
    """
    Apply 2D CA-CFAR to each frame of a range–azimuth data cube.

    Parameters
    ----------
    ra_cube : ndarray, shape (num_frames, num_range_bins, num_azimuth_bins)
        Range-azimuth power maps for each frame.
    guard_cells, training_cells, pfa : see `ca_cfar_2d`.

    Returns
    -------
    detections : list of ndarrays
        detections[i] is an array of shape (Ni, 2) containing
        (range_idx, azimuth_idx) of detections in frame i.
    detection_masks : ndarray, same shape as ra_cube, dtype=bool
        Binary CFAR detection masks for each frame.
    """
    ra_cube = np.asarray(ra_cube, dtype=float)
    num_frames = ra_cube.shape[0]
    detections = []
    detection_masks = np.zeros_like(ra_cube, dtype=bool)

    for i in range(num_frames):
        mask, _ = ca_cfar_2d(ra_cube[i], guard_cells=guard_cells,
                             training_cells=training_cells, pfa=pfa)
        detection_masks[i] = mask
        pts = np.argwhere(mask)  # (N, 2)
        detections.append(pts)
    return detections, detection_masks


# ============================================================
# D. Unsupervised Machine Learning: DBSCAN
# ============================================================

def dbscan(points, eps=2.0, min_samples=5):
    """
    Simple DBSCAN clustering implementation (Section II.D).

    Parameters
    ----------
    points : ndarray, shape (N, D)
        Input points.
    eps : float
        Neighborhood radius.
    min_samples : int
        Minimum number of points required to form a dense region.

    Returns
    -------
    labels : ndarray, shape (N,)
        Cluster labels for each point. -1 indicates noise. Clusters are
        numbered starting from 0.
    """
    points = np.asarray(points, dtype=float)
    N = points.shape[0]
    if N == 0:
        return np.empty((0,), dtype=int)

    labels = np.full(N, -2, dtype=int)  # -2: unvisited, -1: noise, >=0 cluster ID
    cluster_id = 0

    # Precompute squared distances matrix (O(N^2))
    sq_dists = np.sum((points[None, :, :] - points[:, None, :]) ** 2, axis=2)

    def region_query(idx):
        return np.where(sq_dists[idx] <= eps ** 2)[0]

    for i in range(N):
        if labels[i] != -2:
            continue  # already processed

        neighbors = region_query(i)
        if neighbors.size < min_samples:
            labels[i] = -1  # noise
            continue

        # Start new cluster
        labels[i] = cluster_id
        seeds = set(neighbors.tolist())
        seeds.discard(i)

        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster_id  # border point
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            j_neighbors = region_query(j)
            if j_neighbors.size >= min_samples:
                for n in j_neighbors:
                    if labels[n] in (-2, -1):
                        seeds.add(int(n))

        cluster_id += 1

    labels[labels == -2] = -1
    return labels


def clusters_from_labels(points, labels):
    """
    Group points by cluster label.

    Parameters
    ----------
    points : ndarray, shape (N, D)
    labels : ndarray, shape (N,)

    Returns
    -------
    clusters : dict
        Mapping from cluster_id (int) to ndarray of member indices.
        Noise points (-1) are excluded.
    """
    clusters = {}
    for idx, lab in enumerate(labels):
        if lab < 0:
            continue
        clusters.setdefault(lab, []).append(idx)
    return {lab: np.array(idxs, dtype=int) for lab, idxs in clusters.items()}


# ============================================================
# E. Tracking and Association Algorithm (Algorithm 1)
#    - Initial position estimation
#    - Association tracking over time
# ============================================================

def _compute_cluster_center_for_frame(agg_points, ra_cube, member_indices, current_frame):
    """
    Internal helper: given aggregated points and a cluster (as indices into
    agg_points), pick representative center for the current frame based on
    max amplitude in the range-azimuth cube.
    """
    cluster_pts = agg_points[member_indices]  # (M, 3) -> (frame, range, azimuth)
    # Prefer points from current frame, if any
    mask_current = (cluster_pts[:, 0] == current_frame)
    if np.any(mask_current):
        pts_use = cluster_pts[mask_current]
    else:
        pts_use = cluster_pts

    amps = np.array([ra_cube[f, r, a] for (f, r, a) in pts_use])
    best_idx = int(np.argmax(amps))
    f, r, a = pts_use[best_idx]
    return np.array([float(r), float(a)], dtype=float), float(amps[best_idx])


def estimate_initial_center(ra_cube,
                            cfar_detections,
                            frame_indices,
                            eps=2.0,
                            min_samples=5,
                            range_scale=1.0,
                            azimuth_scale=1.0,
                            time_scale=1.0):
    """
    Estimate initial subject center based on Section II.E.1:
    select the cluster with the maximum number of detected points over
    the specified initial frames (subject standing still).

    Parameters
    ----------
    ra_cube : ndarray, shape (num_frames, num_range_bins, num_azimuth_bins)
        Range-azimuth power maps.
    cfar_detections : list of ndarrays
        CFAR detections as returned by `build_cfar_detections`.
    frame_indices : sequence of int
        Indices of initial frames where the subject is standing still or just
        about to walk.
    eps, min_samples, range_scale, azimuth_scale, time_scale : float
        DBSCAN and feature scaling parameters.

    Returns
    -------
    initial_center : ndarray, shape (2,)
        Estimated initial center (range_idx, azimuth_idx).
    """
    ra_cube = np.asarray(ra_cube, dtype=float)
    frame_indices = list(frame_indices)

    # Aggregate detections over the specified frames
    agg_points_list = []
    for f in frame_indices:
        pts = cfar_detections[f]
        if pts is None or len(pts) == 0:
            continue
        f_col = np.full((pts.shape[0], 1), f, dtype=int)
        agg_points_list.append(np.hstack([f_col, pts.astype(int)]))

    if not agg_points_list:
        raise ValueError("No CFAR detections found in the specified initial frames.")

    agg_points = np.vstack(agg_points_list)  # (M,3)

    # Features for DBSCAN: scaled [range, azimuth, time_rel]
    t0 = frame_indices[0]
    time_rel = (agg_points[:, 0] - t0).astype(float)

    features = np.stack([
        agg_points[:, 1].astype(float) * range_scale,
        agg_points[:, 2].astype(float) * azimuth_scale,
        time_rel * time_scale,
    ], axis=1)

    labels = dbscan(features, eps=eps, min_samples=min_samples)
    clusters = clusters_from_labels(features, labels)
    if not clusters:
        raise RuntimeError("DBSCAN found no valid clusters in initial frames.")

    # Select cluster with maximum number of points
    best_cluster_id = None
    best_count = -1
    for cid, member_indices in clusters.items():
        if member_indices.size > best_count:
            best_cluster_id = cid
            best_count = member_indices.size

    # Use last frame in frame_indices as reference for initial center
    ref_frame = frame_indices[-1]
    initial_center, _ = _compute_cluster_center_for_frame(
        agg_points, ra_cube, clusters[best_cluster_id], current_frame=ref_frame
    )
    return initial_center


def association_tracking(ra_cube,
                         cfar_detections,
                         initial_center,
                         n_frame=3,
                         eps=2.0,
                         min_samples=5,
                         range_scale=1.0,
                         azimuth_scale=1.0,
                         time_scale=1.0):
    """
    Association tracking algorithm (Algorithm 1, Section II.E.2) to follow
    the walking subject over time and suppress multipath / ghosting clusters.

    Parameters
    ----------
    ra_cube : ndarray, shape (num_frames, num_range_bins, num_azimuth_bins)
        Range-azimuth power maps.
    cfar_detections : list of ndarrays
        Output of `build_cfar_detections`. cfar_detections[i] is (Ni, 2)
        array of (range_idx, azimuth_idx) for frame i.
    initial_center : array-like of shape (2,)
        Initial estimate of subject center (range_idx, azimuth_idx) at frame 0.
    n_frame : int
        Number of frames to aggregate for clustering (N_frame in the paper).
    eps : float
        DBSCAN epsilon parameter (neighborhood radius).
    min_samples : int
        DBSCAN min_samples parameter.
    range_scale, azimuth_scale, time_scale : float
        Scaling factors used when building the 3D feature space
        [range, azimuth, time] for DBSCAN.

    Returns
    -------
    centers : ndarray, shape (num_frames, 2)
        Estimated subject center (range_idx, azimuth_idx) for each frame.
    """
    ra_cube = np.asarray(ra_cube, dtype=float)
    num_frames = ra_cube.shape[0]
    centers = np.zeros((num_frames, 2), dtype=float)
    center_old = np.asarray(initial_center, dtype=float)
    centers[0] = center_old

    for i in range(1, num_frames):
        # Frames to aggregate: [i - n_frame + 1, ..., i]
        start_frame = max(0, i - n_frame + 1)
        window_frames = list(range(start_frame, i + 1))

        # Build aggregated list of detections: each point = (frame, range, azimuth)
        agg_points_list = []
        for f in window_frames:
            pts = cfar_detections[f]
            if pts is None or len(pts) == 0:
                continue
            f_col = np.full((pts.shape[0], 1), f, dtype=int)
            agg_points_list.append(np.hstack([f_col, pts.astype(int)]))

        if not agg_points_list:
            centers[i] = center_old
            continue

        agg_points = np.vstack(agg_points_list)  # (M,3)

        # Features for DBSCAN: scaled [range, azimuth, time_rel]
        time_rel = (agg_points[:, 0] - i).astype(float)

        features = np.stack([
            agg_points[:, 1].astype(float) * range_scale,
            agg_points[:, 2].astype(float) * azimuth_scale,
            time_rel * time_scale,
        ], axis=1)

        labels = dbscan(features, eps=eps, min_samples=min_samples)
        clusters = clusters_from_labels(features, labels)

        if not clusters:
            centers[i] = center_old
            continue

        # Choose cluster whose center is closest to previous center_old
        best_center = center_old
        best_dist = None
        for cluster_id, member_indices in clusters.items():
            center, _ = _compute_cluster_center_for_frame(
                agg_points, ra_cube, member_indices, current_frame=i
            )
            dist = np.linalg.norm(center - center_old)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_center = center

        centers[i] = best_center
        center_old = best_center

    return centers


# ============================================================
# F. Gait Extraction Algorithm
#    - Peak detection on torso velocity
#    - Step points, step length, stride length, step count, speed
# ============================================================

def find_peaks_1d(signal, min_height=0.0, min_distance=1):
    """
    Simple 1D peak detection (MATLAB-style, Section II.F).

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Input 1D signal.
    min_height : float
        Minimum peak height (MPH).
    min_distance : int
        Minimum number of samples between consecutive peaks (MPD).

    Returns
    -------
    peaks : ndarray of ints
        Indices of detected local maxima.
    """
    signal = np.asarray(signal, dtype=float)
    N = signal.size
    if N < 3:
        return np.array([], dtype=int)

    peaks = []

    for i in range(1, N - 1):
        if signal[i] < min_height:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if peaks and (i - peaks[-1] < min_distance):
                # Keep the higher of two close peaks
                if signal[i] > signal[peaks[-1]]:
                    peaks[-1] = i
                continue
            peaks.append(i)

    return np.array(peaks, dtype=int)


def extract_gait_parameters(center_indices,
                            frame_times,
                            doppler_spectra,
                            velocity_axis,
                            range_resolution,
                            mph=0.5,
                            mpd=0.1):
    """
    Gait extraction algorithm (Section II.F, Fig. 11).

    Parameters
    ----------
    center_indices : ndarray, shape (num_frames, 2)
        Subject center indices (range_idx, azimuth_idx) for each frame,
        as returned by `association_tracking`.
    frame_times : ndarray, shape (num_frames,)
        Time stamp of each frame in seconds.
    doppler_spectra : ndarray, shape (num_frames, num_doppler_bins)
        Complex Doppler-FFT of the torso's range bin for each frame.
        Typically obtained by computing the Doppler FFT at the selected
        range index and combining over antennas.
    velocity_axis : ndarray, shape (num_doppler_bins,)
        Velocity value (m/s) corresponding to each Doppler bin.
    range_resolution : float
        Range resolution in meters per range bin.
    mph : float
        Minimum peak height for torso velocity (m/s). Default 0.5 m/s.
    mpd : float
        Minimum peak distance (s) between two steps. Default 0.1 s.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'frame_times'
        - 'range_positions' (torso trajectory)
        - 'speed_series' (per-frame torso speed from trajectory)
        - 'torso_velocity' (from Doppler)
        - 'step_indices'
        - 'step_times'
        - 'step_positions'
        - 'step_lengths'
        - 'step_intervals'
        - 'step_speeds'
        - 'stride_lengths'
        - 'stride_times'
        - 'stride_speeds'
        - 'overall_speed'
        - 'step_count'
    """
    center_indices = np.asarray(center_indices)
    frame_times = np.asarray(frame_times, dtype=float)
    doppler_spectra = np.asarray(doppler_spectra, dtype=complex)
    velocity_axis = np.asarray(velocity_axis, dtype=float)

    num_frames = center_indices.shape[0]
    if frame_times.shape[0] != num_frames:
        raise ValueError("frame_times length must match number of frames.")
    if doppler_spectra.shape[0] != num_frames:
        raise ValueError("doppler_spectra must have same number of frames as center_indices.")
    if doppler_spectra.shape[1] != velocity_axis.shape[0]:
        raise ValueError("velocity_axis length must match doppler_spectra.shape[1].")

    # Convert range indices to meters (torso trajectory)
    range_positions = center_indices[:, 0].astype(float) * float(range_resolution)

    # Per-frame torso speed from trajectory (finite difference)
    if num_frames >= 2:
        speed_series = np.gradient(range_positions, frame_times)
    else:
        speed_series = np.zeros_like(range_positions)

    # Torso velocity from Doppler spectra: take bin with maximum power
    power = np.abs(doppler_spectra) ** 2
    max_bin = np.argmax(power, axis=1)
    torso_velocity = np.abs(velocity_axis[max_bin])

    # Peak detection to find step times (foot contact)
    if num_frames >= 2:
        dt = np.median(np.diff(frame_times))
        if dt <= 0:
            dt = 1.0
    else:
        dt = 1.0
    mpd_samples = max(1, int(round(mpd / dt)))

    step_indices = find_peaks_1d(torso_velocity, min_height=mph, min_distance=mpd_samples)

    if step_indices.size == 0:
        return {
            "frame_times": frame_times,
            "range_positions": range_positions,
            "speed_series": speed_series,
            "torso_velocity": torso_velocity,
            "step_indices": step_indices,
            "step_times": np.array([], dtype=float),
            "step_positions": np.array([], dtype=float),
            "step_lengths": np.array([], dtype=float),
            "step_intervals": np.array([], dtype=float),
            "step_speeds": np.array([], dtype=float),
            "stride_lengths": np.array([], dtype=float),
            "stride_times": np.array([], dtype=float),
            "stride_speeds": np.array([], dtype=float),
            "overall_speed": 0.0,
            "step_count": 0,
        }

    step_times = frame_times[step_indices]
    step_positions = range_positions[step_indices]
    step_count = step_indices.size

    # Step-to-step quantities
    if step_count >= 2:
        step_lengths = np.diff(step_positions)
        step_intervals = np.diff(step_times)
        step_speeds = step_lengths / step_intervals
    else:
        step_lengths = np.array([], dtype=float)
        step_intervals = np.array([], dtype=float)
        step_speeds = np.array([], dtype=float)

    # Stride quantities (two consecutive steps)
    if step_count >= 3:
        stride_lengths = step_positions[2:] - step_positions[:-2]
        stride_times = step_times[2:] - step_times[:-2]
        stride_speeds = stride_lengths / stride_times
    else:
        stride_lengths = np.array([], dtype=float)
        stride_times = np.array([], dtype=float)
        stride_speeds = np.array([], dtype=float)

    # Overall average speed between first and last detected step
    if step_count >= 2:
        total_distance = step_positions[-1] - step_positions[0]
        total_time = step_times[-1] - step_times[0]
        overall_speed = total_distance / total_time if total_time > 0 else 0.0
    else:
        overall_speed = 0.0

    results = {
        "frame_times": frame_times,
        "range_positions": range_positions,
        "speed_series": speed_series,
        "torso_velocity": torso_velocity,
        "step_indices": step_indices,
        "step_times": step_times,
        "step_positions": step_positions,
        "step_lengths": step_lengths,
        "step_intervals": step_intervals,
        "step_speeds": step_speeds,
        "stride_lengths": stride_lengths,
        "stride_times": stride_times,
        "stride_speeds": stride_speeds,
        "overall_speed": overall_speed,
        "step_count": int(step_count),
    }
    return results
