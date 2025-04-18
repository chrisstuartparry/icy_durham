from glob import glob
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

matplotlib.rcParams["figure.figsize"] = 16, 10


def extract_metadata(data_filepath):
    metadata = {}
    header_index = None
    with open(data_filepath, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Current" in line:
            header_index = i + 1
            break
        if line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                metadata[parts[0].strip()] = parts[1].strip()
    return metadata, header_index


def extract_values(data_filepath):
    metadata, header_index = extract_metadata(data_filepath)

    dataset = pd.read_csv(
        data_filepath,
        sep=r"\s+",
        skiprows=header_index,
        names=["Current (A)", "Voltage (uV)", "Time (s)"],
    )
    current = dataset["Current (A)"].to_numpy()
    voltage = dataset["Voltage (uV)"].to_numpy()
    time = dataset["Time (s)"].to_numpy()
    return current, voltage, time, metadata


def plot_iv(current, voltage, field_value, angle_value, ic):
    (iv_curve,) = plt.plot(
        current,
        voltage,
        linestyle="-",
        label=f"Angle {angle_value}° | Field {field_value} T",
    )
    if ic:
        v_at_ic = np.interp(ic, current, voltage)
        plt.axhline(y=v_at_ic, linestyle="--", color=iv_curve.get_color())
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    return field_value, angle_value


def find_critical_current(current, voltage, sample_length_m, criterion_uvm):
    # Convert voltage (µV) to electric field (µV/m)
    e_data = voltage / sample_length_m
    # Find where e_data crosses criterion_uvm
    # Sort so we can use np.interp safely
    sort_idx = np.argsort(e_data)
    e_sorted = e_data[sort_idx]
    i_sorted = current[sort_idx]
    if criterion_uvm < e_sorted[0]:
        return i_sorted[0]  # If criterion is below the first point
    if criterion_uvm > e_sorted[-1]:
        return i_sorted[-1]  # If criterion is above the last point
    # Linear interpolation
    ic = np.interp(criterion_uvm, e_sorted, i_sorted)
    return ic


def find_background_region(current, voltage, slope_threshold=0.2, smooth=True):
    """
    Returns (voltage_smoothed, start_idx, end_idx) for the region over which dV/dI is
    below slope_threshold, indicating mostly 'linear' background.

    slope_threshold must be tuned to your data units—e.g. µV/A.
    """
    if smooth:
        voltage_smoothed = savgol_filter(voltage, window_length=30, polyorder=3)
        print("Using smoothed voltage!")
        print(f"Original voltage: {voltage[:5]}\n")
        print(f"Smoothed voltage: {voltage_smoothed[:5]}\n")
    else:
        voltage_smoothed = voltage
        print("Not using smoothed voltage!")
    # Numerical derivative:
    dVdI: npt.ArrayLike | tuple[npt.ArrayLike] = np.gradient(voltage_smoothed, current)

    # We'll assume the background region starts at index = 0
    # and continues as long as |dV/dI| < slope_threshold.
    # Once slope exceeds threshold, we consider it the onset of exponential rise.
    valid_mask = np.abs(dVdI) < slope_threshold

    # Find the first index where the slope fails (goes above threshold)
    # np.argmax returns the first True in ~valid_mask (equivalent to the first False in valid_mask)
    first_fail = np.argmax(~valid_mask)

    # If argmax finds no True in ~valid_mask (meaning all are valid),
    # 'first_fail' might be 0 if the slope is never above threshold
    # or it might be 0 if the very first point is invalid.
    print(f"valid_mask is {valid_mask} \n with shape: {valid_mask.shape}")
    print(f"np.any(valid_mask) is: {np.any(valid_mask)}")
    if first_fail == 0 and valid_mask[0] is False:
        # Means the slope at the very first point is already above threshold
        # => no real "background" region found
        print("Slope at very first point is already above threshold!")
        return voltage_smoothed, 0, 0
    if first_fail == 0 and valid_mask[0] is True:
        # Means no slope ever exceeded threshold, entire dataset is "background" (asterisk)
        print("Slope never exceeded threshold!")
        return voltage_smoothed, 0, len(voltage)
    print(f"First fail at index: {first_fail}")
    return voltage_smoothed, 0, first_fail


def find_background_region_adaptive(
    current, voltage, percentile=75, smooth=True, window_length=30, polyorder=3
):
    """
    Uses an adaptive threshold based on the data's slope distribution.

    Parameters:
    - percentile: The percentile of absolute slopes to use as threshold (e.g., 75 means
      75% of points with the lowest slopes are considered "background")
    - smooth: Whether to smooth the voltage data
    - window_length: Window length for Savitzky-Golay filter
    - polyorder: Polynomial order for Savitzky-Golay filter
    """
    import numpy as np
    from scipy.signal import savgol_filter

    if smooth:
        voltage_smoothed = savgol_filter(
            voltage, window_length=window_length, polyorder=polyorder
        )
    else:
        voltage_smoothed = voltage

    # Numerical derivative:
    dVdI = np.gradient(voltage_smoothed, current)

    # Calculate adaptive threshold based on percentile of absolute derivatives
    abs_dVdI = np.abs(dVdI)
    threshold = np.percentile(abs_dVdI, percentile)
    print(f"Adaptive threshold (percentile {percentile}): {threshold:.4f}")

    # Find points below threshold
    valid_mask = abs_dVdI < threshold

    # Find the first index where the slope exceeds threshold
    first_fail = np.argmax(~valid_mask)

    # Handle edge cases
    if first_fail == 0 and not valid_mask[0]:
        print("No background region found (all slopes above threshold)")
        return voltage_smoothed, 0, 0
    if first_fail == 0 and valid_mask[0]:
        print("All data points considered background")
        return voltage_smoothed, 0, len(voltage)

    return voltage_smoothed, 0, first_fail


def find_background_region_robust(
    current,
    voltage,
    skip_first_n,
    slope_threshold=None,
    min_region_size=10,
    smooth=True,
    percentile=None,
    window_length=30,
    polyorder=3,
):
    """
    A more robust background region detection that:
    1. Doesn't assume background starts at index 0
    2. Finds the longest continuous segment below threshold
    3. Requires a minimum number of consecutive points to detect a transition
    4. Can use either a fixed slope_threshold or derive it from data percentile

    Parameters:
    - skip_first_n: Number of points to skip at the start of the arrays
    - slope_threshold: Fixed threshold for slope. If None, uses percentile method
    - min_region_size: Minimum number of points required for a valid background region
    - smooth: Whether to smooth the voltage data
    - percentile: If slope_threshold is None, use this percentile to derive threshold
    - window_length, polyorder: Parameters for Savitzky-Golay filter
    """
    import numpy as np
    from scipy.signal import savgol_filter

    if smooth:
        voltage_smoothed = savgol_filter(
            voltage, window_length=window_length, polyorder=polyorder
        )
    else:
        voltage_smoothed = voltage

    current_subset = current[skip_first_n:]
    voltage_subset = voltage_smoothed[skip_first_n:]

    # Numerical derivative:
    # dVdI = np.gradient(voltage_smoothed, current)
    dVdI = np.gradient(voltage_subset, current_subset)
    abs_dVdI = np.abs(dVdI)

    # Determine threshold
    if slope_threshold is None:
        if percentile is None:
            percentile = 75  # Default percentile
        slope_threshold = np.percentile(abs_dVdI, percentile)
        print(
            f"Using adaptive threshold (percentile {percentile}): {slope_threshold:.4f}"
        )
    else:
        print(f"Using fixed threshold: {slope_threshold:.4f}")

    # Smooth the derivative to reduce noise impact
    smoothed_dVdI = savgol_filter(
        abs_dVdI,
        window_length=max(5, min(15, len(abs_dVdI) // 10)),
        polyorder=min(2, polyorder),
    )

    # Find all points with slope below threshold
    valid_points = smoothed_dVdI < slope_threshold

    # Find segments of valid points
    segments = []
    current_segment = []

    for i, valid in enumerate(valid_points):
        if valid:
            current_segment.append(i)
        elif current_segment:  # End of a segment
            segments.append(current_segment)
            current_segment = []

    # Don't forget the last segment if it exists
    if current_segment:
        segments.append(current_segment)

    # Filter segments by minimum size
    valid_segments = [s for s in segments if len(s) >= min_region_size]

    if not valid_segments:
        print(f"No valid background segments found (min size: {min_region_size})")
        return voltage_smoothed, 0, 0

    # Find the longest segment
    longest_segment = max(valid_segments, key=len)

    bg_start = longest_segment[0] + skip_first_n
    bg_end = (
        longest_segment[-1] + 1 + skip_first_n
    )  # +1 because Python slicing is exclusive of end

    print(
        f"Found background region from index {bg_start} to {bg_end} ({bg_end-bg_start} points)"
    )

    return voltage_smoothed, bg_start, bg_end


def remove_linear_background(
    current, voltage, slope_threshold=0.2, smooth=True, smoothed_plot=True
):
    """
    1) Identify background region using derivative threshold
    2) Fit a line only in that region
    3) Subtract the fit from the entire voltage array

    Returns (voltage_corrected, slope, intercept, bg_end_idx)
    """
    voltage_smoothed, i_start, i_end = find_background_region(
        current, voltage, slope_threshold, smooth
    )

    if i_end <= i_start:
        # No valid region identified, just return original data
        print(" No valid region identified, returning original data!")
        return voltage_smoothed, None, None, (i_start, i_end)

    # Fit a line to the 'background' portion
    m, c = np.polyfit(current[i_start:i_end], voltage[i_start:i_end], 1)

    # Subtract from the entire voltage array
    voltage_corrected = voltage_smoothed - (m * current + c)
    return voltage_corrected, m, c, (i_start, i_end)


def remove_linear_background_improved(current, voltage, method="robust", **kwargs):
    """
    Improved background removal with multiple detection methods

    Parameters:
    - method: 'original', 'adaptive', or 'robust'
    - **kwargs: Parameters passed to the chosen method

    Returns:
    - voltage_corrected, slope, intercept, (bg_start, bg_end)
    """
    import numpy as np

    # Choose the background detection method
    if method == "original":
        from scipy.signal import savgol_filter

        smooth = kwargs.get("smooth", True)
        slope_threshold = kwargs.get("slope_threshold", 0.2)

        if smooth:
            voltage_smoothed = savgol_filter(voltage, window_length=30, polyorder=3)
        else:
            voltage_smoothed = voltage

        # Numerical derivative:
        dVdI = np.gradient(voltage_smoothed, current)

        # We'll assume the background region starts at index = 0
        # and continues as long as |dV/dI| < slope_threshold.
        valid_mask = np.abs(dVdI) < slope_threshold

        # Find the first index where the slope fails (goes above threshold)
        first_fail = np.argmax(~valid_mask)

        if first_fail == 0 and not valid_mask[0]:
            # No valid region
            voltage_smoothed, i_start, i_end = voltage_smoothed, 0, 0
        elif first_fail == 0 and valid_mask[0]:
            # All valid
            voltage_smoothed, i_start, i_end = voltage_smoothed, 0, len(voltage)
        else:
            voltage_smoothed, i_start, i_end = voltage_smoothed, 0, first_fail

    elif method == "adaptive":
        voltage_smoothed, i_start, i_end = find_background_region_adaptive(
            current, voltage, **kwargs
        )

    elif method == "robust":
        voltage_smoothed, i_start, i_end = find_background_region_robust(
            current, voltage, skip_first_n=10, **kwargs
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    if i_end <= i_start:
        # No valid region identified, just return original data
        print("No valid background region identified, returning original data!")
        return voltage_smoothed, None, None, (i_start, i_end)

    # Fit a line to the 'background' portion
    m, c = np.polyfit(current[i_start:i_end], voltage[i_start:i_end], 1)
    print(f"Fitted line: y = {m:.4f}x + {c:.4f}")

    # Subtract from the entire voltage array
    voltage_corrected = voltage_smoothed - (m * current + c)
    return voltage_corrected, m, c, (i_start, i_end)


def visualize_background_detection(
    current,
    voltage,
    bg_start,
    bg_end,
    m=None,
    c=None,
    corrected=None,
    data_filepath=None,
):
    """
    Visualizes the detected background region and the linear fit.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    # Plot the original I-V curve
    plt.subplot(2, 1, 1)
    plt.plot(current, voltage, "b.-", label="Original Data")

    # Highlight the background region
    plt.plot(
        current[bg_start:bg_end],
        voltage[bg_start:bg_end],
        "r.",
        label=f"Background Region ({bg_end-bg_start} points)",
    )

    # If a linear fit was performed, plot it
    if m is not None and c is not None:
        fit_line = m * current + c
        plt.plot(current, fit_line, "g--", label=f"Linear Fit (slope={m:.4f})")

    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    if data_filepath is not None:
        plt.title(f"Background Region Detection for {data_filepath}")
    else:
        plt.title("Background Region Detection")
    plt.legend()
    plt.grid(True)

    # Plot the corrected data if available
    if corrected is not None:
        plt.subplot(2, 1, 2)
        plt.plot(current, corrected, "b.-", label="Corrected Data")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        plt.xlabel("Current (A)")
        plt.ylabel("Voltage (uV) - Corrected")
        plt.title("Background-Subtracted Data")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_threshold_sensitivity(
    current, voltage, threshold_values, smooth=True, data_filepath=None
):
    """
    Analyzes how sensitive the background detection is to different threshold values.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    for i, threshold in enumerate(threshold_values):
        voltage_corrected, m, c, (bg_start, bg_end) = remove_linear_background_improved(
            current,
            voltage,
            method="original",
            slope_threshold=threshold,
            smooth=smooth,
        )

        plt.subplot(2, len(threshold_values), i + 1)
        plt.plot(current, voltage, "b-", alpha=0.5, label="Original")
        plt.plot(
            current[bg_start:bg_end],
            voltage[bg_start:bg_end],
            "r.",
            label=f"BG ({bg_end-bg_start} pts)",
        )

        if m is not None and c is not None:
            fit_line = m * current + c
            plt.plot(current, fit_line, "g--", label=f"Fit (m={m:.2f})")

        plt.title(f"Threshold = {threshold}")
        plt.legend(loc="upper left", fontsize="small")
        plt.grid(True)

        plt.subplot(2, len(threshold_values), i + 1 + len(threshold_values))
        if voltage_corrected is not None:
            plt.plot(current, voltage_corrected, "b-")
        else:
            plt.text(
                0.5,
                0.5,
                "No correction applied",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )
        plt.title("Corrected Data")
        plt.grid(True)

    if data_filepath is not None:
        plt.suptitle(f"Threshold Sensitivity Analysis for {data_filepath}")

    plt.tight_layout()
    plt.show()


def visualize_slope_distribution(current, voltage, smooth=True, data_filepath=None):
    """
    Visualizes the distribution of slopes in the data to help identify
    an appropriate threshold.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import savgol_filter

    if smooth:
        voltage_smoothed = savgol_filter(voltage, window_length=30, polyorder=3)
    else:
        voltage_smoothed = voltage

    # Numerical derivative:
    dVdI = np.gradient(voltage_smoothed, current)
    abs_dVdI = np.abs(dVdI)

    plt.figure(figsize=(12, 8))

    # Histogram of absolute slopes
    plt.subplot(2, 1, 1)
    plt.hist(abs_dVdI, bins=50)
    plt.xlabel("Absolute Slope (|dV/dI|)")
    plt.ylabel("Frequency")
    if data_filepath is not None:
        plt.title(f"Distribution of Absolute Slopes for {data_filepath}")
    else:
        plt.title("Distribution of Absolute Slopes")
    plt.axvline(x=48.4, color="r", linestyle="--", label="Critical threshold (~48.4)")
    plt.legend()
    plt.grid(True)

    # Cumulative distribution
    plt.subplot(2, 1, 2)
    sorted_slopes = np.sort(abs_dVdI)
    plt.plot(sorted_slopes, np.arange(len(sorted_slopes)) / len(sorted_slopes))
    plt.xlabel("Absolute Slope (|dV/dI|)")
    plt.ylabel("Cumulative Fraction")
    if data_filepath is not None:
        plt.title(f"Cumulative Distribution of Absolute Slopes for {data_filepath}")
    else:
        plt.title("Cumulative Distribution of Absolute Slopes")
    plt.axvline(x=48.4, color="r", linestyle="--", label="Critical threshold (~48.4)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Min slope: {np.min(abs_dVdI):.4f}")
    print(f"Max slope: {np.max(abs_dVdI):.4f}")
    print(f"Mean slope: {np.mean(abs_dVdI):.4f}")
    print(f"Median slope: {np.median(abs_dVdI):.4f}")
    print(f"Standard deviation: {np.std(abs_dVdI):.4f}")

    # Print percentiles around the critical threshold
    print("\nPercentiles:")
    for p in range(45, 56):
        percentile = np.percentile(abs_dVdI, p)
        print(f"{p}th percentile: {percentile:.4f}")


def diagnose_dataset(current, voltage, time=None, dataset_name=""):
    """
    Diagnoses issues with the dataset that might cause problems with gradient calculations
    and background subtraction.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n=== DIAGNOSING DATASET: {dataset_name} ===")
    print(f"With first 10 currents of: {current[:10]}")
    print(f"With first 10 voltages of: {voltage[:10]}")
    # Check for NaN or infinite values
    print(f"NaN in current: {np.isnan(current).any()}")
    print(f"NaN in voltage: {np.isnan(voltage).any()}")
    print(f"Inf in current: {np.isinf(current).any()}")
    print(f"Inf in voltage: {np.isinf(voltage).any()}")

    # Check for duplicate x-values
    unique_current = np.unique(current)
    duplicates = len(current) - len(unique_current)
    print(f"Duplicate current values: {duplicates}")

    if duplicates > 0:
        # Find where duplicates occur
        from collections import Counter

        value_counts = Counter(current)
        print("Current values with multiple occurrences:")
        for value, count in value_counts.most_common():
            if count > 1:
                print(f"  {value}: {count} times")

    # Check for very small spacings
    current_sorted = np.sort(current)
    differences = np.diff(current_sorted)
    min_diff = np.min(differences)
    print(f"Minimum difference between current values: {min_diff}")

    if min_diff < 1e-10:
        # Find where very small differences occur
        small_diff_indices = np.where(differences < 1e-10)[0]
        print(f"Very small differences at indices: {small_diff_indices}")
        print("Current values at these points:")
        for idx in small_diff_indices:
            print(
                f"  {current_sorted[idx]} and {current_sorted[idx+1]}, diff={current_sorted[idx+1]-current_sorted[idx]}"
            )

    # Plot the data for visual inspection
    plt.figure(figsize=(12, 10))

    # Raw data plot
    plt.subplot(2, 2, 1)
    plt.plot(current, voltage, ".-")
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    plt.title("Raw I-V Curve")
    plt.grid(True)

    # Sorted data plot
    sort_idx = np.argsort(current)
    plt.subplot(2, 2, 2)
    plt.plot(current[sort_idx], voltage[sort_idx], ".-")
    plt.xlabel("Current (A) - Sorted")
    plt.ylabel("Voltage (uV)")
    plt.title("Sorted I-V Curve")
    plt.grid(True)

    # Current vs index
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(len(current)), current, ".-")
    plt.xlabel("Index")
    plt.ylabel("Current (A)")
    plt.title("Current vs Index")
    plt.grid(True)

    # Differences in current
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(len(differences)), differences, ".-")
    plt.xlabel("Index")
    plt.ylabel("Difference in Current")
    plt.title("Current Differences")
    plt.yscale("log")  # Log scale to highlight small differences
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return duplicates > 0 or min_diff < 1e-10


def preprocess_data(current, voltage, min_spacing=1e-10, handle_duplicates="average"):
    """
    Preprocesses the data to handle issues that can cause gradient problems:
    1. Removes duplicate current values by averaging, taking first/last, or similar
    2. Ensures minimum spacing between current values

    Parameters:
    - min_spacing: Minimum allowed spacing between current values
    - handle_duplicates: How to handle duplicate current values
      'average' - average the voltage values
      'first' - take the first occurrence
      'last' - take the last occurrence
    """
    import numpy as np

    print("\n=== PREPROCESSING DATA ===")
    original_length = len(current)

    # Sort data by current
    sort_idx = np.argsort(current)
    current_sorted = current[sort_idx]
    voltage_sorted = voltage[sort_idx]

    # Find duplicate current values
    unique_current, unique_indices, counts = np.unique(
        current_sorted, return_index=True, return_counts=True
    )
    has_duplicates = np.any(counts > 1)

    if has_duplicates:
        print(f"Found {np.sum(counts > 1)} unique current values with duplicates")

        # Create new arrays without duplicates
        preprocessed_current = []
        preprocessed_voltage = []

        # Group by current value
        i = 0
        while i < len(current_sorted):
            c = current_sorted[i]
            # Find all indices with this current value
            same_current_indices = np.where(current_sorted == c)[0]

            if len(same_current_indices) > 1:
                # Get all voltage values for this current
                v_values = voltage_sorted[same_current_indices]

                # Handle according to strategy
                if handle_duplicates == "average":
                    v = np.mean(v_values)
                elif handle_duplicates == "first":
                    v = v_values[0]
                elif handle_duplicates == "last":
                    v = v_values[-1]
                else:
                    raise ValueError(
                        f"Unknown duplicate handling method: {handle_duplicates}"
                    )

                preprocessed_current.append(c)
                preprocessed_voltage.append(v)

                # Skip all these indices
                i = same_current_indices[-1] + 1
            else:
                # No duplicates for this value
                preprocessed_current.append(c)
                preprocessed_voltage.append(voltage_sorted[i])
                i += 1

        # Convert back to numpy arrays
        preprocessed_current = np.array(preprocessed_current)
        preprocessed_voltage = np.array(preprocessed_voltage)
    else:
        # No duplicates, just use sorted arrays
        preprocessed_current = current_sorted
        preprocessed_voltage = voltage_sorted

    # Check for minimum spacing
    differences = np.diff(preprocessed_current)
    too_close = differences < min_spacing

    if np.any(too_close):
        print(
            f"Found {np.sum(too_close)} pairs of points too close together (< {min_spacing})"
        )

        # Create new arrays with proper spacing
        final_current = [preprocessed_current[0]]
        final_voltage = [preprocessed_voltage[0]]

        for i in range(1, len(preprocessed_current)):
            if preprocessed_current[i] - final_current[-1] >= min_spacing:
                final_current.append(preprocessed_current[i])
                final_voltage.append(preprocessed_voltage[i])

        preprocessed_current = np.array(final_current)
        preprocessed_voltage = np.array(final_voltage)

    print(f"Original data points: {original_length}")
    print(f"After preprocessing: {len(preprocessed_current)}")

    return preprocessed_current, preprocessed_voltage


def find_background_region_safe(
    current,
    voltage,
    slope_threshold=None,
    min_region_size=10,
    percentile=85,
    window_length=30,
    polyorder=3,
    smooth=True,
):
    """
    A safer version of the background region detection that handles numerical issues.
    """
    import numpy as np
    from scipy.signal import savgol_filter

    # First, validate and preprocess the data
    has_issues = diagnose_dataset(current, voltage, dataset_name="Before preprocessing")

    if has_issues:
        # Preprocess to handle problematic data
        current, voltage = preprocess_data(
            current, voltage, min_spacing=1e-10, handle_duplicates="average"
        )
        # Re-check
        diagnose_dataset(current, voltage, dataset_name="After preprocessing")

    # Now the data should be clean for gradient calculations
    if smooth:
        try:
            voltage_smoothed = savgol_filter(
                voltage,
                window_length=min(window_length, len(voltage) - 1),
                polyorder=min(polyorder, min(window_length, len(voltage) - 1) - 1),
            )
        except Exception as e:
            print(f"Error in Savitzky-Golay filter: {e}")
            print("Falling back to simple moving average smoothing")
            # Simple moving average as fallback
            window = min(11, len(voltage) // 5)
            if window % 2 == 0:
                window += 1  # Make it odd
            kernel = np.ones(window) / window
            voltage_smoothed = np.convolve(voltage, kernel, mode="same")
    else:
        voltage_smoothed = voltage

    # Safely calculate numerical derivative
    try:
        # Check if current has proper spacing for gradient calculation
        if len(np.unique(current)) < len(current) * 0.9:  # If more than 10% duplicates
            print(
                "WARNING: Many duplicate current values, using rank instead of actual values for gradient"
            )
            # Use point indices instead of current values
            x_for_gradient = np.arange(len(current))
            dVdI = np.gradient(voltage_smoothed, x_for_gradient)

            # Scale dVdI based on average current spacing to get appropriate units
            avg_current_diff = (current.max() - current.min()) / (len(current) - 1)
            dVdI = dVdI / avg_current_diff
        else:
            dVdI = np.gradient(voltage_smoothed, current)

        # Check for NaN/inf values in derivative
        if np.isnan(dVdI).any() or np.isinf(dVdI).any():
            print("WARNING: NaN/Inf in derivative, replacing with interpolated values")
            # Create a mask of valid values
            valid_mask = ~(np.isnan(dVdI) | np.isinf(dVdI))
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) < 2:
                print("ERROR: Not enough valid derivative points")
                return voltage_smoothed, 0, 0

            # Interpolate the invalid values
            invalid_indices = np.where(~valid_mask)[0]
            dVdI[invalid_indices] = np.interp(
                invalid_indices, valid_indices, dVdI[valid_indices]
            )

    except Exception as e:
        print(f"ERROR in gradient calculation: {e}")
        return voltage_smoothed, 0, 0

    # Determine threshold
    abs_dVdI = np.abs(dVdI)
    if slope_threshold is None:
        try:
            slope_threshold = np.percentile(abs_dVdI, percentile)
            print(
                f"Using adaptive threshold (percentile {percentile}): {slope_threshold:.4f}"
            )
        except Exception as e:
            print(f"ERROR in percentile calculation: {e}")
            print("Using default threshold of 1.0")
            slope_threshold = 1.0

    # Find points with slope below threshold
    valid_points = abs_dVdI < slope_threshold

    # Visualize
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(current, voltage, "b-", label="Original")
    plt.plot(current, voltage_smoothed, "g-", label="Smoothed")
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(current, abs_dVdI, "r-", label="|dV/dI|")
    plt.axhline(
        y=slope_threshold,
        color="k",
        linestyle="--",
        label=f"Threshold: {slope_threshold:.4f}",
    )
    plt.scatter(
        current[valid_points],
        abs_dVdI[valid_points],
        c="g",
        s=20,
        alpha=0.7,
        label="Below Threshold",
    )
    plt.xlabel("Current (A)")
    plt.ylabel("|dV/dI|")
    plt.yscale("log")  # Log scale to better see the threshold
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Find segments of valid points
    segments = []
    current_segment = []

    for i, valid in enumerate(valid_points):
        if valid:
            current_segment.append(i)
        elif current_segment:  # End of a segment
            segments.append(current_segment)
            current_segment = []

    # Don't forget the last segment if it exists
    if current_segment:
        segments.append(current_segment)

    # Filter segments by minimum size
    valid_segments = [s for s in segments if len(s) >= min_region_size]

    if not valid_segments:
        print(f"No valid background segments found (min size: {min_region_size})")

        # Last resort: try with a much higher threshold
        emergency_threshold = np.percentile(abs_dVdI, 95)
        print(
            f"Trying emergency threshold (95th percentile): {emergency_threshold:.4f}"
        )

        valid_points_emergency = abs_dVdI < emergency_threshold
        # Quick check if this gives us anything useful
        if np.sum(valid_points_emergency) >= min_region_size:
            print(
                f"Emergency threshold found {np.sum(valid_points_emergency)} valid points"
            )
            # Find the longest continuous segment
            segments_emergency = []
            current_segment = []

            for i, valid in enumerate(valid_points_emergency):
                if valid:
                    current_segment.append(i)
                elif current_segment:  # End of a segment
                    segments_emergency.append(current_segment)
                    current_segment = []

            if current_segment:
                segments_emergency.append(current_segment)

            valid_segments_emergency = [
                s for s in segments_emergency if len(s) >= min_region_size
            ]

            if valid_segments_emergency:
                print("Using emergency threshold for background region")
                valid_segments = valid_segments_emergency
            else:
                return voltage_smoothed, 0, 0
        else:
            return voltage_smoothed, 0, 0

    # Find the longest segment
    longest_segment = max(valid_segments, key=len)

    bg_start = longest_segment[0]
    bg_end = longest_segment[-1] + 1  # +1 because Python slicing is exclusive of end

    print(
        f"Found background region from index {bg_start} to {bg_end} ({bg_end-bg_start} points)"
    )

    return voltage_smoothed, bg_start, bg_end


def remove_linear_background_safe(current, voltage, **kwargs):
    """
    Safer version of background removal that handles numerical issues
    """
    import numpy as np

    voltage_smoothed, bg_start, bg_end = find_background_region_safe(
        current, voltage, **kwargs
    )

    if bg_end <= bg_start:
        # No valid region identified, just return original data
        print("No valid background region identified, returning original data!")
        return voltage_smoothed, None, None, (bg_start, bg_end)

    # Fit a line to the 'background' portion
    try:
        m, c = np.polyfit(current[bg_start:bg_end], voltage[bg_start:bg_end], 1)
        print(f"Fitted line: y = {m:.4f}x + {c:.4f}")
    except Exception as e:
        print(f"ERROR in linear fit: {e}")
        return voltage_smoothed, None, None, (bg_start, bg_end)

    # Subtract from the entire voltage array
    voltage_corrected = voltage_smoothed - (m * current + c)

    # Visualize the result
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(current, voltage, "b-", alpha=0.5, label="Original")
    plt.plot(
        current[bg_start:bg_end],
        voltage[bg_start:bg_end],
        "r.",
        label=f"Background ({bg_end-bg_start} pts)",
    )
    plt.plot(current, m * current + c, "g--", label=f"Fit: y = {m:.4f}x + {c:.4f}")
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    plt.title("Background Detection and Linear Fit")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(current, voltage_corrected, "b-", label="Corrected Data")
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV) - Corrected")
    plt.title("Background-Subtracted Data")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return voltage_corrected, m, c, (bg_start, bg_end)


def main() -> None:

    TRANSITION_CRITERION: Final = 10  # in units of µV⋅m^-1

    filepath_match = "data/2mm*field45angle.txt"
    all_filepaths = glob(filepath_match)
    all_filepaths.sort()
    print("Found files:\n", all_filepaths, "\n")

    ic_values = []
    final_plot_values = []
    for data_filepath in all_filepaths:
        print(f"\nProcessing '{data_filepath}':\n")
        current, voltage, time, metadata = extract_values(data_filepath)
        field_value = float(metadata.get("field / T", "0"))
        angle_value = float(metadata.get("Angle (deg.)", "0"))

        # visualize_slope_distribution(current, voltage, smooth=True)

        voltage_corrected, m, c, (bg_start, bg_end) = remove_linear_background_improved(
            current,
            voltage,
            method="robust",
            slope_threshold=None,
            percentile=85,
            min_region_size=10,
            smooth=True,
        )

        # Visualize the background detection
        visualize_background_detection(
            current,
            voltage,
            bg_start,
            bg_end,
            m,
            c,
            voltage_corrected,
            data_filepath=data_filepath,
        )

        # # Test different threshold values
        # analyze_threshold_sensitivity(
        #     current,
        #     voltage,
        #     threshold_values=[45.0, 47.0, 48.0, 48.3, 48.4, 48.5, 49.0],
        #     smooth=True,
        # )
        if voltage_corrected is not None:
            ic = find_critical_current(
                current,
                voltage_corrected,
                sample_length_m=2e-3,
                criterion_uvm=TRANSITION_CRITERION,
            )
            ic_values.append((field_value, angle_value, ic))

            final_plot_values.append(
                (current, voltage_corrected, field_value, angle_value, ic)
            )

        else:
            print(f"WARNING: Could not process {data_filepath}")

    save_filename = f"images/{str(angle_value)}angle_new.png"

    for final_plot_value in final_plot_values:
        plot_iv(*final_plot_value)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_filename, dpi=100)
    plt.show()

    print("Critical Currents:\n")
    for f, a, ic in ic_values:
        print(f"Field={f} T,\t Angle={a}°, \t Ic={ic:.3f} A")


if __name__ == "__main__":
    main()
