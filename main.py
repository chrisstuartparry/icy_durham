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

    # Numerical derivative:
    dVdI = np.gradient(voltage_smoothed, current)
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

    bg_start = longest_segment[0]
    bg_end = longest_segment[-1] + 1  # +1 because Python slicing is exclusive of end

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
            current, voltage, **kwargs
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
