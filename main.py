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


def main() -> None:

    TRANSITION_CRITERION: Final = 10  # in units of µV⋅m^-1

    filepath_match = "data/2mm*field45angle.txt"
    all_filepaths = glob(filepath_match)
    all_filepaths.sort()
    print("Found files:\n", all_filepaths, "\n")

    ic_values = []

    for data_filepath in all_filepaths:
        print(f"\nStarting filepath '{data_filepath}':\n")
        current, voltage, time, metadata = extract_values(data_filepath)
        field_value = float(metadata.get("field / T", "0"))
        angle_value = float(metadata.get("Angle (deg.)", "0"))
        voltage_corrected, m, c, (bg_start, bg_end) = remove_linear_background(
            current, voltage, slope_threshold=48.3, smooth=True
        )
        ic = find_critical_current(
            current,
            voltage_corrected,
            sample_length_m=2e-3,
            criterion_uvm=TRANSITION_CRITERION,
        )
        ic_values.append((field_value, angle_value, ic))

        plot_iv(current, voltage_corrected, field_value, angle_value, ic)

    save_filename = f"images/{str(angle_value)}angle.png"

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_filename, dpi=100)
    plt.show()

    print("Critical Currents:\n")
    for f, a, ic in ic_values:
        print(f"Field={f} T,\t Angle={a}°, \t Ic={ic:.3f} A")


if __name__ == "__main__":
    main()
