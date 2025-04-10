from glob import glob
from typing import Final

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    time = dataset["Time (s)"].to_numpy
    return current, voltage, time, metadata


def plot_iv(current, voltage, field_value, angle_value):
    plt.plot(
        current,
        voltage,
        linestyle="-",
        label=f"Angle {angle_value}° | Field {field_value} T",
    )
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    return field_value, angle_value


def find_critical_current(current, voltage, sample_length_m=2e-3, criterion_uvm=100):
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


def main() -> None:

    TRANSITION_CRITERION: Final = 100  # in units of µV⋅m^-1

    filepath_match = "data/2mm*field45angle.txt"
    all_filepaths = glob(filepath_match)
    all_filepaths.sort()
    print("Found files:\n", all_filepaths, "\n")

    ic_values = []

    for data_filepath in all_filepaths:
        current, voltage, time, metadata = extract_values(data_filepath)
        field_value = float(metadata.get("field / T", "0"))
        angle_value = float(metadata.get("Angle (deg.)", "0"))
        ic = find_critical_current(
            current, voltage, sample_length_m=2e-3, criterion_uvm=TRANSITION_CRITERION
        )
        ic_values.append((field_value, angle_value, ic))

        plot_iv(current, voltage, field_value, angle_value)

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
