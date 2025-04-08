from glob import glob

import matplotlib
import matplotlib.pyplot as plt
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


def plot_iv(current, voltage, time, metadata, data_filepath):

    current, voltage, time, metadata = extract_values(data_filepath)

    field_value = round(float(metadata.get("field / T", "N/A")), 2)
    angle_value = int(round(float(metadata.get("Angle (deg.)", "N/A")), 0))

    # Plot the I-V curve
    plt.plot(
        current,
        voltage,
        linestyle="-",
        label=f"Angle {str(angle_value)} (degrees), Field {str(field_value)} (T)",
    )
    plt.xlabel("Current (A)")
    plt.ylabel("Voltage (uV)")
    plt.title(f"I-V Curve at {str(angle_value)} degrees angle")
    return field_value, angle_value


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


def main() -> None:

    # TRANSITION_CRITERION: Final = 100
    # TRANSITION_CRITERION is in units of µV⋅m^-1
    # It is the electric field criterion often denoted E_C.

    filepath_match = "data/2mm*field45angle.txt"
    all_filepaths = glob(filepath_match)
    all_filepaths.sort()
    print(all_filepaths)

    # for index, data_filepath in enumerate(all_filepaths):
    #     current, voltage, time, metadata = extract_values(data_filepath)
    #     plot_iv(current, voltage, time, metadata, data_filepath)
    # electric_field = TRANSITION_CRITERION * (
    # (current / critical_current) ** exponent
    # )
    # E = Ec * (J/Jc)^n
    # if index == len(all_filepaths):
    # _, angle_value = plot_iv(current, voltage, time, metadata)
    # save_filename = f"images/{str(angle_value)}angle.png"
    plt.legend()
    plt.tight_layout()
    # plt.savefig(save_filename, dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
