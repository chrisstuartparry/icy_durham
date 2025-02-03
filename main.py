import matplotlib.pyplot as plt
import pandas as pd
from glob import glob


def main():

    data_filepath = "/Users/csp/repositories/icy_durham/data/2mm0field0angle.txt"

    header_row_index = 11

    sample_dataset = pd.read_csv(
        data_filepath,
        sep=r"\s+",
        skiprows=header_row_index,
        names=["Current (A)", "Voltage (uV)", "Time (s)"],
    )
    print(sample_dataset.head())
    # sample_dataset.plot("Current (A)", "Voltage (uV)")
    current = sample_dataset["Current (A)"].to_numpy()
    voltage = sample_dataset["Voltage (uV)"].to_numpy()
    time = sample_dataset["Time (s)"].to_numpy

    plt.show()


if __name__ == "__main__":
    main()
