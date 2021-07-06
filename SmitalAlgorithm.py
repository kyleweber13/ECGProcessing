import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import scipy.fft
import Filtering
import PIL

fs = 1000


def calc_descriptive(annotations, signal, folder, subfolder):
    """Function that calculates descriptive statistics and runs cumulative FFT on raw and Wiener-filtered data
       for each section of specified data.

       :argument
       -annotations: dataframe of signal quality annotations.
       -signal: array of ECG signal
       -folder: pathway to subject's folder
       -subfolder: name of subject's specific collection folder that contains relevant files
    """

    # Empty data to be populated
    range_list_f = []  # voltage range of Wiener-filtered data
    sd_list_f = []  # voltage SD of Wiener-filtered data

    range_list = []  # voltage range of raw data (.25Hz highpass filtered)
    sd_list = []  # voltage SD of raw data (.25Hz highpass filtered)

    # Cumulative FFT dataframe for raw data
    df_fft_raw = pd.DataFrame([[], [], [], [], []]).transpose()
    df_fft_raw.columns = ["Percent", "Freq", "ID", "category", "event"]

    # Cumulative FFT dataframe for Wiener-filtered data
    df_fft_w = pd.DataFrame([[], [], [], [], []]).transpose()
    df_fft_w.columns = ["Percent", "Freq", "ID", "category", "event"]

    # Loops through each flagged data segment
    for row in annotations.itertuples():
        d = signal.iloc[int(row.start_idx):int(row.end_idx)]  # Subsection of data

        # Only includes segments of data at least 5-seconds long
        if d.shape[0] >= (fs*5):

            # Wiener filtered
            desc = d["signal_AWWF"].describe()
            range_list_f.append(desc["max"] - desc['min'])
            sd_list_f.append(desc["std"])

            del desc

            # Highpass filtered to remove baseline wander
            desc = pd.Series(Filtering.filter_signal(data=d["signal_raw"], sample_f=fs, high_f=.25,
                                                     filter_order=3, filter_type='highpass')).describe()
            range_list.append(desc["max"] - desc['min'])
            sd_list.append(desc["std"])

            """Cumulative FFT data: calculates frequencies that account for np.arange(10, 101, 10)% of signal power"""
            # FFT data: raw (.25Hz highpass)
            raw, cs_raw = cumulative_fft(signal=Filtering.filter_signal(data=d["signal_raw"],
                                                                        sample_f=fs, high_f=.25,
                                                                        filter_order=3,
                                                                        filter_type='highpass'))

            cs_raw["ID"] = [f"{folder}_{subfolder}" for i in range(cs_raw.shape[0])]
            cs_raw["category"] = [row.quality for i in range(cs_raw.shape[0])]
            cs_raw["event"] = [row.Index + 1 for i in range(cs_raw.shape[0])]

            df_fft_raw = df_fft_raw.append(cs_raw)

            # FFT data: Wiener filtered data
            w, cs_w = cumulative_fft(signal=d["signal_AWWF"])
            cs_w["ID"] = [f"{folder}_{subfolder}" for i in range(cs_w.shape[0])]
            cs_w["category"] = [row.quality for i in range(cs_w.shape[0])]
            cs_w["event"] = [row.Index + 1 for i in range(cs_w.shape[0])]

            df_fft_w = df_fft_w.append(cs_w)

        if d.shape[0] < (fs*5):
            range_list_f.append(None)
            sd_list_f.append(None)
            range_list.append(None)
            sd_list.append(None)

    # Adds columns with descriptive stats to annotations df
    annotations["range_AWWF"] = range_list_f
    annotations['sd_AWWF'] = sd_list_f
    annotations["range_raw"] = range_list
    annotations['sd_raw'] = sd_list

    return df_fft_raw, df_fft_w


def cumulative_fft(signal):
    """Runs FFT on given signal and calculates the frequencies that account for np.arange(10, 101, 10)% of the
       signal power.

        :argument
        -signal: array of ECG signal

        :returns
        -df_fft: df for all calculated frequencies
        -df_cs: df of only 10%-interval data
    """

    length = len(signal)
    fft = scipy.fft.fft([i for i in signal])

    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / fs)), length // 2)

    df_fft = pd.DataFrame(list(zip(xf, 2.0 / length / 2 * np.abs(fft[0:length // 2]))), columns=["Freq", "Power"])

    # Removes rows of data below .25Hz (DC component)
    df_fft = df_fft.loc[df_fft["Freq"] >= .25]

    # Calculates cumulative % below value for each row
    df_fft["CSum"] = 100 * df_fft["Power"].cumsum() / sum(df_fft["Power"])

    # Finds frequencies at eat 10%-interval
    percent = 10
    freqs = []
    for row in df_fft.itertuples():
        if row.CSum >= percent:
            freqs.append(row.Freq)
            percent += 10

            if percent > 100:
                break

    df_cs = pd.DataFrame(list(zip(np.arange(10, 101, 10), freqs)), columns=["Percent", "Freq"])

    return df_fft, df_cs


def write_files_loop(root_folder="/Users/kyleweber/Desktop/Smital_ECG_Validation/",
                     fft_write_dir="/Users/kyleweber/Desktop/ECG_FFT/",
                     descriptive_write_dir="/Users/kyleweber/Desktop/ECG_Validation_Descriptive/"):
    """Loops through all files in all folders in root_folder. Runs cumulative FFT (cumulative_fft()) for each dataset
       using annotations from the manually-detected signal quality (annotations_manual.csv). Adds subject ID to
       manual annotations df and overwrites original file (used if combining all data into one df)

    :argument
    -root_folder: pathway to folder that contains all data
    -fft_write_dir: pathway to where FFT data is written
    -descriptive_write_dir: pathway to where descriptive data (range, SD) is written
    """

    ids = os.listdir(root_folder)

    for folder in ids:

        if folder != ".DS_Store":

            sub_f = os.listdir(root_folder + folder)

            for subfolder in sub_f:
                if subfolder != ".DS_Store":  # stupid Macs

                    print(folder, subfolder)

                    manual = pd.read_csv(f"{root_folder}/{folder}/{subfolder}/annotations_manual.csv")
                    s = pd.read_csv(f"{root_folder}/{folder}/{subfolder}/signal.csv")

                    df_fft_raw, df_fft_w = calc_descriptive(annotations=manual, signal=s,
                                                            folder=folder, subfolder=subfolder)
                    df_fft_raw.to_csv(f"{fft_write_dir}{folder}_{subfolder}_Raw.csv", index=False)
                    df_fft_w.to_csv(f"{fft_write_dir}{folder}_{subfolder}_AWWF.csv", index=False)

                    manual["ID"] = [f"{folder}_{subfolder}" for i in range(manual.shape[0])]

                    manual.to_csv(f'{descriptive_write_dir}{folder}_{subfolder}.csv',
                                  index=False)


def combine_files(folder="/Users/kyleweber/Desktop/ECG_Validation_Descriptive/", remove_unknown=True):

    df = pd.DataFrame([[], [], [], [],[], [], [], []]).transpose()
    df.columns = ['start_idx', 'end_idx', 'quality', 'range_AWWF', 'sd_AWWF', "range_raw", "sd_raw", 'ID']

    for file in os.listdir(folder):
        print(f"Reading {file}...")
        d = pd.read_csv(f"{folder}/{file}")

        df = df.append(d)

    if remove_unknown:
        df = df.loc[df["quality"] != "UNKNOWN"]

    return df


def import_fft_data(folder):

    files = [i for i in os.listdir(folder) if "csv" in i]

    df_all = pd.DataFrame([[], [], [], [], [], []]).transpose()
    df_all.columns = ["Percent", "Freq", "ID", "category", "event", "data"]

    for file in files:
        print(file)

        df = pd.read_csv(folder + file)

        data_type = "raw" if "Raw" in file else "AWWF"
        df["data"] = [data_type for i in range(df.shape[0])]

        df_all = df_all.append(df)

    return df_all


def gen_fft_heatmap(df, save_fig=False, save_dir="/Users/kyleweber/Desktop"):
    """Generates a hexbin/heatmap of the cumulative FFT data sorted by raw/wiener-filtered and signal quality category.

    :argument
    -df: dataframe from import_fft_data()
    -save_fig: boolean; will save as 200dpi tiff file
    -save_dir: pathway to where tiff will be saved
    """

    fig, axes = plt.subplots(2, 3, sharey='row', sharex='col', figsize=(11, 7))
    axes[0][0].set_ylabel("Percent")
    axes[1][0].set_ylabel("Percent")
    axes[1][0].set_xlabel("Freq")
    axes[1][1].set_xlabel("Freq")
    axes[1][2].set_xlabel("Freq")

    axes[0][0].set_title(".25Hz HP Q1")
    axes[0][1].set_title(".25Hz HP Q2")
    axes[0][2].set_title(".25Hz HP Q3")

    axes[1][0].set_title("Wiener Q1")
    axes[1][1].set_title("Wiener Q2")
    axes[1][2].set_title("Wiener Q3")

    d = df.loc[(df["data"]=="raw") & (df["category"]=="Q1")]
    axes[0][0].hexbin(d["Freq"], d["Percent"], cmap='nipy_spectral')

    d = df.loc[(df["data"]=="raw") & (df["category"]=="Q2")]
    axes[0][1].hexbin(d["Freq"], d["Percent"], cmap="nipy_spectral")

    d = df.loc[(df["data"]=="raw") & (df["category"]=="Q3")]
    axes[0][2].hexbin(d["Freq"], d["Percent"], cmap="nipy_spectral")

    d = df.loc[(df["data"]=="AWWF") & (df["category"]=="Q1")]
    axes[1][0].hexbin(d["Freq"], d["Percent"], cmap="nipy_spectral")

    d = df.loc[(df["data"]=="AWWF") & (df["category"]=="Q2")]
    axes[1][1].hexbin(d["Freq"], d["Percent"], cmap="nipy_spectral")

    d = df.loc[(df["data"]=="AWWF") & (df["category"]=="Q3")]
    axes[1][2].hexbin(d["Freq"], d["Percent"], cmap="nipy_spectral")

    axes[0][0].set_xlim(0, 100)
    axes[0][1].set_xlim(0, 100)
    axes[0][2].set_xlim(0, 100)
    axes[1][0].set_xlim(0, 100)
    axes[1][1].set_xlim(0, 100)
    axes[1][2].set_xlim(0, 100)

    if save_fig:
        plt.savefig(f"{save_dir}CumulativeFFT.tiff", dpi=200)


# write_files_loop()
# desc = combine_files(remove_unknown=True)
# df_all_fft = import_fft_data(df=folder="/Users/kyleweber/Desktop/ECG_FFT/")
# gen_fft_heatmap(df=df_all_fft, save_fig=False)
