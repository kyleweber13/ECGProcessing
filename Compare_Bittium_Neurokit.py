import neurokit2 as nk
import ImportEDF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ImportEDF
from NeuroKitTesting import *


def import_data(bittium_output, raw_edf):

    # Data generated from Cardioscope
    cols = pd.read_excel(bittium_output).columns
    bf = pd.read_excel(bittium_output, skiprows=2)

    cols = [i for i in cols]
    cols[0] = 'Timestamp'
    cols[25] = "HR"
    bf.columns = cols

    # Imports raw EDF data
    data = ImportEDF.Bittium(filepath=raw_edf, start_offset=0, end_offset=0, epoch_len=15, load_accel=True,
                             low_f=1, high_f=25, f_type="bandpass")

    # Crops Cardioscope data of epochs before Bittium file started...
    bf = bf.loc[bf["Timestamp"] >= data.timestamps[0]]
    bf = bf.reset_index()
    bf = bf.drop("index", axis=1)

    return bf, data


def generate_epoch_indexes():

    bf_crop = bf.loc[bf["Timestamp"] >= data.timestamps[0]]
    inds = [int((row.Timestamp - data.timestamps[0]).total_seconds() * data.sample_rate) for
            row in bf_crop.itertuples()]
    inds.append(len(data.timestamps))

    return inds


bf, data = import_data(bittium_output="/Volumes/nimbal$/OBI/ONDRI@Home/Device Information and Sample Data Files/"
                                      "User Manuals or Other Information/Bittium Faros/"
                                      "Processing Software/Cardiscope Test/MG_Run1/MG_3LeadRun1.xlsx",
                       raw_edf="/Users/kyleweber/Desktop/3LeadRun1.edf")

epoch_inds = generate_epoch_indexes()
epoch_len = int((epoch_inds[1] - epoch_inds[0])/data.sample_rate)  # seconds

# NeuroKit Processing: peak detection and HR analysis -------------------------
# p = peaks; pc = cleaned peaks
p, pc = find_peaks(signal=data.filtered, fs=data.sample_rate, show_plot=False,
                   peak_method="pantompkins1985", clean_method='neurokit')
p = p["ECG_R_Peaks"]


def process_epochs_neurokit():
    # Within-epoch processing using NeuroKit to match Cardioscope output
    df_nk = pd.DataFrame([[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]).transpose()
    df_nk.columns = ["Timestamp", "Index", "Quality", "HR", "meanRR", "sdRR", "meanNN", "SDNN", "pNN20", 'pNN50',
                     "VLF", "LF", "HF", "LF/HF", "LFn", "HFn"]

    print("\nProcessing data in epochs with NeuroKit2...")

    for start, stop in zip(epoch_inds[:], epoch_inds[1:]):
        print(f"{round(100*start/epoch_inds[-1], 1)}%")

        s, i = nk.ecg_process(data.filtered[start:stop], sampling_rate=data.sample_rate)
        s = s.loc[s["ECG_R_Peaks"] == 1]
        inds = [i for i in s.index]

        hrv = nk.hrv_time(inds, sampling_rate=data.sample_rate, show=False)
        freq = nk.hrv_frequency(inds, sampling_rate=data.sample_rate, show=False)

        rr_ints = [(d2 - d1)/data.sample_rate for d1, d2 in zip(inds[:], inds[1:])]
        mean_rr = 1000*np.mean(rr_ints)
        sd_rr = 1000*np.std(rr_ints)

        out = [data.timestamps[start], start,
               100*s["ECG_Quality"].mean(), s["ECG_Rate"].mean().__round__(3),
               mean_rr, sd_rr,
               hrv["HRV_MeanNN"].iloc[0], hrv["HRV_SDNN"].iloc[0],
               hrv["HRV_pNN20"].iloc[0], hrv["HRV_pNN50"].iloc[0],
               freq["HRV_VLF"].iloc[0], freq["HRV_LF"].iloc[0], freq["HRV_HF"].iloc[0], freq["HRV_LFHF"].iloc[0],
               freq["HRV_LFn"].iloc[0], freq["HRV_HFn"].iloc[0]]

        df_out = pd.DataFrame(out).transpose()
        df_out.columns = df_nk.columns
        df_nk = df_nk.append(df_out, ignore_index=True)

    print("100%")

    df_nk["Timestamp"] = pd.date_range(start=bf["Timestamp"].iloc[0], freq=f"{epoch_len}S", periods=df_nk.shape[0])

    return df_nk


df_nk = process_epochs_neurokit()

# TODO
# nk.hrv_frequency(peaks, sampling_rate=100, show=True)
# Replacing cardioscope missing values with None instead of 0?
# check to see if freq domain categories are the same


def compare_variable(variable=""):

    # Column in df_nk and its equivalent in bf
    var_dict = {"Timestamp": "Timestamp", "Quality": "Quality", "HR": "HR", "meanRR": "Ø(RR)",
                "meanNN": "Ø(NN)", "SDNN": "SDNN", "pNN20": "pNN20", "pNN50": "pNN50",
                "VLF": "VLF", "LF": "LF", "HF": "HF", "LF/HF": "LF/HF", "LFn": "LF normalized", "HFn": "HF normalized"}
    units_dict = {"Quality": "%", "HR": "bpm", "meanRR": "ms",
                  "meanNN": "ms", "SDNN": "ms", "pNN20": "N", "pNN50": "N",
                  "VLF": "Power", "LF": "Power", "HF": "Power", "LF/HF": "Power", "LFn": "Power", "HFn": "Power"}

    fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))
    plt.subplots_adjust(hspace=.25)
    axes[0].plot(df_nk["Timestamp"], df_nk[variable], color='dodgerblue', label='NeuroKit')
    axes[0].plot(df_nk["Timestamp"], bf[var_dict[variable]], color='red', label="Cardioscope")
    axes[0].legend()
    axes[0].set_ylabel(units_dict[variable])
    axes[0].set_title(variable)

    axes[1].plot(df_nk["Timestamp"], [a - b for a, b in zip(df_nk[variable], bf[var_dict[variable]])], color='orange')
    axes[1].set_ylabel(f"{units_dict[variable]} (NK - CS)")
    axes[1].set_title("Difference")
    axes[1].axhline(color='black', y=0, linestyle='dotted')


compare_variable("Quality")
