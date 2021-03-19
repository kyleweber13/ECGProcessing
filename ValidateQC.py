import ImportEDF
import ECG
import pandas as pd
import ECG_Quality_Check
import matplotlib.pyplot as plt
import random
import numpy as np
import Filtering


def import_goldstandard(filepath):

    gs = pd.read_excel(filepath)

    gs = gs[["ID", "Index", "Voltage Range", "ExpertDecision"]]

    return gs


df = import_goldstandard("/Users/kyleweber/Desktop/Data/OND07/Tabular Data/ECG_QualityControl_Testing_KW.xlsx")

percent_valid = []

for row in df.itertuples():
    subj_id = row.ID
    start_index = row.Index

    print("-Importing subject {}, index {}...".format(subj_id, start_index))

    data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.EDF".format(subj_id),
                             start_offset=start_index, end_offset=int(start_index + 15 * 250),
                             epoch_len=15, load_accel=False,
                             low_f=1, high_f=30, f_type="bandpass")

    r = ECG_Quality_Check.RedmondQC(ecg_signal=data.raw, sample_rate=data.sample_rate,
                                    start_index=0, stop_index=None, epoch_len=data.epoch_len)

    percent_val = round(r.final_mask.count("Valid") * 100 / len(r.final_mask), 2)
    percent_valid.append(percent_val)

    if row.Index == df.Index[-1]:

        df["Redmond%Valid"] = percent_valid

num_list = [i for i in range(df.shape[0])]


def plot_segment(segment=None):

    plt.close("all")

    if segment is None:
        rando = random.choice(num_list)
        num_list.remove(rando)

    if segment is not None:
        rando = segment

    data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.EDF".format(df["ID"].loc[rando]),
                             start_offset=df["Index"].loc[rando], end_offset=int(15 * 250),
                             epoch_len=15, load_accel=False,
                             low_f=1, high_f=30, f_type="bandpass")

    filt = Filtering.filter_signal(data=data.raw, sample_f=data.sample_rate, low_f=.6, high_f=30, filter_type='bandpass')

    r = ECG_Quality_Check.RedmondQC(ecg_signal=data.raw, sample_rate=data.sample_rate,
                                    start_index=0, stop_index=None, epoch_len=data.epoch_len)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 6))
    ax1.set_title("Row {}, {} % valid".format(rando, round(r.final_mask.count("Valid") * 100 / len(r.final_mask), 2)))

    ax1.plot(np.arange(0, len(data.raw))/data.sample_rate, data.raw, color='red', label="Raw")
    ax1.plot(np.arange(0, len(data.raw))/data.sample_rate, filt, color='black', label="Filt")
    ax1.legend()

    ax2.plot(np.arange(0, len(data.raw))/data.sample_rate, r.lowf_data, color='dodgerblue', label="Low F Data")
    ax2.legend()

    ax3.plot(np.arange(0, len(data.raw))/data.sample_rate, r.highf_data, color='green', label='High F Data')
    ax3.axhline(y=30, color='black')
    ax3.legend()

    ax4.plot(np.arange(0, len(r.final_mask))/data.sample_rate, r.final_mask, color='black')
    ax4.plot(np.arange(0, len(r.lowf_mask))/data.sample_rate, r.lowf_mask, color='dodgerblue')
    ax4.plot(np.arange(0, len(r.highf_mask))/data.sample_rate, r.highf_mask, color='green')

    return r


# r = plot_segment(None)
