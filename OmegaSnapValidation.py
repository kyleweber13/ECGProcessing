import pyedflib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Filtering
import nwecg.nwecg.ecg_quality as qc

file = "/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Data Files/007_OmegaSnap.EDF"
nw_file = "/Volumes/nimbal$/OBI/ONDRI@Home/Device Validation Protocols/Bittium Faros/Omega Snap Testing/OmegaSnap_Nonwear.xlsx"
df_nw = pd.read_excel(nw_file)
df_nw = df_nw.loc[df_nw["File"] == file]

"""Data import"""
f = pyedflib.EdfReader(file)

ecg = f.readSignal(0)
ecg_fs = f.getSampleFrequency(0)
ecg_filt = Filtering.filter_signal(data=ecg, sample_f=ecg_fs, low_f=.5, high_f=25, filter_type="bandpass")

acc = np.array([f.readSignal(1), f.readSignal(2), f.readSignal(3)])
acc_fs = f.getSampleFrequency(1)

temp = f.readSignal(5)
temp_fs = f.getSampleFrequency(5)

start_time = f.getStartdatetime()

f.close()


def plot_all_data():

    fig, axes = plt.subplots(3, sharex='col', figsize=(10, 7))
    plt.suptitle(file)

    axes[0].plot(pd.date_range(start=start_time, periods=int(len(ecg_filt)/3), freq=f"{1000/(ecg_fs/3)}ms"), ecg_filt[::3], color='red')
    axes[1].plot(pd.date_range(start=start_time, periods=int(len(acc[0])), freq=f"{1000/acc_fs}ms"), acc[0], color='black')
    axes[1].plot(pd.date_range(start=start_time, periods=int(len(acc[0])), freq=f"{1000/acc_fs}ms"), acc[1], color='red')
    axes[1].plot(pd.date_range(start=start_time, periods=int(len(acc[0])), freq=f"{1000/acc_fs}ms"), acc[2], color='dodgerblue')
    axes[2].plot(pd.date_range(start=start_time, periods=int(len(temp)), freq=f"{1000/temp_fs}ms"), temp, color='orange')

    for row in df_nw.itertuples():
        axes[0].fill_between(x=[row.Start, row.Stop], y1=min(ecg_filt), y2=max(ecg_filt), color='grey', alpha=.3)
        axes[1].fill_between(x=[row.Start, row.Stop], y1=min(acc[0]), y2=max(acc[0]), color='grey', alpha=.3)
        axes[2].fill_between(x=[row.Start, row.Stop], y1=min(temp), y2=max(temp), color='grey', alpha=.3)


filt, wiener_filt, snr, annots, thresh = qc.annotate_ecg_quality(sample_rate=ecg_fs, signal=ecg)
annots._annotations["quality"] = [row.quality.value for row in annots._annotations.itertuples()]
df_annots = annots._annotations.copy()