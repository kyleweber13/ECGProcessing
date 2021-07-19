import ImportEDF
import nwecg.nwecg.awwf as wiener_filter
import nwecg.nwecg.ecg_quality as qc
import matplotlib.pyplot as plt
import Filtering
import numpy as np
import pandas as pd


def create_filt_snr_df():

    cats = []
    for a in snr_filt:
        if a < thresh[0]:
            cats.append("Q3")
        if thresh[0] <= a < thresh[1]:
            cats.append("Q2")
        if a >= thresh[1]:
            cats.append("Q1")

    inds = []
    for i in range(len(cats) - 1):
        if cats[i] != cats[i + 1]:
            inds.append(i + 1)

    if len(inds) % 2 == 1:
        inds.append(len(snr_filt))

    df_filt_annots = pd.DataFrame([inds[::2], inds[1::2], [cats[i] for i in inds[::2]]]).transpose()
    df_filt_annots.columns = ["start_idx", "end_idx", "quality"]

    rows = []
    for i in range(df_filt_annots.shape[0] - 1):
        end = df_filt_annots.iloc[i]["end_idx"]
        start = df_filt_annots.iloc[i + 1]["start_idx"]
        cat = cats[int(end)]

        rows.append([end, start, cat])

    rows.append([rows[-1][1], len(snr_filt), cats[-1]])

    df_append = pd.DataFrame(rows, columns=df_filt_annots.columns)

    df_all = pd.concat([df_filt_annots, df_append], ignore_index=True)
    df_all = df_all.sort_values("start_idx")
    df_all = df_all.reset_index()
    df_all = df_all[["start_idx", "end_idx", "quality"]]

    df_all['duration'] = [(row.end_idx - row.start_idx) / data.sample_rate for row in df_all.itertuples()]
    return df_all


def plot_data(ds_ratio=3):

    fig, axes = plt.subplots(2, sharex='col', figsize=(11, 7))

    axes[0].plot(data.timestamps[::ds_ratio], wiener_filt[::ds_ratio], color='black')
    axes[0].set_title("Wiener filtered")

    axes[1].plot(data.timestamps[::ds_ratio], snr[::ds_ratio], color='black', label="Raw SNR")
    axes[1].plot(data.timestamps[::ds_ratio], snr_filt[::ds_ratio], color='red', label="LP SNR")
    axes[1].legend()
    axes[1].set_ylabel("SNR (dB)")

    c = {"UNKNOWN": "grey", "Q1": 'green', "Q2": "dodgerblue", "Q3": 'red'}

    min_snr = min(snr)
    max_snr = max(snr)
    r = (max_snr - min_snr) / 2

    for row in annots._annotations.itertuples():
        axes[1].fill_between(x=[data.timestamps[row.start_idx],
                                data.timestamps[row.end_idx-1] if len(data.timestamps) - 1 >= row.end_idx else
                                data.timestamps[-1]],
                             y1=(r + min_snr) * 1.01, y2=max_snr, color=c[row.quality], alpha=.35)

    for row in df_all.itertuples():
        try:
            axes[1].fill_between(x=[data.timestamps[row.start_idx],
                                    data.timestamps[row.end_idx-1] if len(data.timestamps) - 1 >= row.end_idx else
                                    data.timestamps[-1]],
                                 y1=min_snr, y2=(max_snr - r) * .99, color=c[row.quality], alpha=.35)
        except IndexError:
            pass

    for t in thresh:
        axes[1].axhline(t, color='orange')


def replace_unusable_data(ecg_signal, annotations_df, value=None):

    df_bad = annotations_df.loc[annotations_df["quality"] == "Q3"]

    ecg = np.array(ecg_signal)

    for row in df_bad.itertuples():
        ecg[int(row.start_idx):int(row.end_idx)] = value

    return ecg


data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/007_OmegaSnap.EDF",
                         start_offset=0, end_offset=0, epoch_len=15, load_accel=False,
                         low_f=1, high_f=25, f_type="bandpass")
f = Filtering.filter_signal(data=data.raw, filter_type='bandpass', low_f=.5, high_f=30,
                            filter_order=3, sample_f=data.sample_rate)

# No bandpass/notch filtering
w = wiener_filter.awwf(f, data.sample_rate)

filt, wiener_filt, snr, annots, thresh = qc.annotate_ecg_quality(sample_rate=data.sample_rate, signal=data.raw)
annots._annotations["quality"] = [row.quality.value for row in annots._annotations.itertuples()]
df_annots = annots._annotations.copy()

# .05Hz lowpass filter on SNR data
snr_filt = Filtering.filter_signal(data=snr, filter_type='lowpass', low_f=.05, sample_f=data.sample_rate)
snr_filt = np.array(snr_filt)

df_all = create_filt_snr_df()
# ecg_edit = replace_unusable_data(ecg_signal=wiener_filt, annotations_df=df_all)

# plot_data(ds_ratio=10)
