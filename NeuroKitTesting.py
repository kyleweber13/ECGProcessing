import neurokit2 as nk
import ImportEDF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from ecgdetectors.ecgdetectors import Detectors
import ecgdetectors
import scipy.stats

# https://neurokit2.readthedocs.io/en/latest/functions.html#module-neurokit2.signal
# https://pypi.org/project/neurokit2/


def test_ecg_process_func(signal, start, n_samples, fs, plot_builtin=False, plot_events=True):
    """Runs neurokit.ecg_process() on specified section of data and plots builtin plot or plot that
       shows P, R, and T waves, atrial/ventricular phases, and peaks.

    :param signal: array of data
    :param start: start index in signal
    :param n_samples: how many samples to process
    :param fs: sample rate, Hz
    :param plot_builtin: Boolean; runs neurokit.ecg_plot()
    :param plot_events: boolean: runs custom plot

    :return: data objects generated in neurokit.ecg_process()
    """
    s, i = nk.ecg_process(signal[start:start + n_samples], fs)

    if plot_events:

        fig, axes = plt.subplots(1, figsize=(10, 6))
        axes.plot(np.arange(0, s.shape[0]) / fs, s["ECG_Clean"], label="Filtered", color='black')

        axes.plot(s.loc[s["ECG_R_Peaks"] == 1].index / fs,
                  s.loc[s["ECG_R_Peaks"] == 1]["ECG_Clean"],
                  marker="o", linestyle="", color='grey', label="R Peak")

        axes.plot(s.loc[s["ECG_T_Peaks"] == 1].index / fs,
                  s.loc[s["ECG_T_Peaks"] == 1]["ECG_Clean"],
                  marker="x", linestyle="", color='red', label="T Peak")

        axes.plot(s.loc[s["ECG_P_Peaks"] == 1].index / fs,
                  s.loc[s["ECG_P_Peaks"] == 1]["ECG_Clean"],
                  marker="x", linestyle="", color='green', label="P Peak")

        axes.legend(loc='lower left')

        axes2 = axes.twinx()
        axes2.set_yticks([])
        axes2.set_ylim(0, 1)

        axes2.fill_between(x=np.arange(0, s.shape[0]) / fs, y1=.5,
                           y2=s["ECG_Phase_Atrial"].replace(to_replace=0, value=.5),
                           color='green', alpha=.25, label="Atrial phase")

        axes2.fill_between(x=np.arange(0, s.shape[0]) / fs, y1=.5,
                           y2=s["ECG_Phase_Ventricular"].replace(to_replace=0, value=.5),
                           color='purple', alpha=.25, label="Ventr. phase")

        p_start = s.loc[s["ECG_P_Onsets"] == 1]
        p_end = s.loc[s["ECG_P_Offsets"] == 1]

        t_start = s.loc[s["ECG_T_Onsets"] == 1]
        t_end = s.loc[s["ECG_T_Offsets"] == 1]

        r_start = s.loc[s["ECG_R_Onsets"] == 1]
        r_end = s.loc[s["ECG_R_Offsets"] == 1]

        for i in range(p_start.shape[0]):
            if i == range(p_start.shape[0])[-1]:
                axes2.fill_between(x=[p_start.index[i] / fs, p_end.index[i] / fs], y1=0, y2=.5,
                                   color='dodgerblue', alpha=.25, label="P wave")
            else:
                axes2.fill_between(x=[p_start.index[i] / fs, p_end.index[i] / fs], y1=0, y2=.5, color='dodgerblue',
                                   alpha=.25)

        for i in range(t_start.shape[0]):
            if i == range(t_start.shape[0])[-1]:
                axes2.fill_between(x=[t_start.index[i] / fs, t_end.index[i] / fs], y1=0, y2=.5,
                                   color='red', alpha=.25, label="T wave")
            else:
                axes2.fill_between(x=[t_start.index[i] / fs, t_end.index[i] / fs], y1=0, y2=.5, color='red', alpha=.25)

        for i in range(r_start.shape[0]):
            if i == range(r_start.shape[0])[-1]:
                axes2.fill_between(x=[r_start.index[i] / fs, r_end.index[i] / fs], y1=0, y2=.5,
                                   color='grey', alpha=.25, label="R wave")
            else:
                axes2.fill_between(x=[r_start.index[i] / fs, r_end.index[i] / fs], y1=0, y2=.5, color='grey', alpha=.25)

        axes2.legend(loc='lower right')
        axes.set_xlabel("Seconds")

    if plot_builtin:
        nk.ecg_plot(s, fs)

    return s, i


def run_qc(signal, epoch_len=10, fs=125, algorithm="averageQRS", show_plot=False):
    """Runs signal qualtiy check algorithm from neurokit2 on specified data. Returns signal quality data.

      :argument
      -signal: array of ecg data
      -epoch_len: seconds
      -fs: sample rate, Hz
      -algorithm: algorithm to use; either "zhao2018" or "averageQRS"
      -show_plot: boolean
    """

    if algorithm == "zhao2018":

        print(f"\nRunning Zhao 2018 quality check algorithm in {epoch_len}-second epochs...")

        # Zhou 2018 quality check
        qc = []
        for i in range(0, len(signal)-int(fs*epoch_len), int(fs*epoch_len)):
            qc.append(nk.ecg_quality(ecg_cleaned=signal[i:int(i+epoch_len*fs)], sampling_rate=fs,
                                     method=algorithm, approach='fuzzy'))

        qc = np.array(qc)
        qc[qc == "Excellent"] = 0
        qc[qc == "Barely acceptable"] = 1
        qc[qc == "Unacceptable"] = 2

        qc = [int(i) for i in qc]

        if show_plot:
            fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))
            axes[0].plot(np.arange(len(signal))/fs/60, signal, color='red', zorder=0)
            axes[0].set_title(f"Filtered Data ({fs}Hz)")
            axes[1].plot(np.arange(0, len(signal), int(epoch_len*fs))[:len(qc)]/fs/60, qc,
                         marker="o", color='black', markeredgecolor='black', markerfacecolor='white')
            axes[1].set_title("Zhao (2018) quality index")
            axes[1].set_yticks([0, 1, 2])

    if algorithm == "averageQRS":
        print(f"\nRunning average QRS quality check algorithm...")

        qc = nk.ecg_quality(ecg_cleaned=signal, sampling_rate=fs, method=algorithm, approach='simple')

        if show_plot:
            fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

            axes[0].plot(np.arange(len(signal))/fs/60, signal, color='red', zorder=0)
            axes[0].set_title(f"Filtered Data ({fs}Hz)")

            axes[1].plot(np.arange(0, len(signal))[:len(qc)]/fs/60, qc, color='black')
            axes[1].set_title("averageQRS quality index")
            axes[1].set_xlabel("Time (min)")

    print("Complete.")

    return qc


def threshold_averageqrs_data(signal=None, qc_signal=None, epoch_len=5, fs=125, pad_val=0,
                              thresh=.95, plot_data=True, method="mean"):
    """Defines valid epochs using QC data and a value threshold. Sets invalid epoch data values to specified value.

    :param signal: array of ecg data
    :param qc_signal: array of quality check output from run_qc's "averageQRS" algorithm
    :param epoch_len: seconds
    :param fs: sample rate, Hz
    :param pad_val: value that invalid epochs' data are set to (e.g. 0 or None)
    :param thresh: threshold of what % of window has to be above threshold; between 0 and 1
    :param plot_data: boolean
    :param method: "mean" or "exclusive"
        -"mean": mean QC value in epoch needs to be above thresh to pass
        -"exlusive": all QC values in epoch need to be above thresh to pass

    :returns
    -signal once is has been thresholded
    """

    print(f"\nThresholding QC data with a threshold of {thresh} in {epoch_len}-second epochs using {method} method...")

    data = []
    for i in range(0, len(signal), int(epoch_len*fs)):

        if method == 'mean':
            mean_sqi = np.mean(qc_signal[i:int(i+epoch_len*fs)])

            if mean_sqi >= thresh:
                for d in range(i, int(i+epoch_len*fs)):
                    try:
                        data.append(signal[d])
                    except IndexError:
                        pass
            if mean_sqi < thresh:
                for d in range(int(epoch_len*fs)):
                    data.append(pad_val)

        if method == "exclusive":

            above_thresh = [i >= thresh for i in qc_signal[i:int(i + epoch_len * fs)]]

            if False not in above_thresh:
                for d in range(i, int(i + epoch_len * fs)):
                    try:
                        data.append(signal[d])
                    except IndexError:
                        pass
            if False in above_thresh:
                for d in range(int(epoch_len * fs)):
                    data.append(pad_val)

    if plot_data:

        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))

        axes[0].plot(np.arange(0, len(signal))/fs, signal, color='black')
        axes[0].set_title("Input ECG")

        axes[1].plot(np.arange(len(data))/fs, data, color='green')
        axes[1].set_title("Thresholded")

        axes[2].plot(np.arange(len(qc_signal))/fs, qc_signal, color='dodgerblue')
        axes[2].axhline(y=thresh, color='red', linestyle='dashed')
        axes[2].set_title("SQI")
        axes[2].set_ylim(0, )

    return data


def find_peaks(signal, fs=125, show_plot=True, peak_method="pantompkins1985", clean_method="neurokit"):

    peaks = nk.ecg_findpeaks(ecg_cleaned=signal, sampling_rate=fs,
                             show=False, method=peak_method)

    peaks_clean = nk.signal_fixpeaks(peaks, sampling_rate=fs, iterative=True, show=False,
                                     interval_min=int(fs*.66),  # HR 240bpm
                                     interval_max=int(fs*2),  # HR 30bpm
                                     robust=False, method=clean_method)

    # peaks_clean = [i for i in peaks_clean if abs(i) >= 5]

    if show_plot:
        plt.plot(signal, color='black', zorder=0)
        plt.scatter(peaks["ECG_R_Peaks"], [signal[i] for i in peaks["ECG_R_Peaks"]], color='red', marker='x', zorder=1)
        plt.scatter(peaks_clean, [signal[i] for i in peaks_clean], color='limegreen', zorder=2)

    return peaks, peaks_clean


def segment_ecg(signal, fs):
    """Segments and plots specified region of data into individual heartbeats."""

    dict = nk.ecg_segment(ecg_cleaned=signal, sampling_rate=fs)

    for i in range(int([i for i in dict.keys()][-1])):
        plt.plot(dict[str(i + 1)]["Index"] / fs, dict[str(i + 1)]["Signal"], color="red" if i % 2 == 0 else 'black')

    plt.xlabel("Seconds")

    return dict


# Desired sample rate, Hz
"""
fs = 250

# Imports data
data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/FastFixRun1.edf",
                         start_offset=0, end_offset=0, epoch_len=15, load_accel=True,
                         low_f=1, high_f=25, f_type="bandpass")

# Downsampling data
if fs != data.sample_rate:
    ratio = int(data.sample_rate / fs)
    data.filtered = data.filtered[::ratio]
    data.timestamps = data.timestamps[::ratio]
"""

""" ============================================== ALL-IN-ONE-PROCESSING ========================================== """
# All-in-one processing
# Returns df; likely to be slow with any length of data
# Data: raw ecg, cleaned ecg, inst. HR, ECG quality, P/T/R onset/offsets/peaks, atrial/ventricular phase/completion,
# df_ecg = nk.ecg_process(ecg_signal=data.raw, sampling_rate=data.sample_rate, method='neurokit')[0]


def plot_df_ecg_peaks(ds_ratio=3):

    x = np.arange(0, df_ecg.shape[0])/data.sample_rate

    fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

    # ECG signal
    axes[0].plot(x[::ds_ratio], df_ecg["ECG_Raw"].iloc[::ds_ratio], color='red', label='raw', zorder=0)
    axes[0].plot(x[::ds_ratio], df_ecg["ECG_Clean"].iloc[::ds_ratio], color='black', label='clean', zorder=1)

    df_beat = df_ecg.loc[df_ecg["ECG_R_Peaks"] == 1]
    axes[0].scatter([i/data.sample_rate for i in df_beat.index], [df_ecg["ECG_Clean"].iloc[i] for i in df_beat.index],
                    color='limegreen', marker='x', label='R Peak', s=35)
    axes[0].legend(loc='lower left')

    axes[1].plot(x[::ds_ratio], df_ecg["ECG_Quality"].iloc[::ds_ratio], color='orange')
    axes[1].set_title("Signal Quality")


def plot_df_ecg_cycle(start=0, stop=10):

    df = df_ecg.iloc[int(start*60*data.sample_rate):int(stop*60*data.sample_rate)]

    x = np.linspace(start, stop, df.shape[0])

    fig, axes = plt.subplots(1, sharex='col', figsize=(10, 6))

    axes.plot(x, df["ECG_Clean"], color='black', zorder=0)

    ppeaks = [i - int(start*60*data.sample_rate) for i in df.loc[df["ECG_P_Peaks"] == 1].index]
    axes.scatter([x[i] for i in ppeaks], [df["ECG_Clean"].iloc[i] for i in ppeaks],
                 color="red", zorder=1, label="P-peak")
    rpeaks = [i - int(start*60*data.sample_rate) for i in df.loc[df["ECG_R_Peaks"] == 1].index]
    axes.scatter([x[i] for i in rpeaks], [df["ECG_Clean"].iloc[i] for i in rpeaks],
                 color="limegreen", zorder=1, label="R-peak")
    tpeaks = [i - int(start*60*data.sample_rate) for i in df.loc[df["ECG_T_Peaks"] == 1].index]
    axes.scatter([x[i] for i in tpeaks], [df["ECG_Clean"].iloc[i] for i in tpeaks],
                 color="dodgerblue", zorder=1, label="T-peak")
    axes.legend(loc='lower left')


#plot_df_ecg_peaks(ds_ratio=3)
# plot_df_ecg_cycle(start=5, stop=10)



# Signal cleaning. Selection of 6 algorithms/filtering methods
# ["neurokit", "biosppy", "pantompkins1985", "hamilton2002", "elgendi2010", "engzeemod2012"]
# ecg_clean = nk.ecg_clean(ecg_signal=data.raw, sampling_rate=data.sample_rate, method='neurokit')


def nothing():
    # Quality check algorithms
    # qcqrs = run_qc(signal=data.filtered, epoch_len=15, fs=fs, algorithm="averageQRS", show_plot=False)

    # Removing invalid data based on QC thresholdling
    #t = threshold_averageqrs_data(signal=data.filtered, qc_signal=qc, epoch_len=10, fs=fs, pad_val=0,
    #                              thresh=.95, method='exclusive', plot_data=False)

    p, pc = find_peaks(signal=data.filtered, fs=fs, show_plot=True, peak_method="pantompkins1985", clean_method='neurokit')
    hrv = nk.hrv_time(peaks=pc, sampling_rate=data.sample_rate, show=False)

    freq = nk.hrv_frequency(peaks=pc, sampling_rate=data.sample_rate, show=True)


    # df_events, info = test_ecg_process_func(signal=data.filtered[15000:15000+1250], start=0, n_samples=int(10*125), fs=fs, plot_builtin=True, plot_events=False)

    # heartbeats = segment_ecg(signal=d, fs=fs)


    waves, sigs = nk.ecg_delineate(ecg_cleaned=data.filtered, rpeaks=pc, sampling_rate=data.sample_rate, method='dwt', show=False)

    intervals, ecg_rate, ecg_hrv = nk.ecg_intervalrelated()


# TODO
# Organizing
# Signal quality in organized section --> able to pick which algorithm