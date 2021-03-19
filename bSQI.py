import matplotlib.pyplot as plt
import ImportEDF
import numpy as np
import scipy.stats as stats
import scipy
import Filtering
from ecgdetectors import Detectors
import pandas as pd
import ECG_Quality_Check

"""================================================== LIU ET AL. 2018 =============================================="""

"""
data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/Converted/"
                                  "Collection 1/3LeadHIIT1.EDF",
                         start_offset=0, end_offset=0, epoch_len=10, load_accel=False,
                         low_f=.67, high_f=30, f_type="bandpass")
"""
data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/Data/ECG Files/OND07_WTL_3023_01_BF.EDF",
                         start_offset=0, end_offset=0, epoch_len=10, load_accel=False,
                         low_f=.67, high_f=30, f_type="bandpass")

detectors = Detectors(data.sample_rate)

# Number of datapoints corresponding to 125ms window
err_size = int(125 / (1000/data.sample_rate))


def run_algorithm(threshold=.9, win_len=10, show_plot=True):

    # Data lists
    bsqi_list = []
    hr_list = []
    wav_hr = []
    ham_hr = []
    all_peaks = []
    all_ham_peaks = []
    all_wav_peaks = []
    orphanidou = []

    # Peak spacing in seconds corresponding to 40bpm
    agree_peaks_lim = [.66 * win_len, 3 * win_len]

    # Loops through data in designated windows
    for i in range(0, len(data.raw) - int(win_len*data.sample_rate), int(win_len*data.sample_rate)):

        # Two peak detection methods: Hamilton and SWT w/ Pan-Tompkins
        ham_peaks, ham_filt = detectors.hamilton_detector(unfiltered_ecg=data.raw[i:int(i+win_len*data.sample_rate)])
        wave_peaks, swt_filt, filt_sq = detectors.swt_detector(unfiltered_ecg=data.raw[i:int(i+win_len*data.sample_rate)])

        # Number of beats in each peak detection method
        n_hamilton = len(ham_peaks)
        n_wave = len(wave_peaks)

        # Finds peaks that detect synchronous peaks (<150ms difference)
        agree_peaks = 0  # number of agreed upon peaks
        peaks_list = []  # list of agreed appoint peaks (indexes)

        for ham_peak in ham_peaks:
            for wave_peak in wave_peaks:
                if abs(ham_peak - wave_peak) <= err_size:
                    agree_peaks += 1
                    peaks_list.append(wave_peak)
                    break

        # Calculates average HR in window
        try:
            hr = (len(peaks_list)-1) / ((peaks_list[-1] - peaks_list[0])/data.sample_rate) * 60
        except IndexError:
            hr = 0
        hr_list.append(hr)

        hr = (len(ham_peaks)-1) / ((ham_peaks[-1] - ham_peaks[0])/data.sample_rate) * 60
        ham_hr.append(hr)

        hr = (len(wave_peaks)-1) / ((wave_peaks[-1] - wave_peaks[0])/data.sample_rate) * 60
        wav_hr.append(hr)

        # Calculates BSQI if HR corresponds to 40-180bpm
        if agree_peaks_lim[0] <= len(peaks_list) <= agree_peaks_lim[1]:
            bsqi = agree_peaks / (n_hamilton + n_wave - agree_peaks)
            bsqi_list.append(bsqi)
        if not agree_peaks_lim[0] <= len(peaks_list) <= agree_peaks_lim[1]:
            bsqi_list.append(-.01)

        # Adds all peaks to all_peaks list
        for j in peaks_list:
            all_peaks.append(j+i)
        for j in ham_peaks:
            all_ham_peaks.append(j+i)
        for j in wave_peaks:
            all_wav_peaks.append(j+i)

        # Orphanidou algorithm ----------
        d = ECG_Quality_Check.CheckQuality(raw_data=data.raw, sample_rate=data.sample_rate, start_index=i,
                                           template_data='raw', voltage_thresh=250, epoch_len=win_len)
        orphanidou.append(1 if d.valid_period else 0)

    # Replaces invalid windows' HR data with None
    hr_list = np.array(hr_list)

    for i in range(len(hr_list)):
        if bsqi_list[i] < threshold:
            hr_list[i] = None

    # Graph
    if show_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
        fig.subplots_adjust(hspace=.33, top=.94, bottom=.05)

        # Data w/ peaks -----------------------------------------------------------------------------------------------
        ax1.set_title("Bandpass Filtered Data w/ Peaks")
        max_val = max(data.filtered) * 1.1

        # Filtered data
        ax1.plot(np.arange(0, len(data.filtered))/data.sample_rate, data.filtered, color='black')

        # Agreed peaks
        ax1.plot([i/data.sample_rate for i in all_peaks], [max_val for i in all_peaks],
                 linestyle="", marker="v", markersize=6, color='limegreen', label="Agree")

        # Hamilton algorithm peaks
        ax1.plot([i/data.sample_rate for i in all_ham_peaks], [data.filtered[i] for i in all_ham_peaks],
                 linestyle="", marker="o", markersize=6, color='dodgerblue', label="Hamilton")

        # Wavelet transform peaks
        ax1.plot([i/data.sample_rate for i in all_wav_peaks], [data.filtered[i] for i in all_wav_peaks],
                 linestyle="", marker="x", markersize=6, color='orange', label="Wavelet")

        ax1.set_ylim(ax1.get_ylim()[0], max_val * 1.2)
        ax1.legend(loc='upper right')

        # HR ----------------------------------------------------------------------------------------------------------
        ax2.set_title("Average HR")
        ax2.plot(np.arange(0, len(data.raw), int(win_len*data.sample_rate))
                 [:len(bsqi_list)]/data.sample_rate+(win_len/2),
                 hr_list, color='red', marker="o", markeredgecolor='black', markerfacecolor='white', markersize=6)
        ax2.set_ylabel("BPM")

        # bSQI --------------------------------------------------------------------------------------------------------
        ax3.set_title("QC Values (bSQI threshold = {})".format(threshold))
        ax3.bar(np.arange(0, len(data.raw), int(win_len * data.sample_rate))[:len(bsqi_list)] / data.sample_rate,
                orphanidou,
                width=win_len, color=['dodgerblue' if i == 1 else "white" for i in orphanidou], zorder=0,
                edgecolor=['black' if i == 1 else "white" for i in orphanidou], align="edge", label='Orphanidou QC')
        ax3.scatter(np.arange(0, len(data.raw), int(win_len*data.sample_rate))[:len(bsqi_list)]/data.sample_rate + win_len/2,
                    bsqi_list, label='bSQI', zorder=1,
                    color=['limegreen' if i >= threshold else "red" for i in bsqi_list], edgecolor='black')
        ax3.axhline(y=threshold, linestyle='dashed', color='orange')
        ax3.axhline(y=0, color='black')
        ax3.axhline(y=1, color='black')
        ax3.set_ylim(-.1, 1.1)
        ax3.set_xlabel("Seconds")
        ax3.legend(loc='lower right')

    agree_tally = 0
    for o, b in zip(orphanidou, bsqi_list):
        if o == 1 and b >= threshold:
            agree_tally += 1
        if o == 0 and b < threshold:
            agree_tally += 1

    perc_agree = round(agree_tally/len(bsqi_list) * 100, 2)

    print("\nOrphanidou and bSQI algorithm agree on {}% of epochs.".format(perc_agree))

    return bsqi_list, all_peaks, all_ham_peaks, all_wav_peaks, hr_list, wav_hr, ham_hr, orphanidou


bsqi, all_peaks, ham_peaks, wav_peaks, hr, wave_hr, ham_hr, orphanidou = run_algorithm(threshold=.6, win_len=15,
                                                                                       show_plot=False)

