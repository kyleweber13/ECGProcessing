import ImportEDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import Filtering
import peakutils


def downsample_data(raw_voltage=None, old_fs=250, new_fs=125):

    print("\nDownsampling from {}Hz to {}Hz...".format(old_fs, new_fs))
    ds = raw_voltage[::int(old_fs/new_fs)]
    print("Complete.")

    return ds


def convert_to_mv(raw_data):
    print("\nConverting from uV to mV...")
    x = [i / 1000 for i in raw_data]
    print("Complete.")
    return x


def filter_data(raw_data):

    print("\nRunning .25-20Hz band-pass filter...")
    d = Filtering.filter_signal(data=raw_data, low_f=.25, high_f=20, filter_order=5,
                                filter_type='bandpass', sample_f=125)
    print("Complete.")
    return d


def check_zeros(data=None):
    pass


def check_variance(data=None, fs=125, nw_thresh=.00015, noise_thresh=.25, fill_gaps=None):
    """Looks for regions of low variance (non-wear)."""

    print("\nCalculating voltage variance in 1-second windows...")
    s2 = []
    for i in range(0, len(data), fs):
        w = data[i:i+fs]
        val = np.std(w)**2
        s2.append(val)
    print("Complete.")

    print("\nFlagging periods of low variance...")
    # 0 = variance above threshold --> not non-wear
    low_s2 = [1 if i <= nw_thresh else 0 for i in s2]
    print("\nFlagging periods of high variance...")
    high_s2 = [1 if i >= noise_thresh else 0 for i in s2]

    if fill_gaps is not None:
        for i in range(len(low_s2)-1):
            if low_s2[i] == 1 and low_s2[i+1] == 0 and low_s2[i+2] == 1:
                low_s2[i+1] = 1


    print("Complete.")

    return s2, low_s2, high_s2


def plot_variance():

    fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

    axes[0].plot(np.arange(0, len(data))/125/60, data, color='dodgerblue')
    axes[0].set_ylabel("Voltage")

    axes[1].plot(np.arange(len(s2))/60, s2, color='black')
    axes[1].set_ylabel("Voltage s^2")
    axes[1].set_xlabel("Minutes")

    ax2 = axes[1].twinx()
    ax2.plot(np.arange(len(s2))/60, low_var_flag, color='red')
    # ax2.plot(np.arange(len(s2))/60, high_var_flag, color='purple')


def remove_low_var_epochs(data, low_var_flag):

    print("\nZeroing regions of non-wear...")
    d = data.copy()
    for i, val in enumerate(low_var_flag):
        if val > 0:
            d[int(i*125):int((i+1)*125)] = np.zeros(125)
    print("Complete.")

    return d


ecg = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/BG7_FastFix.edf")
data = downsample_data(raw_voltage=ecg.raw, old_fs=ecg.sample_rate, new_fs=125)
# data = convert_to_mv(raw_data=data)
# data = filter_data(raw_data=data)
# s2, low_var_flag, high_var_flag = check_variance(data=data, fs=125, nw_thresh=.00015, noise_thresh=1, fill_gaps=5)
#plot_variance()
# d = remove_low_var_epochs(data, low_var_flag)

"""
# BG7_Lead
d = [11400000, int(1.22e7)]  # clean
# d = [int(1.46e7), int(1.48e7)]  # nw

f, Pxx = scipy.signal.welch(x=data[d[0]:d[1]], fs=125, nperseg=int(125*60), scaling="density")

psd = pd.DataFrame(list(zip(f.transpose(), Pxx.transpose())), columns=["Freq", "Power"])
del f, Pxx
psd = psd.loc[psd["Freq"] <= 5]

peaks = peakutils.indexes(y=psd["Power"], min_dist=62)  # peaks .5Hz apart

fig, axes = plt.subplots(2, figsize=(10, 6))
axes[0].plot(psd["Freq"], psd["Power"], color='black')

axes[1].plot(data[d[0]:d[1]], color='dodgerblue')
for i in peaks:
    axes[0].axvline(psd["Freq"].iloc[i], linestyle='dashed', color='red')
"""

"""OBSERVATIONS"""
# Noisy sections finds many more peaks --> white noise
plt.plot(ecg.timestamps[::2], data, color='black')