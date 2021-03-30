import ImportEDF
import math
import pandas as pd
import pywt
import Filtering
import matplotlib.pyplot as plt
import scipy.signal
import statistics as stats
import numpy as np


class SmitalProcessing:

    def __init__(self, file):

        self.file = file

        """RUNS METHODS"""
        self.signal, self.fs = self.import_ecg()
        self.u, self.y = self.initial_wavelets()
        self.df_g = self.wiener_filter()
        self.sig_est, self.y_est, self.noise_est = self.estimate_signals()
        self.stnr = self.calculate_stnr()

    def import_ecg(self):
        """FINALIZED AND CONFIDENT"""

        # Imports data
        data = ImportEDF.Bittium(filepath=self.file)
        fs = data.sample_rate

        # Runs .67Hz HP and 60Hz notch filters
        signal = Filtering.filter_signal(data=data.raw, high_f=.67, filter_type='highpass',
                                         filter_order=5, sample_f=fs)
        signal = Filtering.filter_signal(data=signal, filter_type='notch', notch_f=60,
                                         filter_order=5, sample_f=fs)

        return signal, fs

    def initial_wavelets(self):

        print("\nProcessing intials wavelets...")

        # data padding to allow level 4 decomposition (requires data len which is multiple of 16 [2**n])
        req_len_mult = 2 ** 4
        data_len = len(self.signal)
        pad_n = np.ceil(data_len / req_len_mult) * req_len_mult - data_len

        for i in range(int(pad_n)):
            self.signal = np.append(self.signal, 0)

        # wavelet transformations
        """INITIAL WAVELET IS GOOD"""
        u = pywt.swt(data=self.signal, wavelet='db4', axis=0, level=4, norm=True)  # noise-free signal coefficients
        # u = pywt.threshold(data=u, mode='garrote', value=(np.median(u) / .6745) ** 2)
        # u = [u[i][0] for i in range(len(u))]  # Gets wavelet coefficients (ignores approx. coeffs)

        u_final = []
        for band in range(len(u)):
            t1 = pywt.threshold(data=u[band][0], mode='garrote', value=(np.median(u[band][0]) / .6745) ** 2)
            t2 = pywt.threshold(data=u[band][1], mode='garrote', value=(np.median(u[band][1]) / .6745) ** 2)

            u_final.append([t1, t2])

        u_final = np.array(u_final)
        u = [u_final[i][1] for i in range(len(u_final))]  # wavelet/details coeffs = [band][1]; approx = [band][0]

        y = pywt.swt(data=self.signal, wavelet='sym4', axis=0, level=4, norm=True)  # used for some threshold?
        y = [y[i][1] for i in range(len(y))]  # wavelet/details coeffs = [band][1]; approx = [band][0]

        return u, y

    def wiener_filter(self):

        print("\nApplying Wiener filter...")

        # Variance of noise coefficients: Equation 3
        sigma_sq = []

        for band in range(len(self.u)):
            dat = np.abs(self.u[band])
            val = (np.median(dat) / .6745) ** 2  # already took absolute values
            sigma_sq.append(val)

        # Wiener correction factor for each band (full collection): Equation 2
        df_g = []

        for band in range(len(self.u)):
            g = [self.u[band][i] ** 2 / (self.u[band][i] ** 2 + sigma_sq[band]) for i in range(len(self.signal))]

            df_g.append(g)
        
        df_g = np.array(df_g)

        return df_g

    def estimate_signals(self):

        print("\nEstimating signals from wavelet data...")

        # Calculates estimated denoised wavelet coefficients. Equation 4
        y_est = self.y * self.df_g

        # Denoised signal estimate from estimated denoised wavelet coefficients.
        sig_est = pywt.iswt(coeffs=y_est, wavelet='sym4', norm=True)

        # Estimating noise by subtracting estimated signal from true signal
        # noise_est = sig_est - self.signal
        noise_est = self.signal - sig_est

        return sig_est, y_est, noise_est

    def calculate_stnr(self):

        print("\nCalculating signal-to-noise ratio...")

        # Equation 5
        stnr = []

        # Calculates SNR every 2 seconds
        for window in range(0, len(self.signal), int(self.fs * 2)):
            sig = [i ** 2 for i in self.sig_est[window:int(window + self.fs * 2)]]
            noise = [i ** 2 for i in self.noise_est[window:int(window + self.fs * 2)]]

            val = 10 * math.log10(sum(sig) / sum(noise))
            stnr.append(val)

        # lowpass filters STNR data
        snrt_final = Filtering.filter_signal(data=stnr, sample_f=.5, filter_type='lowpass', low_f=.05, filter_order=3)
        print("Complete.")

        return snrt_final

    def plot_data(self):

        fig, axes = plt.subplots(3, sharex='col', sharey='col', figsize=(10, 7))
        plt.subplots_adjust(hspace=.33)
        plt.suptitle(self.file.split("/")[-1])

        axes[1].plot(np.arange(0, len(self.sig_est)) / self.fs, self.sig_est, color='green')
        axes[1].set_title("Estimated noise-free signal (db4 wavelet)")
        axes[1].set_ylabel("Voltage")

        axes[2].plot(np.arange(0, len(self.noise_est)) / self.fs, self.noise_est, color='red')
        axes[2].set_title("Estimated noise signal")
        axes[2].set_ylabel("Voltage")

        ax3 = axes[0].twinx()
        ax3.set_ylabel("SNR (dB)")
        ax3.yaxis.label.set_color("dodgerblue")

        xlim = ax3.get_xlim()
        ylim = ax3.get_ylim()

        ax3.fill_between(x=xlim, y1=ylim[0] if ylim[0] < -10 else -15, y2=-10, color='red', alpha=.3)
        ax3.fill_between(x=xlim, y1=-10, y2=10, color='yellow', alpha=.3)
        ax3.fill_between(x=xlim, y1=10, y2=ylim[1], color='green', alpha=.3)
        ax3.plot(np.arange(0, len(self.stnr))*2, self.stnr, color='dodgerblue', label="2-s window")

        axes[0].plot(np.arange(0, len(self.signal)) / self.fs, self.signal, color='black')
        axes[0].set_title("Measured Signal (.67Hz highpass)")
        axes[0].set_ylabel("Voltage")


ecg = SmitalProcessing("/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/"
                       "Converted/Collection 1/3LeadRun1.edf")
ecg.plot_data()
