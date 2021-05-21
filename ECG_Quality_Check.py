from ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors

from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy
import Filtering
import ImportEDF


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, raw_data, sample_rate, start_index, accel_data=None, accel_fs=1,
                 template_data='filtered', voltage_thresh=250, epoch_len=15):
        """Initialization method.

        :param
        -ecg_object: EcgData class instance created by ImportEDF script
        -random_data: runs algorithm on randomly-generated section of data; False by default.
                      Takes priority over start_index.
        -start_index: index for windowing data; 0 by default
        -epoch_len: window length in seconds over which algorithm is run; 15 seconds by default
        """

        self.voltage_thresh = voltage_thresh
        self.epoch_len = epoch_len
        self.fs = sample_rate
        self.start_index = start_index
        self.template_data = template_data

        self.raw_data = [i for i in raw_data[self.start_index:self.start_index+self.epoch_len*self.fs]]
        self.wavelet = None
        self.filt_squared = None

        self.accel_data = accel_data
        self.accel_fs = accel_fs

        self.qrs_fail = False

        self.index_list = np.arange(0, len(self.raw_data), self.epoch_len*self.fs)

        self.rule_check_dict = {"Valid Period": False,
                                "HR Valid": False, "HR": None,
                                "Max RR Interval Valid": False, "Max RR Interval": None,
                                "RR Ratio Valid": False, "RR Ratio": None,
                                "Voltage Range Valid": False, "Voltage Range": None,
                                "Correlation Valid": False, "Correlation": None,
                                "Accel Counts": None}

        # prep_data parameters
        self.r_peaks = None
        self.r_peaks_index_all = None
        self.rr_sd = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0

        # apply_rules parameters
        self.valid_hr = None
        self.valid_rr = None
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_qrs = None
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter(template_data=self.template_data)
            self.calculate_correlation()
            self.apply_rules()

        if self.valid_period:
            self.r_peaks_index_all = [peak + start_index for peak in self.r_peaks]

    def prep_data(self):
        """Function that:
        -Initializes ecgdetector class instance
        -Runs stationary wavelet transform peak detection
            -Implements 0.1-10Hz bandpass filter
            -DB3 wavelet transformation
            -Pan-Tompkins peak detection thresholding
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Initializes Detectors class instance with sample rate
        detectors = Detectors(self.fs)

        # Runs peak detection on raw data ----------------------------------------------------------------------------
        # Uses ecgdetectors package -> stationary wavelet transformation + Pan-Tompkins peak detection algorithm
        self.r_peaks, swt, squared = detectors.swt_detector(unfiltered_ecg=self.raw_data)

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40/60*self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats-1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40/60*self.epoch_len):
            self.enough_beats = False
            self.valid_period = False
            return

        # Calculates RR intervals in seconds -------------------------------------------------------------------------
        for peak1, peak2 in zip(self.r_peaks[:], self.r_peaks[1:]):
            rr_interval = (peak2 - peak1) / self.fs
            self.delta_rr.append(rr_interval)

        # Approach 1: median RR characteristics ----------------------------------------------------------------------
        # Calculates median RR-interval in seconds
        median_rr = np.median(self.delta_rr)

        # SD of RR intervals in ms
        self.rr_sd = np.std(self.delta_rr) * 1000

        # Converts median_rr to samples
        self.median_rr = int(median_rr * self.fs)

        # Removes any peak too close to start/end of data section: affects windowing later on ------------------------
        # Peak removed if within median_rr/2 samples of start of window
        # Peak removed if within median_rr/2 samples of end of window
        for i, peak in enumerate(self.r_peaks):
            if peak < (self.median_rr / 2 + 1) or (self.epoch_len * self.fs - peak) < (self.median_rr / 2 + 1):
                self.removed_peak.append(self.r_peaks.pop(i))
                self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.raw_data) - min(self.raw_data)

    def adaptive_filter(self, template_data="filtered"):
        """Method that runs an adaptive filter that generates the "average" QRS template for the window of data.

        - Calculates the median RR interval
        - Generates a sub-window around each peak, +/- RR interval/2 in width
        - Deletes the final beat sub-window if it is too close to end of data window
        - Calculates the "average" QRS template for the window
        """

        # Approach 1: calculates median RR-interval in seconds  -------------------------------------------------------
        # See previous method

        # Approach 2: takes a window around each detected R-peak of width peak +/- median_rr/2 ------------------------
        for peak in self.r_peaks:
            if template_data == "raw":
                window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "filtered":
                window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "wavelet":
                window = self.wavelet[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]

            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        correct_window_len = 2*int(self.median_rr/2)

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        if len(self.ecg_windowed[-1]) != correct_window_len:
            self.removed_peak.append(self.r_peaks.pop(-1))
            self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except (ValueError, TypeError):
            self.average_qrs = [0 for i in range(len(self.ecg_windowed[0]))]
            self.qrs_fail = True

    def calculate_correlation(self):
        """Method that runs a correlation analysis for each beat and the average QRS template.

        - Runs a Pearson correlation between each beat and the QRS template
        - Calculates the average individual beat Pearson correlation value
        - The period is deemed valid if the average correlation is >= 0.66, invalid is < 0.66
        """

        # Calculates correlation between each beat window and the average beat window --------------------------------
        for beat in self.ecg_windowed:

            r = stats.pearsonr(x=beat, y=self.average_qrs)
            self.beat_ppmc.append(abs(r[0]))

        self.average_r = float(np.mean(self.beat_ppmc))
        self.average_r = round(self.average_r, 3)

    def apply_rules(self):
        """First stage of algorithm. Checks data against three rules to determine if the window is potentially valid.
        -Rule 1: HR needs to be between 40 and 180bpm
        -Rule 2: no RR interval can be more than 3 seconds
        -Rule 3: the ratio of the longest to shortest RR interval is less than 2.2
        -Rule 4: the amplitude range of the raw ECG voltage must exceed n microV (approximate range for non-wear)
        -Rule 5: the average correlation coefficient between each beat and the "average" beat must exceed 0.66
        -Verdict: all rules need to be passed
        """

        # Rule 1: "The HR extrapolated from the sample must be between 40 and 180 bpm" -------------------------------
        if 40 <= self.hr <= 180:
            self.valid_hr = True
        else:
            self.valid_hr = False

        # Rule 2: "the maximum acceptable gap between successive R-peaks is 3s ---------------------------------------
        for rr_interval in self.delta_rr:
            if rr_interval < 3:
                self.valid_rr = True

            if rr_interval >= 3:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= 2.5:
            self.valid_ratio = False

        if self.rr_ratio < 2.5:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= 0.66:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.rule_check_dict = {"Valid Period": self.valid_period,
                                "HR Valid": self.valid_hr, "HR": round(self.hr, 1),
                                "Max RR Interval Valid": self.valid_rr, "Max RR Interval": round(max(self.delta_rr), 1),
                                "RR Ratio Valid": self.valid_ratio, "RR Ratio": round(self.rr_ratio, 1),
                                "Voltage Range Valid": self.valid_range, "Voltage Range": round(self.volt_range, 1),
                                "Correlation Valid": self.valid_corr, "Correlation": self.average_r,
                                "Accel Flatline": None}

        if self.accel_data is not None:
            accel_start = int(self.start_index / (self.fs / self.accel_fs))
            accel_end = accel_start + self.accel_fs * self.epoch_len

            svm = sum([i for i in self.accel_data["VM"].iloc[accel_start:accel_end]])
            self.rule_check_dict["Accel Counts"] = round(svm, 2)

            flatline = True if max(self.accel_data["VM"].iloc[accel_start:accel_end]) - \
                               min(self.accel_data["VM"].iloc[accel_start:accel_end]) <= .05 else False
            self.rule_check_dict["Accel Flatline"] = flatline

            sd = np.std(self.accel_data["VM"].iloc[accel_start:accel_end])
            self.rule_check_dict["Accel SD"] = sd


class RedmondQC:

    def __init__(self, ecg_signal, sample_rate, start_index=0, stop_index=None, epoch_len=25):
        """Runs algorithm from Redmond, Lovell, Basilakis, & Celler (2008) to flag regions of high- and low-quality
           ECG data. Original algorithm used 500Hz, single-lead ECG data recorded in 25-second segments.
           The stated accuracy is 89% sensitivity, 98% specificity, 98% PPV, and 97% NPV.

        :argument
        -ecg_signal: list/array of raw ECG data
        -sample_rate: Hz
        -start_index: corresponding to index in ecg_signal
        -stop_index: None or "end". If None, segment of data will be cropped using start_index and epoch_len.
                     If "end", will use entire file starting from start_index
        -epoch_len: interval with which data are processed. seconds.
        """

        self.epoch_len = epoch_len
        self.sample_rate = sample_rate
        self.start_index = start_index
        self.stop_index = start_index + sample_rate * epoch_len if stop_index is None else -1
        self.raw = ecg_signal[self.start_index:self.stop_index]

        self.filt = Filtering.filter_signal(data=self.raw, filter_type="bandpass", filter_order=5,
                                            low_f=.7, high_f=33, sample_f=self.sample_rate)

        self.clipping = None
        self.highf_mask = None
        self.highf_data = None
        self.lowf_mask = None
        self.lowf_data = None
        self.final_mask = None
        self.final_mask_epoch = None

        self.highf_thresh = 30
        self.lowf_thresh = 10

        """RUNS METHODS"""
        # self.clipping_mask()
        # self.high_freq_mask()
        # self.low_freq_mask(threshold=500)
        # self.combine_masks()

    def clipping_mask(self, voltage_range=2**15):
        """Checks if there is 99% clipping in one-second intervals. If any clipping is found,
           the surrounding +/- 1 second is flagged.

        :argument
        -voltage_range: magnitude of range of ECG sensor. E.g. if range is -10 to +10, use 10.
        """

        print('\nChecking data for 99% clipping...')

        max_val = voltage_range

        # Checks for values >= 99% of range (each data point)
        clip_mask = np.array([1 if abs(dp) >= max_val * .99 else 0 for dp in self.raw])

        # Loop that sets surrounding +/- 1-second intervals to clipping
        for i in range(0, len(clip_mask), self.sample_rate):

            if 1 in clip_mask[i:i+self.sample_rate]:

                # Regions after first second and before last second
                if self.sample_rate <= i <= len(clip_mask) - self.sample_rate:
                    clip_mask[i-self.sample_rate:i+self.sample_rate] = 1

                # First or last second of data
                if i < self.sample_rate or len(clip_mask) - i <= self.sample_rate:
                    clip_mask[i:i+self.sample_rate] = 1

        print("Complete.")

        self.clipping = clip_mask

    def high_freq_mask(self, threshold=30):
        """Checks for signal content in the high frequency range due to EMG artefact in 1-s increments.

           Procedure:
           -Runs 60Hz notch filter to remove AC noise
           -Data are filtered using a 5th order 40Hz highpass filter
           -Filtered data are squared
           -Lowpass filter using a .05-second hamming window filter
           -Square root is taken to return to original units
           -A mask is created using a threshold of 30 mV
        """

        print("\nChecking high-frequency content...")

        self.highf_thresh = threshold

        # 60Hz notch filter
        notch_filt = Filtering.filter_signal(data=self.raw, notch_f=60, filter_type='notch',
                                             sample_f=self.sample_rate, filter_order=5)

        # 40Hz 5th order highpass filter
        high_p = Filtering.filter_signal(data=notch_filt, filter_order=5, filter_type='highpass', high_f=40,
                                         sample_f=self.sample_rate)

        # Squares high-passed data
        high_p2 = high_p * high_p

        # Lowpass Hamming filter -- UNSURE IF CORRECT
        jump_size = int(.05 * self.sample_rate)

        """hamm = scipy.signal.hamming(M=jump_size)

        hamm_filt = []
        for i in range(0, len(high_p2)-jump_size, jump_size):

            data = high_p2[i:i+jump_size]

            data_hamm = data * hamm

            l = np.zeros(len(self.raw))

            l[i:i+jump_size] = data_hamm
            hamm_filt.append(l)

        hamm_data = np.array(hamm_filt).sum(axis=0)"""

        # Runs .05Hz lowpass filter cause I'm not sure what a ".05 second normalized Hamming window filter" is
        hamm_data = Filtering.filter_signal(data=high_p2, filter_order=5, filter_type='lowpass', low_f=.1,
                                            sample_f=self.sample_rate)

        # Square root
        square_root = np.sqrt(hamm_data)

        # Output. Above threshold is flagged for high_f
        high_mask = np.array([0 if i <= threshold else 1 for i in square_root])

        for i in range(0, len(high_mask), self.sample_rate):

            if 0 in high_mask[i:i+self.sample_rate]:
                high_mask[i:i+self.sample_rate] = 0

        self.highf_mask = high_mask
        self.highf_data = square_root

        print("Complete.")

    def low_freq_mask(self, threshold=10):
        """Checks for periods where there is too much low frequency content that lasts more than 3 seconds.

           Procedure:
           -Bandpass filter with passband of .7-33Hz
           -Signal is squared
           -Normalized .05-second hamming window filter
           -Square root is taken
           -Mask is created using threshold of 10 mV
           -Sections of low power less than 3 seconds long are ignored
        """

        print("\nChecking low-frequency content...")

        self.lowf_thresh = threshold

        # .7-33Hz BP filter
        # Squares filtered data
        squared = self.filt * self.filt

        """"
        # Hamming window
        jump_size = int(.05 * self.sample_rate)

        hamm = scipy.signal.hamming(M=jump_size)

        hamm_filt = []
        for i in range(0, len(squared) - jump_size, jump_size):
            data = squared[i:i + jump_size]

            data_hamm = data * hamm

            l = np.zeros(len(self.raw))

            l[i:i + jump_size] = data_hamm
            hamm_filt.append(l)

        hamm_data = np.array(hamm_filt).sum(axis=0)"""

        # Runs .05Hz lowpass filter cause I'm not sure what a ".05 second normalized Hamming window filter" is
        hamm_data = Filtering.filter_signal(data=squared, filter_order=5, filter_type='lowpass', low_f=.1,
                                            sample_f=self.sample_rate)

        # Square root or Hamming windowed data
        square_root = np.sqrt(hamm_data)

        # Binary list, compares values to threshold
        low_mask = np.array([1 if i <= threshold else 0 for i in square_root])

        # Removes low_f periods less than 3 seconds long -----------------------------
        starts = []
        stops = []

        # Finds start/stop indexes of low frequency periods
        for i in range(0, len(low_mask) - 1):
            if low_mask[i] == 0 and low_mask[i+1] == 1:
                starts.append(i+1)
            if low_mask[i] == 1 and low_mask[i+1] == 0:
                stops.append(i)

        # Adds index of final data point if collection ends with low frequency detected
        if len(starts) > len(stops) and low_mask[-1] == 1:
            stops.append(len(low_mask))

        # Removes low power regions less than 3 seconds long
        indexes = []
        for start, stop in zip(starts, stops):
            if (stop - start) / self.sample_rate >= 3:
                indexes.append([start, stop])

        low_mask_final = np.zeros(len(r.raw))

        # Sets detected periods more than 3s long to flagged
        for period in indexes:
            try:
                low_mask_final[period[0]:period[1]] = 1
            except IndexError:
                print(period)

        self.lowf_mask = low_mask_final
        self.lowf_data = square_root

        print("Complete.")

    def combine_masks(self):
        """Combines flags from clipping, high_freq, and low_freq masks to arrive at final verdict.
           Non-flagged sections of data less than 5-seconds long are included as flagged.
        """

        print("\nCombinging clipping, high-frequency, and low-frequency masks...")

        self.final_mask = [1 if clip == 1 or highf == 1 or lowf == 1 else 0 for
                           clip, highf, lowf in zip(self.clipping, self.highf_mask, self.lowf_mask)]

        """TRIES TO IGNORE VALID PERIODS LESS THAN 5-SECONDS LONG - CONTINUE WORKING"""
        """for i in range(1, len(self.final_mask) - 1):

            if len(self.final_mask) - i >= self.sample_rate:
                flagged_tally = self.final_mask[i:i + 5 * self.sample_rate].count("Valid")
                if flagged_tally == self.sample_rate * 5 and \
                        self.final_mask[i-1] == "Invalid" and \
                        self.final_mask[i + 5 * self.sample_rate + 1] == "Invalid":
                    self.final_mask[i:i + self.sample_rate * 5] == "Invalid"

            if len(self.final_mask) - i < self.sample_rate:
                flagged_tally = self.final_mask[i:].count("Valid")
                if flagged_tally == len(self.final_mask) - i:
                    self.final_mask[i:] == 'Invalid'"""

        print("Complete.")

    def plot_results(self):

        plt.close("all")

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex='col', figsize=(10, 7))
        plt.subplots_adjust(top=.9, right=.95)
        plt.suptitle("Index = {}".format(self.start_index))

        ax1.plot(np.arange(len(self.raw)) / self.sample_rate, self.raw, label='Raw', color='red')
        ax1.plot(np.arange(len(self.filt)) / self.sample_rate, self.filt, label='BPFilt', color='black')
        ax1.set_ylabel("Voltage")
        ax1.legend(loc="upper left")

        ax2.plot(np.arange(len(self.clipping)) / self.sample_rate, self.clipping, label='Clipping', color='black')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["No", "Yes"])
        ax2.set_ylabel("Power")
        ax2.legend(loc="upper left")

        ax3.plot(np.arange(0, len(self.raw))[:len(self.highf_mask)] / self.sample_rate, self.highf_mask,
                 label="HighF", color='dodgerblue')
        ax3.set_ylabel("Power")
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["Valid", "Invalid"])
        ax3.legend(loc="upper left")

        ax4.plot(np.arange(0, len(self.raw))[:len(self.lowf_mask)] / self.sample_rate, self.lowf_mask,
                 label="LowF", color='green')
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(["Valid", "Invalid"])
        ax4.legend(loc="upper left")

        ax5.plot(np.arange(0, len(self.raw))[:len(self.final_mask)] / self.sample_rate, self.final_mask,
                 label="Final", color='Purple')
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(["Valid", "Invalid"])
        ax5.legend(loc="upper left")
        ax5.set_xlabel("Seconds")

    def plot_processed_data(self, use_epoch_mask=False):

        fig, axes = plt.subplots(4, sharex='col', figsize=(10, 6))
        plt.subplots_adjust(top=.9, right=.95)
        plt.suptitle("Processed Data Used for Mask Thresholding")

        axes[0].plot(np.arange(len(self.raw)) / self.sample_rate, self.raw, label='Raw', color='red')
        axes[0].plot(np.arange(len(self.filt)) / self.sample_rate, self.filt, label='BPFilt', color='black')
        axes[0].set_ylabel("Voltage")
        axes[0].legend(loc="upper left")

        axes[1].plot(np.arange(0, len(self.highf_data))[:len(self.highf_data)] / self.sample_rate, self.highf_data,
                     label="HighF", color='dodgerblue')
        axes[1].axhline(self.highf_thresh, color='black', linestyle='dashed')
        axes[1].legend(loc="upper left")
        axes[1].set_ylabel("Power")

        axes[2].plot(np.arange(len(self.lowf_data)) / self.sample_rate, self.lowf_data,
                     label='LowF', color='green')
        axes[2].set_ylabel("Power")
        axes[2].axhline(self.lowf_thresh, color='black', linestyle='dashed')
        axes[2].legend(loc="upper left")

        if use_epoch_mask:
            axes[3].plot(np.arange(0, len(self.raw))[:len(self.final_mask)] / self.sample_rate, self.final_mask,
                         label="Final Raw", color='Purple')
        if not use_epoch_mask:
            axes[3].plot(np.arange(0, len(self.final_mask_epoch))[:len(self.final_mask_epoch)]/self.epoch_len,
                         self.final_mask_epoch,
                         label="Final Epoched", color='Purple')
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(["Valid", "Invalid"])
        axes[3].legend(loc="upper left")
        axes[-1].set_xlabel("Seconds")

    def epoch_data(self):

        final = np.zeros(len(np.arange(0, len(self.final_mask), int(self.epoch_len * self.sample_rate))))

        epoch_ind = 0
        for i in np.arange(0, len(self.final_mask), int(self.epoch_len * self.sample_rate)):
            if 1 in self.final_mask[i:i+int(self.epoch_len * self.sample_rate)]:
                final[epoch_ind] = 1
            epoch_ind += 1

        self.final_mask_epoch = final


# ================================================= RUNS CODE =========================================================

# Imports data
data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/"
                                  "Student Supervision/Winter 2021/"
                                  "Kin 472 - Megan/Data/Converted/Collection 1/3LeadRun1.edf",
                         start_offset=0, end_offset=0, epoch_len=5, load_accel=False,
                         low_f=1, high_f=30, f_type="bandpass")

r = RedmondQC(ecg_signal=data.raw, sample_rate=250, start_index=0, stop_index="end", epoch_len=1)

del data

r.clipping_mask(voltage_range=32768)  # Need to make into 1s windows
r.high_freq_mask(threshold=60)
r.low_freq_mask(threshold=10)
r.combine_masks()
r.epoch_data()

r.plot_processed_data(use_epoch_mask=False)
# r.plot_results()

# TODO
# Figure out Hamming window in high/low frequency masks

"""================================================================================================================="""


def run_redmond(epoch_len=10, plot_data=False):

    if plot_data:

        fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))

        axes[0].set_ylabel("Voltage")
        axes[0].set_title("Raw")

        axes[1].set_title("Redmond Algorithm Results")
        axes[1].set_ylabel("Final Mask")
        axes[1].set_xlabel("Seconds")

    redmond_data = []

    # Loops through data in epochs
    epoch_ind = 0
    tally_ind = 0
    markers = np.arange(0, len(data.raw) * 1.09, int(len(data.raw) / 10))

    for i in range(0, len(data.raw), int(epoch_len * data.sample_rate)):

        if markers[tally_ind] % i <= int(epoch_len * data.sample_rate):
            print("{}% done...".format(tally_ind * 10))
            tally_ind += 1

        r = RedmondQC(ecg_signal=data.raw[i:], sample_rate=data.sample_rate,
                      start_index=0, stop_index=None, epoch_len=epoch_len)

        # Values for each datapoint if it's valid or not
        for j in r.final_mask:
            redmond_data.append(j)

        if plot_data:
            x = np.arange(i, i+r.sample_rate*epoch_len if
                            i+r.sample_rate*epoch_len < len(data.raw) else len(data.raw))
            y = data.raw[i:i+data.sample_rate*epoch_len]

            axes[0].plot([k/data.sample_rate for k in x[::3]], y[::3],
                         color='grey' if epoch_ind % 2 == 0 else "black")
            axes[1].plot([k/data.sample_rate for k in x[::3]], r.final_mask[::3], color='red')

        epoch_ind += 1

    # Raw data: invalid regions are assigned a value of 0
    raw = np.array(data.raw)

    for i, dp in enumerate(redmond_data):
        if dp == "Invalid":
            raw[i] = 0

    return redmond_data, raw


#  redmond_data, redmond_raw = run_redmond(epoch_len=15, plot_data=False)


def find_all_peaks(raw_data, epoch_len, fs=250, plot_data=False, use_epochs=True):

    all_peaks = []
    swt_filt = []
    filt_sq = []

    if use_epochs:
        for i in range(0, len(raw_data), int(fs*epoch_len)):

            detectors = Detectors(fs)
            r_peaks, swt, squared = detectors.swt_detector(unfiltered_ecg=raw_data[i:i+int(fs*epoch_len)])

            for r in r_peaks:
                all_peaks.append(r + i)

            for s, sq in zip(swt, squared):
                swt_filt.append(s)
                filt_sq.append(sq)

    if not use_epochs:
        detectors = Detectors(fs)
        all_peaks, swt_filt, filt_sq = detectors.swt_detector(unfiltered_ecg=raw_data)
        # all_peaks = detectors.swt_detector(unfiltered_ecg=raw_data)

    if plot_data:

        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))
        fig.subplots_adjust(hspace=.35)
        axes[0].set_title("Raw")
        axes[1].set_title("Wavelet")
        axes[2].set_title("Wavelet Filtered")

        if use_epochs:
            for i in np.arange(0, len(raw_data), int(fs*epoch_len)):
                axes[0].axvline(i/fs, linestyle="dashed", color='green')

                axes[1].axvline(i/fs, linestyle="dashed", color='green')
                axes[2].axvline(i/fs, linestyle="dashed", color='green')

        axes[0].plot(np.arange(len(raw_data))[::3]/fs, raw_data[::3], color='black')
        axes[0].plot([i/fs for i in all_peaks], [raw_data[i] for i in all_peaks],
                 linestyle="", marker="o", color='dodgerblue', markersize=4)

        length = min([len(np.arange(len(raw_data))/fs), len(swt_filt), len(filt_sq)])

        axes[1].plot(np.arange(len(raw_data))[:length:3]/fs, swt_filt[:length:3], color='black')

        axes[1].plot([i / fs for i in all_peaks], [swt_filt[i] for i in all_peaks],
                     linestyle="", marker="o", color='dodgerblue', markersize=4)

        axes[2].plot(np.arange(len(raw_data))[:length:3]/fs, filt_sq[:length:3], color='black')

        axes[2].plot([i / fs for i in all_peaks], [filt_sq[i] for i in all_peaks],
                     linestyle="", marker="o", color='dodgerblue', markersize=4)

    return all_peaks, swt_filt, filt_sq


# Individual algorithms -----------------------------------------------------------------------------------------------

"""
orphanidou = []
redmond = []
for i in range(0, len(data.raw), int(data.sample_rate * 10)):
    d = CheckQuality(raw_data=data.raw, sample_rate=data.sample_rate, start_index=i,
                     template_data='raw', voltage_thresh=250, epoch_len=10)
    orphanidou.append("Valid" if d.valid_period else "Invalid")

    r = RedmondQC(ecg_signal=data.raw, sample_rate=data.sample_rate, start_index=i, stop_index=None, epoch_len=10)
    for j in r.final_mask:
        redmond.append(j)
"""

# Combined algorithms -------------------------------------------------------------------------------------------------
"""
final = np.array(["Invalid" for i in range(len(data.raw))])

for i in range(0, len(data.raw), data.sample_rate * data.epoch_len):

    # Always checks Orphanidou algorithm
    o = CheckQuality(raw_data=data.raw, sample_rate=data.sample_rate, start_index=i,
                              template_data='raw', voltage_thresh=250, epoch_len=10)

    # If Orphanidou finds invalid and low voltage (i.e. non-wear), period is invalid
    if not o.valid_period and o.volt_range < o.voltage_thresh:
        final[i:i+data.sample_rate*data.epoch_len] = "Invalid"
        # pass

    # If Orphanidou finds invalid with high enough voltage (i.e. wear), checks Redmond algorithm
    # THIS LETS THROUGH USELESS DATA --> MAY NOT FIND HEARTBEATS THOUGH --> investigate!
    if not o.valid_period and o.volt_range >= o.voltage_thresh:
        r = RedmondQC(ecg_signal=data.raw, sample_rate=data.sample_rate, start_index=i, stop_index=None, epoch_len=10)

        final[i:i+data.sample_rate*data.epoch_len] = r.final_mask

    # Valid period if Orphanidou finds it valid
    if o.valid_period:
        final[i:i + data.sample_rate * data.epoch_len] = "Valid"

# Turns invalid datapoints into 0
raw = np.array(data.raw)
code = np.array(final)

for i, val in enumerate(code):
    if val == "Invalid":
        raw[i] = 0

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 7))
plt.subplots_adjust(hspace=.35, top=.95)

ax1.plot(data.timestamps[::3], raw[::3], color='red')
ax1.set_title("Raw ECG")
ax1.set_ylabel("Voltage")

ax2.set_title("Remond Quality Check Algorithm")
ax2.fill_between(x=data.timestamps, y1="Valid", y2=redmond, color='green', label="Redmond", alpha=.5)

ax3.set_title("Orphanidou Quality Check Algorithm")
ax3.fill_between(x=data.epoch_timestamps, y1="Valid", y2=orphanidou, color='purple', label="Orphanidou", alpha=.5)

ax4.set_title("Combined Quality Check Algorithm")
ax4.fill_between(x=data.timestamps, y1='Valid', y2=final, color='dodgerblue', label="Combined", alpha=.5)

"""
