import ImportEDF
import ECG_Quality_Check

from ecgdetectors.ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
from datetime import datetime
from matplotlib.ticker import PercentFormatter
from random import randint
import matplotlib.dates as mdates
import os
import pyedflib

# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- ECG CLASS OBJECT ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class ECG:

    def __init__(self, raw_filepath=None, processed_filepath=None, output_dir=None,
                 ecg_downsample=1, load_raw=True, from_processed=False,
                 qc_algorithm="Orphanidou",
                 age=0, start_offset=0, end_offset=0,
                 rest_hr_window=60, n_epochs_rest=10,
                 epoch_len=15, load_accel=False,
                 filter_data=False, low_f=1, high_f=30, f_type="bandpass"):

        print()
        print("============================================= ECG DATA ==============================================")

        # Parameters -------------------------------------------------------------------------------------------------
        self.filepath = raw_filepath
        self.proc_filepath = processed_filepath
        self.output_dir = output_dir

        self.qc_algorithm = qc_algorithm

        self.subject_id = self.extract_subject_id()
        self.age = age
        self.epoch_len = epoch_len
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest = n_epochs_rest
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.filter_data = filter_data
        self.low_f = low_f
        self.high_f = high_f
        self.f_type = f_type

        self.load_raw = load_raw
        self.ecg_downsample = ecg_downsample
        self.load_accel = load_accel
        self.from_processed = from_processed
        self.valid_accel = False

        # Data -------------------------------------------------------------------------------------------------------
        self.timestamps = None
        self.sample_rate = None

        self.df_epoch = None
        self.df_raw = None

        self.accel_sample_rate = 1
        self.df_accel = None

        self.r_peaks = None
        self.qrs_fail_index = None

        # RUNS METHODS -----------------------------------------------------------------------------------------------

        # Raw data
        if self.load_raw:
            self.load_raw_data()

        # Loads epoched data from existing file
        if self.from_processed:
            self.df_epoch = self.load_processed()

        # Performs quality control check on raw data and epochs data
        if not self.from_processed:
            if self.qc_algorithm.capitalize() == "Orphanidou":
                self.check_quality_orphanidou()
            if self.qc_algorithm.capitalize() == "Redmond":
                self.check_quality_redmond()

        self.quality_report = self.generate_quality_report()

        self.rest_hr, self.awake_hr = self.find_resting_hr(window_size=self.rest_hr_window,
                                                           n_windows=self.n_epochs_rest)

        self.calculate_percent_hrr()
        self.calculate_intensity()

        self.epoch_intensity = None
        self.epoch_intensity_totals = None

        self.nonwear = None

    def extract_subject_id(self):

        fname_parts = self.filepath.split("/")[-1].split(".")[0].split("_")
        subject_id = None

        for part in fname_parts:
            try:
                int(part)
                if len(part) == 4:
                    subject_id = int(part)
                    break
            except ValueError:
                pass

        return subject_id

    def load_raw_data(self):

        ecg = ImportEDF.Bittium(filepath=self.filepath, load_accel=self.load_accel,
                                start_offset=self.start_offset, end_offset=self.end_offset,
                                low_f=self.low_f, high_f=self.high_f, f_type=self.f_type)

        self.sample_rate = ecg.sample_rate
        self.accel_sample_rate = ecg.accel_sample_rate
        raw = ecg.raw
        filtered = ecg.filtered
        timestamps = ecg.timestamps

        if self.ecg_downsample != 1:
            print("\n-ECG data will be downsampled by a factor of "
                  "{} to {}Hz...".format(self.ecg_downsample, round(self.sample_rate/self.ecg_downsample, 1)))
            self.sample_rate = int(self.sample_rate / self.ecg_downsample)

            timestamps = timestamps[::self.ecg_downsample]
            raw = raw[::self.ecg_downsample]
            filtered = filtered[::self.ecg_downsample]

        self.df_raw = pd.DataFrame(list(zip(timestamps, raw, filtered)), columns=["Timestamp", "Raw", "Filtered"])

        if not self.from_processed:
            t = [i for i in self.df_raw["Timestamp"].iloc[::self.sample_rate * self.epoch_len]]
            self.df_epoch = pd.DataFrame(list(zip(t)), columns=["Timestamp"])

        if self.load_accel:
            self.df_accel = pd.DataFrame(list(zip(self.df_raw["Timestamp"].
                                                  iloc[::int(self.sample_rate / self.accel_sample_rate)],
                                              ecg.x, ecg.y, ecg.z, ecg.vm)),
                                         columns=["Timestamp", "X", "Y", "Z", "VM"])

            # Calculates accel activity counts
            self.epoch_accel(vm_data=ecg.vm)

            f = pyedflib.EdfReader(self.filepath)
            if f.getSignalHeader(1)["transducer"] == "X-axis":
                self.valid_accel = True
            f.close()

    def load_processed(self):
        """Requires 3-column csv file. Rows are epochs. Columns are 'Timestamps', "ECG_Validity' and 'HR'. """

        print("\nLoading data from processed...")

        df = pd.read_csv(self.proc_filepath)

        df["Timestamp"] = [datetime.strptime(i.split(".")[0], "%Y-%m-%d %H:%M:%S") for i in df["Timestamp"]]

        return df

    def write_epoched_data(self, output_dir=None):

        if output_dir is None:
            if self.output_dir is not None:
                output_dir = self.output_dir
            if self.output_dir is None:
                output_dir = os.getcwd()

        output_loc = output_dir + self.proc_filepath.split("/")[-1]

        if os.path.exists(output_loc):
            overwrite = input("File already exists. Overwrite? y/n: ")

        if not os.path.exists(output_loc):
            overwrite = "y"

        if overwrite == "n" or overwrite == "N" or overwrite == "No" or overwrite == "no":
            print("File will not be written.")

        if overwrite == "y" or overwrite == "Y" or overwrite == "Yes" or overwrite == "yes":
            print("Overwriting file ({})".format(output_loc))
            self.df_epoch.to_csv(output_loc, index=False)

    def epoch_accel(self, vm_data):

        if self.load_raw:

            svm = []
            avm = []
            for i in range(0, len(vm_data), int(self.accel_sample_rate * self.epoch_len)):

                if i + self.epoch_len * self.accel_sample_rate > len(vm_data):
                    break

                vm_sum = sum(vm_data[i:i + self.epoch_len * self.accel_sample_rate])
                vm_avg = np.mean(vm_data[i:i + self.epoch_len * self.accel_sample_rate])

                svm.append(round(vm_sum, 5))
                avm.append(round(vm_avg, 5))

            if self.df_epoch is not None:
                if len(svm) < self.df_epoch.shape[0]:
                    for i in range(self.df_epoch.shape[0] - len(svm)):
                        svm.append(0)
                        avm.append(0)

                    self.df_epoch["SVM"] = svm[0:self.df_epoch.shape[0]]
                    self.df_epoch["AVM"] = avm[0:self.df_epoch.shape[0]]

            if self.df_epoch is None:
                self.df_epoch = pd.DataFrame(list(zip(svm, avm)), columns=["SVM", "AVM"])

    def check_quality_orphanidou(self):
        """Performs quality check using Orphanidou et al. (2015) algorithm that has been tweaked to factor in voltage
           range as well.

           This function runs a loop that creates object from the class CheckQuality for each epoch in the raw data.
        """

        print("\n" + "Running quality check with Orphanidou et al. (2015) algorithm...")

        t0 = datetime.now()

        validity_list = []  # window's validity (binary; 1 = invalid)
        epoch_hr = []  # window's HRs
        volt_range = []  # window's voltage range
        rr_sd = []  # window's RR SD
        r_peaks = []  # all R peak indexes
        qrs_fail_index = []

        for start_index in range(0, self.df_raw.shape[0], self.epoch_len * self.sample_rate):
            print(round(100*start_index/self.df_raw.shape[0], 1))

            qc = ECG_Quality_Check.CheckQuality(raw_data=[i for i in self.df_raw["Raw"]],
                                                start_index=start_index, epoch_len=self.epoch_len,
                                                sample_rate=self.sample_rate)

            volt_range.append(qc.volt_range)

            if qc.valid_period:
                validity_list.append("Valid")
                epoch_hr.append(round(qc.hr, 2))
                rr_sd.append(qc.rr_sd)

                for peak in qc.r_peaks_index_all:
                    r_peaks.append(peak)
                for peak in qc.removed_peak:
                    r_peaks.append(peak + start_index)

                r_peaks = sorted(r_peaks)

            if not qc.valid_period:
                validity_list.append("Invalid")
                epoch_hr.append(0)  # originally 0
                rr_sd.append(0)  # Originally 0

            if qc.qrs_fail:
                qrs_fail_index.append(start_index)

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Quality check complete ({} seconds).".format(round(proc_time, 2)))
        print("-Processing time of {} seconds per "
              "hour of data.".format(round(proc_time / (self.df_raw.shape[0]/self.sample_rate/3600)), 2))

        if len(validity_list) < self.df_epoch.shape[0]:
            for i in range(self.df_epoch.shape[0] - len(validity_list)):
                validity_list.append("Invalid")
                epoch_hr.append(None)
                volt_range.append(0)
                rr_sd.append(None)
        if len(validity_list) > self.df_epoch.shape[0]:
            validity_list = validity_list[:self.df_epoch.shape[0]]
            epoch_hr = epoch_hr[:self.df_epoch.shape[0]]
            volt_range = volt_range[:self.df_epoch.shape[0]]
            rr_sd = rr_sd[:self.df_epoch.shape[0]]

        self.df_epoch["ECG_Validity"] = validity_list
        self.df_epoch["HR"] = [i if i != 0 else None for i in epoch_hr]
        self.df_epoch["Voltage Range"] = volt_range
        self.df_epoch["RR_SD"] = [i if i != 0 else None for i in rr_sd]

        self.r_peaks = r_peaks
        self.qrs_fail_index = qrs_fail_index

    def check_quality_redmond(self):
        pass

    def generate_quality_report(self):
        """Calculates how much of the data was usable. Returns values in dictionary."""

        invalid_epochs = self.df_epoch.loc[self.df_epoch["ECG_Validity"] == "Invalid"].shape[0]
        hours_lost = round(invalid_epochs / (60 / self.epoch_len) / 60, 2)  # hours of invalid data
        perc_invalid = round(invalid_epochs / self.df_epoch.shape[0] * 100, 1)  # percent of invalid data

        quality_report = {"Invalid epochs": invalid_epochs,
                          "Hours lost": hours_lost,
                          "Percent invalid": perc_invalid}

        print("-{}% of the data is valid.".format(round(100 - perc_invalid), 3))

        return quality_report

    def find_resting_hr(self, window_size=60, n_windows=30, sleep_status=None, start_index=None, end_index=None):
        """Function that calculates resting HR based on inputs.

        :argument
        -window_size: size of window over which rolling average is calculated, seconds
        -n_windows: number of epochs over which resting HR is averaged (lowest n_windows number of epochs)
        -sleep_status: data from class Sleep that corresponds to asleep/awake epochs
        """

        if start_index is not None and end_index is not None:
            epoch_hr = np.array(self.df_epoch["HR"].iloc[start_index:end_index])
        else:
            epoch_hr = np.array(self.df_epoch["HR"])

        try:
            epoch_hr = [i if not np.isnan(i) else 0 for i in epoch_hr]
        except TypeError:
            epoch_hr = [i for i in self.df_epoch["HR"] if i is not None]

        # Sets integer for window length based on window_size and epoch_len
        window_len = int(window_size / self.epoch_len)

        try:
            rolling_avg = [statistics.mean(epoch_hr[i:i + window_len]) if 0 not in epoch_hr[i:i + window_len]
                           else None for i in range(len(epoch_hr))]

        except statistics.StatisticsError:
            print("No data points found.")
            rolling_avg = []

        # Calculates resting HR during waking hours if sleep_log available --------------------------------------------
        if sleep_status is not None:
            print("\n" + "Calculating resting HR from periods of wakefulness...")

            awake_hr = [rolling_avg[i] for i in range(0, min([len(sleep_status), len(rolling_avg)]))
                        if sleep_status[i] == 0 and rolling_avg[i] is not None]

            sorted_hr = sorted(awake_hr)

            if len(sorted_hr) < n_windows:
                resting_hr = "N/A"

            if len(sorted_hr) >= n_windows:
                resting_hr = round(sum(sorted_hr[0:n_windows]) / n_windows, 1)

            print("Resting HR (average of {} lowest {}-second periods while awake) is {} bpm.".format(n_windows,
                                                                                                      window_size,
                                                                                                      resting_hr))

        # Calculates resting HR during all hours if sleep_log not available -------------------------------------------
        if sleep_status is None:
            print("\n" + "Calculating resting HR from periods of all data (sleep data not available)...")

            awake_hr = None

            valid_hr = [i for i in rolling_avg if i is not None]

            sorted_hr = sorted(valid_hr)

            resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

            print("Resting HR (sleep not removed; average of {} lowest "
                  "{}-second periods) is {} bpm.".format(n_windows, window_size, resting_hr))

        return resting_hr, awake_hr

    def calculate_percent_hrr(self):
        """Calculates HR as percent of heart rate reserve using resting heart rate and predicted HR max using the
           equation from Tanaka et al. (2001).
           Removes negative %HRR values which are possible due to how resting HR is defined.
        """

        hr_max = 208 - 0.7 * self.age

        try:
            perc_hrr = [round(100 * (hr - self.rest_hr) / (hr_max - self.rest_hr), 2) if
                        not np.isnan(hr) else None for hr in self.df_epoch["HR"]]
        except TypeError:
            perc_hrr = [round(100 * (hr - self.rest_hr) / (hr_max - self.rest_hr), 2) if
                        hr is not None else None for hr in self.df_epoch["HR"]]

        # A single epoch's HR can be below resting HR based on how it's defined
        # Changes any negative values to 0, maintains Nones and positive values
        # Can't figure out how to do this as a list comprehension - don't judge
        perc_hrr_final = []

        for i in perc_hrr:
            if i is not None:
                if i >= 0:
                    perc_hrr_final.append(i)
                if i < 0:
                    perc_hrr_final.append(0)
            if i is None:
                perc_hrr_final.append(None)

        if len(perc_hrr_final) < self.df_epoch.shape[0]:
            for i in range(self.df_epoch.shape[0] - len(perc_hrr_final)):
                perc_hrr_final.append(None)

        self.df_epoch["HRR"] = perc_hrr_final[:self.df_epoch.shape[0]]

    def calculate_intensity(self):
        """Calculates intensity category based on %HRR ranges.
           Sums values to determine total time spent in each category.

        :returns
        -intensity: epoch-by-epoch categorization by intensity. 0=sedentary, 1=light, 2=moderate, 3=vigorous
        -intensity_minutes: total minutes spent at each intensity, dictionary
        """

        # INTENSITIY DEFINITIONS
        # Sedentary = %HRR < 30, light = 30 < %HRR <= 40, moderate = 40 < %HRR <= 60, vigorous = %HRR >= 60

        intensity = []

        for hrr in self.df_epoch["HRR"]:
            if hrr is None:
                intensity.append(None)

            if hrr is not None:
                if hrr < 30:
                    intensity.append(0)
                if 30 <= hrr < 40:
                    intensity.append(1)
                if 40 <= hrr < 60:
                    intensity.append(2)
                if hrr >= 60:
                    intensity.append(3)

        n_valid_epochs = self.df_epoch.shape[0] - self.quality_report["Invalid epochs"]

        if n_valid_epochs == 0:
            n_valid_epochs = self.df_epoch.shape[0]

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.epoch_len),
                            "Sedentary%": round(intensity.count(0) / n_valid_epochs, 3),
                            "Light": intensity.count(1) / (60 / self.epoch_len),
                            "Light%": round(intensity.count(1) / n_valid_epochs, 3),
                            "Moderate": intensity.count(2) / (60 / self.epoch_len),
                            "Moderate%": round(intensity.count(2) / n_valid_epochs, 3),
                            "Vigorous": intensity.count(3) / (60 / self.epoch_len),
                            "Vigorous%": round(intensity.count(3) / n_valid_epochs, 3)}

        print("\n" + "HEART RATE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return intensity, intensity_totals

    def plot_histogram(self):
        """Generates a histogram of heart rates over the course of the collection with a bin width of 5 bpm.
           Marks calculated average and resting HR."""

        # Data subset: only valid HRs
        valid_heartrates = [i for i in self.df_epoch["HR"] if not np.isnan(i)]
        avg_hr = sum(valid_heartrates) / len(valid_heartrates)

        # Bins of width 5bpm between 40 and 180 bpm
        n_bins = np.arange(40, 180, 5)

        plt.figure(figsize=(10, 7))
        plt.hist(x=valid_heartrates, weights=np.ones(len(valid_heartrates)) / len(valid_heartrates), bins=n_bins,
                 edgecolor='black', color='grey')
        plt.axvline(x=avg_hr, color='red', linestyle='dashed', label="Average HR ({} bpm)".format(round(avg_hr, 1)))
        plt.axvline(x=self.rest_hr, color='green', linestyle='dashed',
                    label='Calculated resting HR ({} bpm)'.format(round(self.rest_hr, 1)))

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.ylabel("% of Epochs")
        plt.xlabel("HR (bpm)")
        plt.title("Heart Rate Histogram")
        plt.legend(loc='upper left')
        plt.show()

    def plot_qc_segment(self, input_index=None, template_data='filtered'):
        """Method that generates a random 10-minute sample of data. Overlays filtered data with quality check output.

        :argument
        -start_index: able to input desired start index. If None, randomly generated
        """

        # Generates random start index
        if input_index is not None:
            start_index = input_index
        if input_index is None:
            start_index = randint(0, self.df_raw.shape[0] - self.epoch_len * self.sample_rate)

        # Rounds random start to an index that corresponds to start of an epoch
        start_index -= start_index % (self.epoch_len * self.sample_rate)

        print("\n" + "Index {}.".format(start_index))

        # End index: one epoch
        end_index = start_index + self.epoch_len * self.sample_rate

        # Data point index converted to seconds
        seconds_seq_raw = np.arange(0, self.epoch_len * self.sample_rate) / self.sample_rate

        # Epoch's quality check
        validity_data = ECG_Quality_Check.CheckQuality(ecg_object=self, start_index=start_index,
                                                       epoch_len=self.epoch_len, template_data=template_data)

        print()
        print("Valid HR: {} (passed {}/5 conditions)".format(validity_data.rule_check_dict["Valid Period"],
                                                             validity_data.rule_check_dict["HR Valid"] +
                                                             validity_data.rule_check_dict["Max RR Interval Valid"] +
                                                             validity_data.rule_check_dict["RR Ratio Valid"] +
                                                             validity_data.rule_check_dict["Voltage Range Valid"] +
                                                             validity_data.rule_check_dict["Correlation Valid"]))

        print("-HR range ({} bpm): {}".format(validity_data.rule_check_dict["HR"],
                                              validity_data.rule_check_dict["HR Valid"]))
        print("-Max RR interval ({} sec): {}".format(validity_data.rule_check_dict["Max RR Interval"],
                                                     validity_data.rule_check_dict["Max RR Interval Valid"]))
        print("-RR ratio ({}): {}".format(validity_data.rule_check_dict["RR Ratio"],
                                          validity_data.rule_check_dict["RR Ratio Valid"]))
        print("-Voltage range ({} uV): {}".format(validity_data.rule_check_dict["Voltage Range"],
                                                  validity_data.rule_check_dict["Voltage Range Valid"]))
        print("-Correlation (r={}): {}".format(validity_data.rule_check_dict["Correlation"],
                                               validity_data.rule_check_dict["Correlation Valid"]))

        # Plot

        plt.close("all")

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 7))

        valid_period = "Valid" if validity_data.rule_check_dict["Valid Period"] else "Invalid"

        ax1.set_title("Participant {}: {} (index = {})".format(self.subject_id, valid_period, start_index))

        # Filtered ECG data
        ax1.plot(seconds_seq_raw, validity_data.filt_data, color='black', label="Filt. ECG")
        ax1.plot([i / self.sample_rate for i in validity_data.r_peaks],
                 [validity_data.filt_data[peak] for peak in validity_data.r_peaks],
                 linestyle="", marker="o", color='limegreen', markersize=4)
        ax1.set_ylabel("Voltage")
        ax1.legend(loc='upper left')

        for peak in validity_data.removed_peak:
            ax1.plot(np.arange(0, len(validity_data.filt_data))[peak] / self.sample_rate,
                     validity_data.filt_data[peak], marker="x", color='red')

        for i, window in enumerate(validity_data.ecg_windowed):
            ax2.plot(np.arange(0, len(window))/self.sample_rate, window, color='black')

        ax2.plot(np.arange(0, len(validity_data.average_qrs))/self.sample_rate, validity_data.average_qrs,
                 label="QRS template ({} data; r={})".format(template_data, validity_data.average_r),
                 color='limegreen', linestyle='dashed')

        ax2.legend()
        ax2.set_ylabel("Voltage")
        ax2.set_xlabel("Seconds")

        return validity_data

    def plot_validity(self, downsample=1):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))

        if downsample > 1:
            ax1.plot(self.df_raw["Timestamp"].iloc[::downsample], self.df_raw["Raw"].iloc[::downsample],
                     color='red')
        if downsample == 1:
            ax1.plot(self.df_raw["Timestamp"], self.df_raw["Raw"], color='red')
        ax1.set_ylabel("Voltage")

        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["HRR"], color='green')
        ax2.set_ylabel("%HRR")

        ax3.plot(self.df_epoch["Timestamp"], self.df_epoch['ECG_Validity'], color='grey')

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax3.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def calculate_nonwear(self, epoch_len=15, plot_data=True):

        def find_nonwear():
            # First accel check: SD and range below threshold calculations -------------------------------------------
            print("\nPerforming non-wear detection algorithm...")

            accel_nw = []

            for i in np.arange(0, len(self.accel_x), self.accel_sample_rate * epoch_len):
                sd_x = np.std(self.accel_x[i:i + self.accel_sample_rate * epoch_len])
                sd_y = np.std(self.accel_y[i:i + self.accel_sample_rate * epoch_len])
                sd_z = np.std(self.accel_z[i:i + self.accel_sample_rate * epoch_len])
                axes_below_thresh = int(sd_x <= 3) + int(sd_y <= 3) + int(sd_z <= 3)

                range_x = max(self.accel_x[i:i + self.accel_sample_rate * epoch_len]) - \
                          min(self.accel_x[i:i + self.accel_sample_rate * epoch_len])
                range_y = max(self.accel_y[i:i + self.accel_sample_rate * epoch_len]) - \
                          min(self.accel_y[i:i + self.accel_sample_rate * epoch_len])
                range_z = max(self.accel_z[i:i + self.accel_sample_rate * epoch_len]) - \
                          min(self.accel_z[i:i + self.accel_sample_rate * epoch_len])

                axes_below_range = int(range_x <= 50) + int(range_y <= 50) + int(range_z <= 50)

                if axes_below_range >= 2 or axes_below_thresh >= 2:
                    accel_nw.append("Nonwear")
                else:
                    accel_nw.append("Wear")

            # Combines accelerometer and ECG non-wear characteristics: epoch-by-epoch ---------------------------------
            df_ecg = pd.DataFrame(list(zip(self.epoch_timestamps, self.epoch_validity,
                                           self.avg_voltage, self.svm, accel_nw)),
                                  columns=["Stamp", "Validity", "VoltRange", "SVM", "AccelNW"])

            nw = []
            for epoch in df_ecg.itertuples():
                if epoch.Validity == "Invalid" and epoch.AccelNW == "Nonwear" and epoch.VoltRange <= 400:

                    # Confirms using FFT that it's a non-wear period
                    # Gets confused with bad ECG signal during sleep sometimes
                    fft, power_cutoff = self.run_ecg_fft(start=epoch.Index * self.epoch_len * self.sample_rate,
                                                         seg_length=epoch_len, show_plot=False)

                    if power_cutoff >= 50:
                        nw.append("Nonwear")
                    if power_cutoff < 50:
                        nw.append("Wear")
                else:
                    nw.append("Wear")

            # 5-minute windows ----------------------------------------------------------------------------------------
            t0 = datetime.now()
            final_nw = np.zeros(len(nw))
            for i in range(len(nw)):

                if final_nw[i] == "Wear" or final_nw[i] == "Nonwear":
                    pass

                if nw[i:i + 20].count("Nonwear") >= 19:
                    final_nw[i:i + 20] = 1

                    for j in range(i, len(nw)):
                        if nw[j] == "Nonwear":
                            pass
                        if nw[j] == "Wear":
                            stop_ind = j
                            if j > i:
                                final_nw[i:stop_ind] = 1

                else:
                    final_nw[i] = 0

            final_nw = ["Nonwear" if i == 1 else "Wear" for i in final_nw]
            t1 = datetime.now()
            print("Algorithm time = {} seconds.".format(round((t1 - t0).total_seconds(), 1)))

            return final_nw

        if self.nonwear is None:
            final_nw = find_nonwear()

        if self.nonwear is not None:
            print("Data already exists. Using previous data.")
            final_nw = self.nonwear

        if plot_data:

            print("Generating plot...")

            manual_log = pd.read_excel("/Users/kyleweber/Desktop/ECG Non-Wear/OND07_VisuallyInspectedECG_Nonwear.xlsx")
            manual_log = manual_log.loc[manual_log["ID"] == self.subject_id]

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
            plt.suptitle(self.subject_id)
            ax1.plot(self.timestamps[::int(5 * self.sample_rate / 250)],
                     self.raw[::int(5 * self.sample_rate / 250)], color='black')
            ax1.set_ylabel("ECG Voltage")

            ax2.plot(self.timestamps[::int(10 * self.sample_rate / 250)], self.accel_x, color='dodgerblue')
            ax2.set_ylabel("Accel VM")

            ax3.plot(self.epoch_timestamps[0:min([len(self.epoch_timestamps), len(self.epoch_validity)])],
                     self.epoch_validity[0:min([len(self.epoch_timestamps), len(self.epoch_validity)])], color='black')
            ax3.fill_between(x=self.epoch_timestamps[0:min([len(self.epoch_timestamps), len(final_nw)])],
                             y1="Wear", y2=final_nw, color='grey')

            if manual_log.shape[0] >= 1:
                for row in manual_log.itertuples():
                    ax1.fill_between(x=[row.Start, row.Stop], y1=min(self.raw[::5]), y2=max(self.raw[::5]),
                                     color='red', alpha=.5)
                    ax2.fill_between(x=[row.Start, row.Stop], y1=min(self.accel_x), y2=max(self.accel_x),
                                     color='red', alpha=.5)

            xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")

            ax3.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

            plt.savefig("/Users/kyleweber/Desktop/ECG Non-Wear/{}.png".format(self.subject_id))

        return final_nw

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- Running Code -----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


f = ECG(raw_filepath="/Users/kyleweber/Desktop/008_OmegaSnap_BG.edf",
        processed_filepath=None,
        qc_algorithm="Orphanidou",
        output_dir="/Users/kyleweber/Desktop/",
        ecg_downsample=2,
        age=0, start_offset=0, end_offset=int(250*3600),
        rest_hr_window=60, n_epochs_rest=30,
        epoch_len=15, load_accel=True,
        filter_data=False, low_f=1, high_f=30, f_type="bandpass",
        load_raw=True, from_processed=False)

"""
l = ECG(raw_filepath="/Users/kyleweber/Desktop/BF7_Lead.edf",
        processed_filepath=None,
        output_dir="/Users/kyleweber/Desktop/",
        ecg_downsample=2,
        age=0, start_offset=0, end_offset=0,
        rest_hr_window=60, n_epochs_rest=30,
        epoch_len=15, load_accel=True,
        filter_data=False, low_f=1, high_f=30, f_type="bandpass",
        load_raw=True, from_processed=False)
"""

# Non-wear algorithm
# x.nonwear = x.calculate_nonwear(epoch_len=15, plot_data=True)

# % tracker for QC loop
