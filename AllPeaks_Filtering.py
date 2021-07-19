import pandas as pd
from ECG_Quality_Check import find_all_peaks
import ImportEDF
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
from datetime import timedelta
import matplotlib.dates as mdates

# file = "/Users/kyleweber/Desktop/Data/ECG Files/OND07_WTL_3023_01_BF.EDF"
ff_file = "/Users/kyleweber/Desktop/BG7_FastFix.edf"
lead_file = "/Users/kyleweber/Desktop/BG7_Lead.edf"


class ContinuousECGAnalytics:

    def __init__(self, filename, epoch_len=10):

        self.filename = filename
        self.raw = None
        self.fs = None
        self.start_time = None
        self.stop_time = None
        self.df = None
        self.epoch_len = epoch_len
        self.jump_int = 1

        """METHODS"""
        self.get_header()
        self.start_inds, self.stop_inds = self.find_loop_indexes()

    def get_header(self):

        f = pyedflib.EdfReader(self.filename)
        self.fs = f.getSampleFrequency(0)
        self.start_time = f.getStartdatetime()
        self.stop_time = self.start_time + timedelta(seconds=f.getFileDuration())

    def find_loop_indexes(self):

        true_start = self.start_time
        true_end = self.stop_time
        start_time = pd.to_datetime(str(true_start.date()) + " 14:00:00")
        stop_time = pd.to_datetime(str(true_end.date()) + " 14:00:00")

        days = pd.date_range(start=start_time, end=stop_time, freq='1D')

        start_inds = []
        stop_inds = []

        if start_time >= true_start:
            start_inds.append(int((start_time - true_start).total_seconds() * self.fs))
        if start_time < true_start:
            start_inds.append(0)

        stop_inds.append(86400 * self.fs + start_inds[0])

        # Gets start indexes
        for day in days[1:-1]:
            if day < true_end:
                start_inds.append(int((day - true_start).total_seconds() * self.fs))
                stop_inds.append(86400 * self.fs)

        return start_inds, stop_inds

    def import_daily_data(self, epoch_len, avg_n_beats=10, day_num=None):

        print("\nProcessing continuous ECG data...")

        # Empty DF
        df_all = pd.DataFrame(columns=["Timestamp", "HR", "AvgHR"])

        if day_num is None:
            tally = 0
            for start_ind, stop_ind in zip(self.start_inds, self.stop_inds):
                print("======================== Day {} ======================== ".format(tally + 1))

                data = ImportEDF.Bittium(filepath=self.filename,
                                         start_offset=start_ind, end_offset=stop_ind, epoch_len=epoch_len,
                                         load_accel=False,
                                         low_f=.67, high_f=30, f_type="bandpass")

                all_peaks, swt, filt = find_all_peaks(raw_data=data.raw, fs=data.sample_rate,
                                                      use_epochs=False, plot_data=False, epoch_len=epoch_len)

                if tally >= 0:
                    all_peaks2 = [i + start_ind for i in all_peaks]
                if tally == 0:
                    all_peaks2 = all_peaks

                inst_hr = []

                for b1, b2 in zip(all_peaks[:], all_peaks[1:]):
                    inst_hr.append(60 / ((b2 - b1) / data.sample_rate))

                keep_index = []
                for row in range(0, len(inst_hr) - 1):
                    if abs(inst_hr[row + 1] - inst_hr[row]) < 5:
                        keep_index.append(row)

                peak_ind = [all_peaks[i] for i in keep_index]
                df = pd.DataFrame(list(zip([data.timestamps[i] for i in peak_ind],
                                           [all_peaks2[i] for i in keep_index],
                                           [inst_hr[i] for i in keep_index])),
                                  columns=["Timestamp", "Peak", "HR"])
                avg_hr = [sum(df["HR"].iloc[i:i + avg_n_beats]) / avg_n_beats for i in range(df.shape[0])]
                for i in range(len(avg_hr) - df.shape[0]):
                    avg_hr.append(None)
                df["AvgHR"] = avg_hr

                df_all = df_all.append(df)

                tally += 1

        if type(day_num) is int:
            print("======================== Day {} ======================== ".format(day_num))

            data = ImportEDF.Bittium(filepath=self.filename,
                                     start_offset=self.start_inds[day_num-1],
                                     end_offset=self.stop_inds[day_num-1], epoch_len=epoch_len,
                                     load_accel=False,
                                     low_f=.67, high_f=30, f_type="bandpass")

            all_peaks, swt, filt = find_all_peaks(raw_data=data.raw, fs=data.sample_rate,
                                                  use_epochs=False, plot_data=False, epoch_len=epoch_len)

            if day_num >= 0:
                all_peaks2 = [i + self.start_inds[day_num - 1] for i in all_peaks]
            if day_num == 0:
                all_peaks2 = all_peaks

            inst_hr = []

            for b1, b2 in zip(all_peaks[:], all_peaks[1:]):
                inst_hr.append(60 / ((b2 - b1) / data.sample_rate))

            keep_index = []
            for row in range(0, len(inst_hr) - 1):
                if abs(inst_hr[row + 1] - inst_hr[row]) < 5:
                    keep_index.append(row)

            peak_ind = [all_peaks[i] for i in keep_index]
            df = pd.DataFrame(list(zip([data.timestamps[i] for i in peak_ind],
                                       [all_peaks2[i] for i in keep_index],
                                       [inst_hr[i] for i in keep_index])),
                              columns=["Timestamp", "Peak", "HR"])
            avg_hr = [sum(df["HR"].iloc[i:i + avg_n_beats]) / avg_n_beats for i in range(df.shape[0])]
            for i in range(len(avg_hr) - df.shape[0]):
                avg_hr.append(None)
            df["AvgHR"] = avg_hr

            df_all = df_all.append(df)

        return df_all

    def calc_epoch_avg(self, epoch_len=60, jump_int=30):

        print("\nCalculating HR averaged over {} seconds in {}-second intervals...".format(epoch_len, jump_int))
        self.epoch_len, self.jump_int = epoch_len, jump_int

        stamps = pd.date_range(start=self.df['Timestamp'].iloc[0], end=self.df['Timestamp'].iloc[-1],
                               freq="{}S".format(jump_int))

        avg = []
        sd = []
        for start, stop in zip(stamps[:], stamps[1:]):
            d = self.df.loc[(self.df["Timestamp"] >= start) &
                            (self.df["Timestamp"] < start + timedelta(seconds=epoch_len))]
            avg.append(d["HR"].mean())
            sd.append(d["HR"].std())

        avg_fix = [i if i > 40 else None for i in avg]

        fig, axes = plt.subplots(2, sharex='col', figsize=(10, 6))
        plt.suptitle(self.filename.split("/")[-1])

        axes[0].plot(self.df["Timestamp"], self.df["HR"], color='red', label="Beat-by-beat")
        axes[0].plot(stamps[:-1], avg_fix, color='black', label="{}s avg (rem. < 40bpm)".format(epoch_len))
        axes[0].legend()
        axes[0].set_title("HR averaged in {}-sec epochs every {} seconds".format(epoch_len, jump_int))
        axes[0].set_ylabel("HR")
        axes[0].set_yticks(np.arange(0, 201, 25))
        axes[0].fill_between(x=[stamps[0], stamps[-1]], y1=40, y2=200, color='green', alpha=.2)

        axes[1].plot(stamps[:-1], sd, color='black', label="{}s sd".format(epoch_len))
        axes[1].set_ylabel("SD HR")
        axes[1].axhline(y=20, linestyle='dashed', color='red')
        axes[1].legend()

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        axes[1].xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def write_data(self, output_dir=""):

        save_loc = output_dir + self.filename.split("/")[-1].split(".")[0] + \
                   "_Epoch{}Jump{}.xlsx".format(self.epoch_len, self.jump_int)
        print("\nSaving data as {}".format(save_loc))
        self.df.to_excel(save_loc, index=False)


ff = ContinuousECGAnalytics(filename=ff_file)
ff.df = ff.import_daily_data(epoch_len=15, avg_n_beats=10, day_num=None)
ff.calc_epoch_avg(epoch_len=60, jump_int=15)
# ff.write_data(output_dir="/Users/kyleweber/Desktop/")

l = ContinuousECGAnalytics(filename=lead_file)
l.df = l.import_daily_data(epoch_len=15, avg_n_beats=10, day_num=None)
l.calc_epoch_avg(epoch_len=60, jump_int=15)
# l.write_data(output_dir="/Users/kyleweber/Desktop/")
