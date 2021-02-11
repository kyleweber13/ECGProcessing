import matplotlib.pyplot as plt
import ImportEDF
import pandas as pd
import ECG_Quality_Check
from datetime import timedelta
import matplotlib.dates as mdates
import numpy as np
import os

data = ImportEDF.Bittium(filepath="/Users/kyleweber/Desktop/Kin 472 - Megan/Test.EDF",
                         start_offset=0, end_offset=0, epoch_len=15, load_accel=False)


def run_orphanidou(win_len=15, age=None, log_file=None, crop_seconds=30, show_plot=True, save_data=False):
    """Function that uses log data to calculate activity-specific epoch data.

    :argument
    -win_len: number of seconds data are epoched over within an activity
    -age: participant age in years used to calculate max HR
    -log_file: excel file containing "Event", "Start", and "Stop" columns for each desired activity.
               Timestamps in YYYY-MM-DD HH:MM:SS format.
    -crop_seconds: number of seconds to remove from start/end of each Event.
                   If Event is "Recovery", cropping will not be applied.
    -show_plot: whether or not to save plot (True/False)
    -save_data: whether or not to save epoched data to working directory (True/False)

    :returns
    -df: dataframe containing epoched data during each desired activity
    """

    if "xlsx" in log_file:
        log = pd.read_excel(log_file)
    if "csv" in log_file:
        log = pd.read_csv(log_file)

    log["Start"] = pd.to_datetime(log["Start"])
    log["Stop"] = pd.to_datetime(log["Stop"])

    # Gets start/stop indexes in raw data
    stamps = []
    validity_list = []
    hr_list = []
    event_list = []

    for row in log.itertuples():

        if row.Event != "Recovery":
            start = int((row.Start - data.timestamps[0]).total_seconds() * data.sample_rate) + \
                    int(crop_seconds * data.sample_rate)
            stop = int((row.Stop - data.timestamps[0]).total_seconds() * data.sample_rate) - \
                    int(crop_seconds * data.sample_rate)

        if row.Event.capitalize() == "Recovery":
            start = int((row.Start - data.timestamps[0]).total_seconds() * data.sample_rate)
            stop = int((row.Stop - data.timestamps[0]).total_seconds() * data.sample_rate)

        for i in range(start, stop, int(data.sample_rate * win_len)):
            if stop - start >= win_len * data.sample_rate:
                d = ECG_Quality_Check.CheckQuality(raw_data=data.raw, sample_rate=data.sample_rate, start_index=i,
                                                   template_data='raw', voltage_thresh=250, epoch_len=win_len)

                stamps.append(data.timestamps[i])
                event_list.append(row.Event)
                validity_list.append("Valid" if d.valid_period else "Invalid")
                hr_list.append(d.hr if d.valid_period else None)

    # Creates dataframe
    df = pd.DataFrame(list(zip(stamps, event_list, validity_list, hr_list)),
                      columns=["Timestamp", "Event", "Validity", "HR"])

    # Calculates resting HR
    resting_hr = None
    max_hr = None
    if "RestingHR" in set([i for i in log["Event"]]):
        hr_data = [i for i in df["HR"].dropna()]
        resting_hr = np.mean(hr_data)
        print("Resting HR is {} bpm".format(round(resting_hr, 1)))
    if age is not None:
        max_hr = 208 - .7*age

    if resting_hr is not None and max_hr is not None:
        hrr_list = [(i - resting_hr) / (max_hr - resting_hr) * 100 if i is not None else None for i in hr_list]

        for i in range(len(hrr_list)):
            if hrr_list[i] is not None:
                if hrr_list[i] < 0:
                    hrr_list[i] = 0

    if resting_hr is None or max_hr is None:
        hrr_list = [None for i in range(df.shape[0])]

    df["HRR"] = hrr_list

    if show_plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 6))
        plt.subplots_adjust(top=.95, bottom=.13)

        ax1.plot(data.timestamps, data.raw, color='red')
        ax1.set_title("Raw ECG")

        ax2.plot(stamps, event_list, linestyle="", marker="s", markeredgecolor='black', markerfacecolor='dodgerblue')
        ax2.set_ylabel("Event")

        ax3.bar(stamps, validity_list, edgecolor='black', color='orange', width=win_len/86400)
        ax3.set_ylabel("Validity")

        ax4.plot(stamps, hr_list,  linestyle="", marker="o", markeredgecolor='black', markerfacecolor='green')

        if max_hr is not None:
            ax4.axhline(y=max_hr, color='green', linestyle='dashed', label='Max HR')
        if resting_hr is not None:
            ax4.axhline(y=resting_hr, color='red', linestyle='dashed', label="Rest HR")

        ax4.legend()
        ax4.set_ylabel("HR")

        ax4.set_ylim(40, 200)

        y_min = ax4.get_ylim()[0]
        y_max = ax4.get_ylim()[1]

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        for row in log.itertuples():
            ax4.fill_between(x=[row.Start, row.Start + timedelta(seconds=crop_seconds)],
                             y1=y_min, y2=y_max, color='grey', alpha=.3)
            ax4.fill_between(x=[row.Stop, row.Stop + timedelta(seconds=-crop_seconds)], y1=y_min, y2=y_max,
                             color='grey', alpha=.3)
            ax4.fill_between(x=[row.Start + timedelta(seconds=crop_seconds),
                                row.Stop + timedelta(seconds=-crop_seconds)], y1=y_min, y2=y_max,
                             color='green', alpha=.3)

    if save_data:
        df.to_excel("HR_Data.xlsx", index=False)
        print("\n-HR data saved to {}".format(os.getcwd()))

    return df


hr_df = run_orphanidou(win_len=15, age=26, log_file="/Users/kyleweber/Desktop/Kin 472 - Megan/FakeLog.xlsx",
                       crop_seconds=15, show_plot=True, save_data=True)


# TODO
# Convert GENEActiv data
# Import GENEActiv data
# Epoched GENEActiv data (1-second epochs)
# View GENEActiv data to generate log file, click for timestamp
# Import ECG file
# Run QC algorithm on ECG data
# Same epoching on GENEActiv data
# Calculate accelerometer intensity
