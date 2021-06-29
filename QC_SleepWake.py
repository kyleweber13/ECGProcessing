import pandas as pd
import matplotlib.pyplot as plt
from ECG_Quality_Check import CheckQuality
import ImportEDF
import numpy as np
import os
import pingouin as pg
import pyedflib
from datetime import timedelta


def run_all_subjs():

    # df_sleep = pd.read_excel("/Users/kyleweber/Desktop/Test.xlsx")
    df_sleep = pd.read_excel("/Users/kyleweber/Desktop/Data/OND07/Tabular Data/OND07_SleepLogs_Reformatted.xlsx")
    epoch_len = 15

    file_dir = "/Volumes/Kyle's External HD/OND07 Bittium/"
    files = [i for i in os.listdir(file_dir) if "edf" in i or "EDF" in i]

    subjs = df_sleep['ID'].unique()

    for subj in subjs[-4:]:

        if os.path.exists(file_dir + f"{subj}_01_BF.EDF"):
            data = ImportEDF.Bittium(filepath=file_dir + f"{subj}_01_BF.EDF",
                                     start_offset=0, end_offset=0, epoch_len=epoch_len, load_accel=False,
                                     low_f=1, high_f=30, f_type="bandpass")

            orphanidou = []

            for i in range(0, len(data.raw[::2]), int(data.sample_rate / 2 * epoch_len)):
                if i % 1000 == 0:
                    print(f"{round(100*i/(len(data.raw)/2), 1)}%")

                d = CheckQuality(raw_data=data.raw[::2], sample_rate=data.sample_rate/2, start_index=i,
                                 template_data='raw', voltage_thresh=250, epoch_len=epoch_len)
                orphanidou.append("Valid" if d.valid_period else "Invalid")

            epoch_stamps = data.timestamps[::int(data.sample_rate * epoch_len)]

            sleep_mask = np.zeros(len(epoch_stamps))

            for row in df_sleep.loc[df_sleep["ID"] == subj].itertuples():
                start_ind = int((row.sleep - epoch_stamps[0]).total_seconds() / epoch_len)
                end_ind = int((row.wake - epoch_stamps[0]).total_seconds() / epoch_len)

                sleep_mask[start_ind:end_ind] = 1

            df_qc = pd.DataFrame(np.array([epoch_stamps, sleep_mask, orphanidou]).transpose(),
                                 columns=["Timestamp", "SleepMask", "QC"])
            df_qc["Timestamp"] = pd.to_datetime(df_qc["Timestamp"])

            df_qc.to_excel(f"/Users/kyleweber/Desktop/ECG Output/{subj}_ECG_Output.xlsx", index=False)

            plt.close("all")
            fig, axes = plt.subplots(3, sharex='col', figsize=(10, 6))

            axes[0].plot(data.timestamps[::5], data.filtered[::5])
            axes[1].plot(df_qc["Timestamp"], df_qc["SleepMask"])
            axes[2].plot(df_qc["Timestamp"], df_qc["QC"])
            plt.savefig(f"/Users/kyleweber/Desktop/ECG Output/Plots/{subj}.png")

            del data, df_qc


def run_all_subjs_acc():

    df_sleep = pd.read_excel("/Users/kyleweber/Desktop/Data/OND07/Tabular Data/OND07_SleepLogs_Reformatted.xlsx")

    epoch_len = 15

    file_dir = "/Volumes/Kyle's External HD/OND07 Bittium/"
    files = [i for i in os.listdir(file_dir) if "edf" in i or "EDF" in i]

    subjs = df_sleep['ID'].unique()

    for subj in subjs:
        print(subj)

        if os.path.exists(file_dir + f"{subj}_01_BF.EDF"):
            file = pyedflib.EdfReader(file_dir + f"{subj}_01_BF.EDF")

            """CHECK FOR CHANNEL HEADERS TO MAKE SURE IS ACCEL"""
            sample_rate = file.getSampleFrequencies()[1]

            x = file.readSignal(chn=1)
            y = file.readSignal(chn=2)
            z = file.readSignal(chn=3)

            starttime = file.getStartdatetime()

            file.close()

            end_time = starttime + timedelta(seconds=len(x) / sample_rate)
            timestamps = np.asarray(pd.date_range(start=starttime, end=end_time, periods=len(x)))
            epoch_stamps = timestamps[::epoch_len * sample_rate]

            vm = (np.sqrt(np.square(np.array([x, y, z])).sum(axis=0)) - 1000) / 1000
            vm[vm < 0] = 0

            avm = [np.mean(vm[i:i+int(epoch_len*sample_rate)]) for i in np.arange(0, len(x), int(15*sample_rate))]

            df = pd.DataFrame(list(zip(epoch_stamps, avm)), columns=["Timestamp", "AVM"])

            df.to_excel(f"/Users/kyleweber/Desktop/ECG Output/{subj}_Acc_Output.xlsx", index=False)

            del df, x, y, z, avm, vm, timestamps


# run_all_subjs_acc()


def combine_files():
    os.chdir("/Users/kyleweber/Desktop/ECG Output/")

    files = sorted([i for i in os.listdir(os.getcwd()) if "xlsx" in i])

    df = pd.DataFrame([[], [], [], [], []]).transpose()
    df.columns = ["ID", "Timestamp", "SleepMask", "QC", "AVM"]

    for file in files:
        subj = file.split("_")[2]
        print(subj)
        d = pd.read_excel(f"{os.getcwd()}/{file}")
        d["ID"] = [subj for i in range(d.shape[0])]

        df = df.append(d)

    return df


def calculate_stats():

    n_valids, n_invalids, valid_wakes, invalid_wakes, valid_sleeps, invalid_sleeps = [], [], [], [], [], []

    for subj in df['ID'].unique():
        print(subj)
        d = df.loc[df["ID"] == subj]

        n = d.shape[0]
        n_valid = d.loc[d["QC"] == "Valid"].shape[0] / n
        n_invalid = d.loc[d["QC"] == "Invalid"].shape[0] / n

        wake = d.loc[d["SleepMask"] == 0]
        sleep = d.loc[d["SleepMask"] == 1]

        valid_wake = wake.loc[wake["QC"] == "Valid"].shape[0] / wake.shape[0]
        invalid_wake = wake.loc[wake["QC"] == "Invalid"].shape[0] / wake.shape[0]
        valid_sleep = sleep.loc[sleep["QC"] == "Valid"].shape[0] / sleep.shape[0]
        invalid_sleep = sleep.loc[sleep["QC"] == "Invalid"].shape[0] / sleep.shape[0]

        n_valids.append(n_valid)
        n_invalids.append(n_invalid)
        valid_wakes.append(valid_wake)
        invalid_wakes.append(invalid_wake)
        valid_sleeps.append(valid_sleep)
        invalid_sleeps.append(invalid_sleep)

    df_stats = pd.DataFrame(list(zip(df["ID"].unique(), n_valids, n_invalids,
                                     valid_wakes, invalid_wakes, valid_sleeps, invalid_sleeps)))
    df_stats.columns = ["ID", "n_valid", "n_invalid", "valid_wake", "invalid_wake", "valid_sleep", "invalid_sleep"]

    return df_stats


# df_stats = calculate_stats()


def gen_histograms(plot_type="histogram"):

    df_pg = pg.ttest(df_stats["valid_wake"], df_stats['valid_sleep'], paired=True)
    df_pg["Variable"] = ["Valid-Invalid"]

    if plot_type == "histogram":
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        plt.subplots_adjust(left=.05, top=.95, hspace=.25)

        bins = np.arange(0, 1.05, .1)
        axes[0][0].hist(df_stats["n_valid"], color='green', alpha=.5, edgecolor='black', bins=bins)
        axes[0][0].set_title("% valid (all)")

        axes[1][0].hist(df_stats["n_invalid"], color='red', alpha=.5, edgecolor='black', bins=bins)
        axes[1][0].set_title("% invalid (all)")

        axes[0][1].hist(df_stats["valid_wake"], color='green', alpha=.5, edgecolor='black', bins=bins)
        axes[0][1].set_title("% valid wake")

        axes[0][2].hist(df_stats["invalid_wake"], color='red', alpha=.5, edgecolor='black', bins=bins)
        axes[0][2].set_title("% invalid wake")

        axes[1][1].hist(df_stats["valid_sleep"], color='green', alpha=.5, edgecolor='black', bins=bins)
        axes[1][1].set_title("% valid sleep")

        axes[1][2].hist(df_stats["invalid_sleep"], color='red', alpha=.5, edgecolor='black', bins=bins)
        axes[1][2].set_title("% invalid sleep")

    if plot_type == 'barplot':

        df_desc = df_stats.describe()

        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.bar(x=df_desc.columns, height=df_desc.loc['mean'], yerr=df_desc.loc["std"], capsize=4,
                color=['green', 'red'], edgecolor='black', alpha=.5)
        ax.set_title("Mean ± SD")

    if plot_type == 'boxplot':

        fig, ax = plt.subplots(1, figsize=(10, 6))
        df_stats.boxplot(grid=False, ax=ax)

    if plot_type == "scatter":
        plt.scatter(df_stats["valid_wake"], df_stats["valid_sleep"], edgecolors='black', color='red')
        plt.ylabel("valid_sleep")
        plt.xlabel("valid_wake")
        plt.plot(np.arange(0, 1.1, .1), np.arange(0, 1.1, .1), color='black', linestyle='dashed')

    return df_pg


# stats = gen_histograms("histogram")
