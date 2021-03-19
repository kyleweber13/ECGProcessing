import pandas as pd
from ECG_Quality_Check import find_all_peaks
import ImportEDF
import matplotlib.pyplot as plt

file = "/Users/kyleweber/Desktop/Data/ECG Files/OND07_WTL_3023_01_BF.EDF"

for day in range(6):
    data = ImportEDF.Bittium(filepath=file,
                             start_offset=int(day*86400*250), end_offset=int(86400*250), epoch_len=10, load_accel=False,
                             low_f=.67, high_f=30, f_type="bandpass")
    all_peaks, swt, filt = find_all_peaks(raw_data=data.raw, fs=data.sample_rate,
                                          use_epochs=False, plot_data=False, epoch_len=60)

    inst_hr = []

    for b1, b2 in zip(all_peaks[:], all_peaks[1:]):
        inst_hr.append(60/((b2-b1)/data.sample_rate))

    keep_index = []
    for row in range(0, len(inst_hr)-1):
        if abs(inst_hr[row+1] - inst_hr[row]) < 5:
            keep_index.append(row)

    peak_ind = [all_peaks[i] for i in keep_index]
    df = pd.DataFrame(list(zip([data.timestamps[i] for i in peak_ind],
                               [all_peaks[i] for i in keep_index],
                               [inst_hr[i] for i in keep_index])),
                      columns=["Timestamp", "Peak", "HR"])
    avg_hr = [sum(df["HR"].iloc[i:i+10])/10 for i in range(df.shape[0])]
    for i in range(len(avg_hr) - df.shape[0]):
        avg_hr.append(None)
    df["AvgHR"] = avg_hr

    df.to_excel(data.filepath.split("/")[-1].split(".")[0] + "Day{}.xlsx".format(day))


df = pd.DataFrame(columns=["Timestamp", "HR", "AvgHR"])
for i in range(1, 7):
    d = pd.read_excel("/Users/kyleweber/Desktop/{}_Day{}.xlsx".format(file.split("/")[-1].split(".")[0], i))
    df = df.append(d)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

plt.plot(df["Timestamp"], df["AvgHR"], color='red')
