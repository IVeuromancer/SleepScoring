import numpy as np  
import pandas as pd
import mne
from mne.utils import _stamp_to_dt
from datetime import datetime, timedelta, timezone

def datetime_to_julian(dt):
    """
    Convert a timezone-aware datetime object to Julian date.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix_epoch_julian = 2440587.5
    julian_date = unix_epoch_julian + (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
    return julian_date

mouse_str_all = ['mouse_96', 'mouse_96'] 
date_str_all = ['250508', '250510']
root = 'D:/Makinson_lab/code/LabCode/Fiber_photometry'

for mouse_str, date_str in zip(mouse_str_all, date_str_all):
    raw = mne.io.read_raw_edf(f"{root}/{mouse_str}/{mouse_str}_{date_str}.edf")
    scores = np.load(f"{mouse_str}_{date_str}.npy")

    for i in range(1, len(scores)):
        if scores[i] == 2 and scores[i-1] == 0:
            scores[i] = 0
        elif scores[i] == 1 and scores[i-1] == 2:
            scores[i] = 0

    scores_converted = []
    for x in scores:
        if x == 0:
            scores_converted.append(1)
        elif x == 1:
            scores_converted.append(2)
        elif x == 2:
            scores_converted.append(3)
        else:
            scores_converted.append(255)

    meas_date = raw.info['meas_date']
    if meas_date is not None:
        if isinstance(meas_date, tuple):
            measurement_datetime = datetime.fromtimestamp(meas_date[0], tz=timezone.utc)
        else:
            measurement_datetime = meas_date
        formatted_datetime = measurement_datetime.strftime('%m/%d/%y %H:%M:%S')
        start_datetime = datetime.strptime(formatted_datetime, '%m/%d/%y %H:%M:%S')
        julian_date = datetime_to_julian(measurement_datetime)
        print("Formatted Measurement Datetime:", formatted_datetime)
    else:
        print("No measurement date available.")
        continue

    datetime_list = [(start_datetime + timedelta(seconds=10 * i)).strftime('%m/%d/%y %H:%M:%S') for i in range(8640)]
    dates_only_list = [dt.split(' ')[0] for dt in datetime_list]  
    times_only_list = [dt.split(' ')[1] for dt in datetime_list] 

    columns = ['Date', 'Time', 'Time Stamp', 'Time from Start', 'yijan_0_Numeric']
    df = pd.DataFrame(columns=columns)
    df['Date'] = dates_only_list
    df['Time'] = times_only_list
    df['Time Stamp'] = [0] * 8640
    df['Time from Start'] = [float(i * 10) for i in range(8640)]
    df['yijan_0_Numeric'] = scores_converted

    last_row = df.iloc[-1]  
    datetime_string = f"{last_row['Date']} {last_row['Time']}"
    julian_string = str(julian_date)

    text_lines = [
        "Channels:	1",
        "Count:	8641",
        "Start:	" + julian_string + "	" + formatted_datetime,
        "End:	" + julian_string + "	" + datetime_string,
        "Parameters:	4",
        "NonRem	2",
        "REM	3",
        "Unscored	255",
        "Wake	1",
        ""  
    ]

    file_path = f"{mouse_str}_{date_str}_scores.tsv"
    with open(file_path, 'w') as file:
        file.write('\n'.join(text_lines) + '\n')

    df.to_csv(file_path, sep='\t', index=False, mode='a')
