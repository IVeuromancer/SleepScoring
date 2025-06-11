import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mne
import joblib
import torch
import os 

def load_and_preprocess_data():
    EEG_files = [
        '../../code/LabCode/Fiber_photometry/mouse_807/mouse_807_230511.edf',
        '../../code/LabCode/Fiber_photometry/mouse_994/mouse_994_230713.edf',
    ]
    epoch_limits = 8640
    all_data, scores_converted = [], []

    for EEG in EEG_files:
        scores_path = EEG.replace('.edf', '_scores.tsv')
        raw = mne.io.read_raw_edf(EEG, preload=True, verbose=False)
        scores = pd.read_csv(scores_path, skiprows=10, delim_whitespace=True)['Time.2']
        for i in range(epoch_limits):
            all_data.append([raw._data[ch][i*1280:(i+1)*1280] for ch in range(4)])
            if scores[i] == 1: scores_converted.append(0)
            elif scores[i] == 2: scores_converted.append(1)
            elif scores[i] == 3: scores_converted.append(2)
            else: scores_converted.append(-2)

    data = np.array(all_data)
    y = np.array(scores_converted).reshape(-1, 1)
    scalers = [StandardScaler().fit(data[:, i, :]) for i in range(4)]
    data_norm = [scaler.transform(data[:, i, :]) for i, scaler in enumerate(scalers)]
    for i, scaler in enumerate(scalers):
        scaler_dir = os.path.join("..", "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(scaler_dir, f'scaler_CH{i+1}_auto.pkl'))
    X = np.stack(data_norm, axis=1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long).squeeze(), EEG_files
