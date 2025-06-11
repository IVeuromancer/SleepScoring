import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mne
import joblib
import time
from model import SleepScoringModel
from utils import load_scalers

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model_path = '../saved_models/sleep_scoring_model_241103a.pth'
model = SleepScoringModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from {model_path}")
print("Model is using", "GPU." if next(model.parameters()).is_cuda else "CPU.")

# Load scalers
scaler_CH1, scaler_CH2, scaler_CH3, scaler_CH4 = load_scalers([
    '../scalers/scaler_CH1_241103.pkl',
    '../scalers/scaler_CH2_241103.pkl',
    '../scalers/scaler_CH3_241103.pkl',
    '../scalers/scaler_CH4_241103.pkl'  
])

# Inference loop
mouse_str_all = ['mouse_96', 'mouse_96'] 
date_str_all = ['250508', '250510']
root = 'D:/Makinson_lab/code/LabCode/Fiber_photometry'
batch_size = 16

for mouse_str, date_str in zip(mouse_str_all, date_str_all):
    scores = []
    raw = mne.io.read_raw_edf(f"{root}/{mouse_str}/{mouse_str}_{date_str}.edf")

    total_batches = (8640 + batch_size - 1) // batch_size
    for batch_idx in range(0, 8640, batch_size):
        start_time = time.time()

        CH1_batch = [raw[0][0][0][1280 * j:1280 * (j + 1)] for j in range(batch_idx, min(batch_idx + batch_size, 8640))]
        CH2_batch = [raw[1][0][0][1280 * j:1280 * (j + 1)] for j in range(batch_idx, min(batch_idx + batch_size, 8640))]
        CH3_batch = [raw[2][0][0][1280 * j:1280 * (j + 1)] for j in range(batch_idx, min(batch_idx + batch_size, 8640))]
        CH4_batch = [raw[3][0][0][1280 * j:1280 * (j + 1)] for j in range(batch_idx, min(batch_idx + batch_size, 8640))]

        X_data_batch = np.stack([
            [scaler_CH1.transform(CH1.reshape(1, 1280))[0],
             scaler_CH2.transform(CH2.reshape(1, 1280))[0],
             scaler_CH3.transform(CH3.reshape(1, 1280))[0],
             scaler_CH4.transform(CH4.reshape(1, 1280))[0]]
            for CH1, CH2, CH3, CH4 in zip(CH1_batch, CH2_batch, CH3_batch, CH4_batch)
        ])

        tensor_batch = torch.tensor(X_data_batch, dtype=torch.float32).to(device)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = model(tensor_batch)
                _, predicted_labels = torch.max(outputs, 1)
                scores.extend(predicted_labels.cpu().tolist())

        batch_time = time.time() - start_time
        current_batch = batch_idx // batch_size + 1
        percentage_complete = (current_batch / total_batches) * 100
        print(f"Processed batch {current_batch}/{total_batches} "
              f"({percentage_complete:.2f}% complete) - Time per batch: {batch_time:.2f} seconds")

    np.save(f"{mouse_str}_{date_str}.npy", scores)
