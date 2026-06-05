# SleepScoring — Automated Sleep Stage Classification

CNN pipeline for automatic sleep stage scoring (Wake / NREM / REM) from mouse EEG/EMG recordings. Supports both offline batch inference from `.edf` files and real-time closed-loop scoring with a Tucker Davis Technologies (TDT) acquisition system for optogenetic stimulation triggered by sleep state.

## Hardware

- **Acquisition**: Tucker Davis Technologies (TDT) system with PO8e streaming
- **Input**: 4-channel EEG/EMG, 1280 samples per epoch
- **Real-time interface**: Serial + TDT Python API (`tdt`)

## Environment

```bash
conda env create -f environment.yml
conda activate SleepScoring
```

## Project structure

```
train/
├── model.py         — 1D CNN classifier (4-channel input → Wake/NREM/REM)
├── dataset.py       — dataset loading and splitting
├── preprocessing.py — signal preprocessing and normalization
├── train.py         — training script
└── utils.py         — shared utilities
inference/
├── inference.py             — batch inference from .edf files via MNE
├── convert_scores_to_tsv.py — convert .npy score arrays to .tsv
├── model.py                 — model definition
└── utils.py                 — scaler loading
realtime_scoring/
└── neuralnet_2mice.py — live scoring pipeline with TDT streaming,
                         tkinter GUI, and optogenetic trigger output
saved_models/      — pretrained .pth checkpoint
scalers/           — per-channel StandardScaler .pkl files
scores/            — output .npy arrays and .tsv files
```

## Model

A lightweight 1D CNN trained on 4-channel EEG/EMG epochs:

- Two 1D conv layers (32 → 64 filters, kernel size 3)
- Dropout (p=0.5) after each conv layer
- Two fully connected layers → 3-class output (Wake / NREM / REM)
- Input shape: `(B, 4, 1280)`

## How to run

```bash
# Train a new model
python train/train.py

# Batch inference on .edf recordings
python inference/inference.py

# Convert score arrays to .tsv
python inference/convert_scores_to_tsv.py

# Real-time scoring with TDT (requires hardware)
python realtime_scoring/neuralnet_2mice.py
```

## Stack

`Python` `PyTorch` `MNE` `TDT` `scikit-learn` `SciPy` `NumPy` `tkinter`
