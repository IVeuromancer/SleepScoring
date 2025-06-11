# Real-Time Sleep Scoring with CNNs (Mouse EEG/EMG)

A convolutional neural network for automatic sleep stage classification using mouse EEG and EMG data. Designed for both offline analysis and **real-time closed-loop experiments** using a Tucker Davis Technologies (TDT) acquisition system. Enables **optogenetic stimulation triggered by sleep state** in live animals.

## 🧠 Overview

This repo includes:

- A trained CNN for classifying Wake/NREM/REM from raw EEG/EMG
- Preprocessing and training scripts
- A real-time Python interface to TDT for closed-loop optogenetics
- Support for fiber photometry-based EEG pipelines

Use it to automate sleep scoring and trigger brain lasers when your mouse hits REM. Or NREM. Or whatever your current hypothesis is.


