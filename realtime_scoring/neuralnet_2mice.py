import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import mne
import tdt
from cffi import FFI
from scipy import signal

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PO8():
    def __init__(self, PORTS_PER_CARD=1):

        MAX_CARDS = 4

        self.data = [[] for i in range(MAX_CARDS)]
        self.offsets = [[] for i in range(MAX_CARDS)]
        self.status = [[] for i in range(MAX_CARDS)]
        self.nchannels = [[] for i in range(MAX_CARDS)]
        self.nblocks = [[] for i in range(MAX_CARDS)]
        self.sample_size = [[] for i in range(MAX_CARDS)]
        self.dtype = [np.float32 for i in range(MAX_CARDS)]
        self.sort_codes = [0 for i in range(MAX_CARDS)]
        self.bits_per_bin = [0 for i in range(MAX_CARDS)]

        self.PORTS_PER_CARD = PORTS_PER_CARD
        self.ffi = FFI()

        self.libName = 'PO8eStreaming'
        if platform.system() != 'Windows':
            self.libName = '{0}/lib{1}.so'.format(os.path.abspath(os.curdir), self.libName)
        print('using', self.libName)
        self.dll = self.ffi.dlopen(self.libName)
        print(self.dll)

        with open('PO8e_Python.h', 'r') as f:
            sss = f.read()
        self.ffi.cdef(sss)

        self.cards = [[] for i in range(MAX_CARDS)]
        self.buffers = [[] for i in range(MAX_CARDS)]
        self.CardPointer = [self.ffi.new_handle(c) for c in self.cards]
        self.StoppedPointer = [self.ffi.new("bool *") for i in range(MAX_CARDS)]
        self.BufferPointer = [self.ffi.new_handle(b) for b in self.buffers]

    def isNull(self, index):
        return self.CardPointer[index] == 0

    def isStopped(self, index):
        return self.StoppedPointer[index][0] == 1

    def cardCount(self):
        return self.dll.cardCount()

    def connectToCard(self, card=0, port=0, dtype=np.float32):
        '''
        if data is 32-bit, use dtype to specify how to interpret this
        this could be np.float32, np.int32, np.uint32
        '''
        index = card * self.PORTS_PER_CARD + port
        self.CardPointer[index] = self.dll.connectToCard(card, port)
        self.dtype[index] = dtype

    def releaseCard(self, index):
        self.dll.releaseCard(self.CardPointer[index])
        self.CardPointer[index] = 0

    def startCollecting(self, index, detectStops):
        return self.dll.startCollecting(self.CardPointer[index], detectStops)

    def getLastError(self, index):
        return self.dll.getLastError(self.CardPointer[index])

    def getStreamInfo(self, index):
        self.nchannels[index] = self.dll.numChannels(self.CardPointer[index])
        self.nblocks[index] = self.dll.numBlocks(self.CardPointer[index])
        self.sample_size[index] = self.dll.dataSampleSize(self.CardPointer[index])

    def waitForDataReady(self, index, timeout=2**31 - 1):
        # timeout is in milliseconds
        self.dll.waitForDataReady(self.CardPointer[index], timeout)

    def samplesReady(self, index):
        numSamples = self.dll.samplesReady(self.CardPointer[index], self.StoppedPointer[index])
        return numSamples

    def readBlock(self, index, nSamples):
        self.getStreamInfo(index)
        sss = self.sample_size[index]

        if sss == 4:
            fmt = self.dtype[index]
        elif sss == 2:
            fmt = np.int16
        elif sss == 1:
            fmt = np.int8
        else:
            raise Exception('sampleSize {0} unrecognized'.format(sss))

        ppp = np.zeros((self.nchannels[index], nSamples), dtype=fmt)
        iii = np.zeros(nSamples, dtype=np.int64)
        pBuffer = self.ffi.cast("float *", ppp.ctypes.data)
        iBuffer = self.ffi.cast("int64_t *", iii.ctypes.data)

        self.status[index] = self.dll.readBlock(self.CardPointer[index], pBuffer, nSamples, iBuffer)
        self.data[index] = ppp
        self.offsets[index] = iii

    def flushBufferedData(self, index, nSamples, release):
        self.dll.flushBufferedData(self.CardPointer[index], nSamples, release)

    def bandpass_filt(self, fs, sig, lowcut, highcut, polynomial):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(polynomial, [low, high], btype='bandpass', analog=False)
        filt_data = signal.filtfilt(b, a, sig)
        return filt_data

    def notch_filt(self, fs, sig, f0, Q):
        b, a = signal.iirnotch(f0, Q, tdt_fs)
        filt_data = signal.filtfilt(b, a, sig)
        return filt_data


###############################~CODE START~###############################

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    syn = tdt.SynapseAPI()
    print(syn.getModeStr())

    class SleepScoringModel(nn.Module):
        def __init__(self, dropout_rate=0.5, weight_decay=1e-4):
            super(SleepScoringModel, self).__init__()

            self.conv1 = nn.Conv1d(4, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

            # Add dropout layers
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)

            self.fc1 = nn.Linear(64 * 1280, 128)
            self.fc2 = nn.Linear(128, 3)

            # Add weight decay to the optimizer (e.g., Adam)
            self.weight_decay = weight_decay

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.dropout1(x)  # Apply dropout
            x = self.conv2(x)
            x = F.relu(x)
            x = self.dropout2(x)  # Apply dropout
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------
    MOUSE_NUM = 'mouse_756_757'
    DATE = '250422'

    MODEL_PATH = 'sleep_scoring_model_241103.pth'
    SCALER_PATHS = {
        'CH1': 'scaler_CH1_241103.pkl',
        'CH2': 'scaler_CH2_241103.pkl',
        'CH3': 'scaler_CH3_241103.pkl',
        'CH4': 'scaler_CH4_241103.pkl',
    }

    TDT_FS = 610       # native TDT sampling rate (Hz)
    FS = 128           # sampling rate after resampling (Hz)
    TIME_CHUNK = 10     # epoch length (s)
    EPOCH_SAMPLES = FS * TIME_CHUNK

    # Row indices of the 4 signal channels within tdt.data[0], per mouse
    CHANNEL_NAMES = ['R_SSC', 'L_SSC', 'R_EMG', 'L_EMG']
    CHANNEL_ROWS = {
        'm1': [2, 3, 4, 5],
        'm2': [10, 11, 12, 13],
    }

    WAKE, NREM, REM = 0, 1, 2
    LABELS = {WAKE: 'wake', NREM: 'NREM', REM: 'REM'}

    # Optogenetic stim windows (seconds since start) and the vigilance
    # state each window triggers stim on. Thresholds/targets unchanged
    # from the original hardcoded logic.
    WINDOW_1_END = 28800
    WINDOW_2_END = 57600
    WINDOW_3_END = 86400

    # -----------------------------------------------------------------
    # Model + scalers
    # -----------------------------------------------------------------
    model = SleepScoringModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")

    scalers = {ch: joblib.load(path) for ch, path in SCALER_PATHS.items()}

    def score_epoch(raw_block, rows):
        """Resample, scale, and classify one mouse's epoch.
        Returns (scaled_channels: list of 4 [1 x EPOCH_SAMPLES] arrays, predicted_label: int)."""
        signals = np.stack([raw_block[r] for r in rows])
        info = mne.create_info(CHANNEL_NAMES, TDT_FS, ch_types=['eeg', 'eeg', 'emg', 'emg'], verbose=False)
        raw = mne.io.RawArray(signals, info, first_samp=0, copy='auto', verbose=False)
        raw.resample(FS)

        scaled = [scalers[f'CH{i + 1}'].transform(raw._data[i].reshape(1, EPOCH_SAMPLES)) for i in range(4)]

        x = torch.tensor(np.vstack(scaled), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
        predicted_label = torch.argmax(outputs, dim=1).item()

        return scaled, predicted_label

    def maybe_trigger_stim(pulse_gen, total_elapsed_time, predicted_label):
        if total_elapsed_time <= WINDOW_1_END and predicted_label == REM:
            syn.setParameterValue(pulse_gen, 'Enable', 1)
        elif WINDOW_1_END < total_elapsed_time <= WINDOW_2_END and predicted_label == WAKE:
            syn.setParameterValue(pulse_gen, 'Enable', 1)
        elif WINDOW_2_END < total_elapsed_time <= WINDOW_3_END and predicted_label == NREM:
            syn.setParameterValue(pulse_gen, 'Enable', 1)

    sleep_scores = []
    corresponding_sigs = []
    elapsed_times = []
    total_elapsed_times = []

    # -----------------------------------------------------------------
    # GUI setup
    # -----------------------------------------------------------------
    root = tk.Tk()
    root.title("real-time score (10s bin)")
    root.resizable(width=False, height=False)
    frame1 = tk.Frame(root, bg='black')
    frame1.grid(row=0, column=0)
    frame2 = tk.Frame(root)
    frame2.grid(row=1, column=0, pady=10)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(5, 8))

    x_axis = np.linspace(0, TIME_CHUNK, EPOCH_SAMPLES)
    y_axis = np.zeros(EPOCH_SAMPLES)
    colors = ['c', 'm', 'y', 'w', 'c', 'm', 'y', 'w']
    lines = [ax[i].plot(x_axis, y_axis, c=colors[i], lw=0.5)[0] for i in range(8)]

    ylabels = ['R EEG (norm)', 'L EEG (norm)', 'R EMG (norm)', 'L EMG (norm)',
               'R EEG (norm)', 'L EEG (norm)', 'R EMG (norm)', 'L EMG (norm)']
    ylims = [(-3.5, 3.5), (-3.5, 3.5), (-0.1, 0.1), (-0.1, 0.1),
             (-3.5, 3.5), (-3.5, 3.5), (-0.1, 0.1), (-0.1, 0.1)]
    for i in range(8):
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xlim(0, TIME_CHUNK)
        ax[i].set_xticks([0, TIME_CHUNK])
        ax[i].set_ylim(*ylims[i])
    ax[7].set_xlabel('time (s)')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame1)
    canvas.get_tk_widget().grid(row=0, column=0)

    # -----------------------------------------------------------------
    # Hardware setup
    # -----------------------------------------------------------------
    tdt = PO8(1)

    cardCount = tdt.cardCount()
    print('Found {0} card(s) in the system.'.format(cardCount))

    if cardCount == 0:
        print('no cards found, exiting')
        sys.exit()

    for card in range(cardCount):
        for port in range(tdt.PORTS_PER_CARD):
            print(' Connecting to card', card, 'port', port)

            if port == 0:
                # first port is expected to get integer 32 data in this demo (sort codes).
                # number of sort codes and bit size should match Binner
                tdt.connectToCard(card, port, dtype=np.float32)
            else:
                # second port receives floats.
                tdt.connectToCard(card, port)

            index = card * tdt.PORTS_PER_CARD + port
            if tdt.isNull(index):
                print('  connection failed')
            else:
                print('  established connection to card', card, 'port', port)
                if not tdt.startCollecting(index, 1):
                    print('  startCollecting() failed with:', tdt.getLastError(index))
                    tdt.releaseCard(index)
                else:
                    print('  port is collecting incoming data.')

    portCount = cardCount * tdt.PORTS_PER_CARD

    print('Total {} ports found'.format(portCount))
    print('Waiting for the stream to start on port 0')

    numSamples = 0
    while numSamples < 1:
        numSamples = tdt.samplesReady(0)
        time.sleep(0.05)

    print('Stream started')

    stoppedCount = 0
    loopCt = 0
    skipCt = 0
    lastPrintTime = time.perf_counter()
    tic = time.time()
    toc = time.time()

    while stoppedCount < portCount:

        port = 0

        numSamples = 0
        while numSamples < TIME_CHUNK * TDT_FS:
            numSamples = tdt.samplesReady(0)

        if numSamples == TIME_CHUNK * TDT_FS:
            tdt.readBlock(port, numSamples)
            if tdt.status[port] == 0:
                break
            else:
                buffer_time = time.time() - toc
                print(f'time from last buffer: {buffer_time}')
                elapsed_times.append(buffer_time)
                toc = time.time()
                total_elapsed_time = toc - tic
                total_elapsed_times.append(total_elapsed_time)
                print(f'total time elapsed: {total_elapsed_time}s')
                tdt.flushBufferedData(port, numSamples, 0)

                m1_scaled, m1_y_hat = score_epoch(tdt.data[0], CHANNEL_ROWS['m1'])
                m2_scaled, m2_y_hat = score_epoch(tdt.data[0], CHANNEL_ROWS['m2'])

                corresponding_sigs.append(m1_scaled + m2_scaled)

                print(f'm1: {LABELS[m1_y_hat]}')
                print(f'm2: {LABELS[m2_y_hat]}')

                maybe_trigger_stim('PulseGen1', total_elapsed_time, m1_y_hat)
                maybe_trigger_stim('PulseGen2', total_elapsed_time, m2_y_hat)

                sleep_scores.append([m1_y_hat, m2_y_hat])

                for line, data in zip(lines, m1_scaled + m2_scaled):
                    line.set_ydata(data)

                prediction_label = tk.Label(
                    master=frame2,
                    text=f"m1 prediction: {LABELS[m1_y_hat]}, m2 prediction: {LABELS[m2_y_hat]}",
                    font=("Arial", 20), borderwidth=3, relief='solid')
                prediction_label.pack()

                canvas.draw()
                root.update()
                canvas.flush_events()
                prediction_label.destroy()

                np.save(f'{MOUSE_NUM}_{DATE}_sleep_scores.npy', sleep_scores)

        else:
            tdt.flushBufferedData(port, numSamples, 0)
            print('SKIPPED')
            buffer_time = time.time() - toc
            print(f'time from last buffer: {buffer_time}')
            elapsed_times.append(buffer_time)
            toc = time.time()
            total_elapsed_time = toc - tic
            total_elapsed_times.append(total_elapsed_time)
            print(f'time elapsed: {total_elapsed_time}s')

            sleep_scores.append([255, 255])
            corresponding_sigs.append([np.zeros((1, EPOCH_SAMPLES))] * 8)
            skipCt += 1
            np.save(f'{MOUSE_NUM}_{DATE}_sleep_scores.npy', sleep_scores)

        loopCt += 1
        if time.perf_counter() - toc > 10:
            stoppedCount += 1
            print('Disconnected')
        if tdt.status[port] == 0:
            print('Disconnected_2')
            stoppedCount += 1

    for port in range(portCount):
        print('Releasing port', port)
        tdt.releaseCard(port)

    print('{:>.3f} s total time'.format(time.perf_counter() - lastPrintTime))
    print('total num loops: {}'.format(loopCt))
    print('total num skips: {}'.format(skipCt))

    root.mainloop()
