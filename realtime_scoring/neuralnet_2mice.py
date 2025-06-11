import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from ctypes import *
from cffi import FFI
import os
import platform
import sys
import time
import random
import numpy as np
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from serial import Serial
import mne
import tdt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
import scipy
from scipy import signal
   
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
        
        self.libName = 'PO8eStreaming';
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
    
    def connectToCard(self, card=0, port=0, dtype= np.float32): #, sort_codes=None, bits_per_bin=None):
        '''
        if data is 32-bit, use dtype to specify how to interpret this
        this could be np.float32, np.int32, np.uint32
        '''
        index = card * self.PORTS_PER_CARD + port
        self.CardPointer[index] = self.dll.connectToCard(card, port)
        self.dtype[index] = dtype
        # self.sort_codes[index] = sort_codes
        # self.bits_per_bin[index] = bits_per_bin
    
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
        
    def waitForDataReady(self, index, timeout=2**31-1):
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
        
    def bandpass_filt(self, fs,sig,lowcut,highcut,polynomial):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b,a = signal.butter(polynomial,[low, high],btype='bandpass', analog = False)
        filt_data = signal.filtfilt(b,a,sig)
        return filt_data

    def notch_filt(self, fs,sig,f0,Q):
        b, a = signal.iirnotch(f0, Q, tdt_fs)
        filt_data = signal.filtfilt(b,a,sig)
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

    # NAME FILES
    mouse_num = 'mouse_756_757'
    date = '250422'

    # Define the file path to load the model
    model_path = 'sleep_scoring_model_241103.pth'

    # Initialize a new model instance with the same architecture
    model = SleepScoringModel().to(device)

    # Load the model's state_dict from the file path
    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"Model loaded from {model_path}")

    scaler_CH1 = joblib.load('scaler_CH1_241103.pkl')
    scaler_CH2 = joblib.load('scaler_CH2_241103.pkl')
    scaler_CH3 = joblib.load('scaler_CH3_241103.pkl')
    scaler_CH4 = joblib.load('scaler_CH4_241103.pkl')

    sleep_scores = []
    corresponding_sigs = []
    elapsed_times = []
    total_elapsed_times = []
    tdt_fs = 610
    fs = 128
    time_chunk = 10 # in seconds

    root = tk.Tk()
    root.title("real-time score (10s bin)")
    root.resizable(width=False, height=False)
    frame1=Frame(root, bg = 'black')
    frame1.grid(row=0, column=0)
    frame2=Frame(root)
    frame2.grid(row=1, column=0, pady=10)
    plt.style.use('dark_background')
    fig,ax = plt.subplots(nrows = 8, ncols = 1,figsize = (5, 8))
    # ax0 = fig.add_subplot(111)
    x_axis = np.linspace(0,time_chunk,time_chunk*fs)
    y_axis = np.zeros(time_chunk*fs)
    ax0, = ax[0].plot(x_axis,y_axis, c = 'c', lw=0.5)
    ax1, = ax[1].plot(x_axis,y_axis, c = 'm', lw=0.5)
    ax2, = ax[2].plot(x_axis,y_axis, c = 'y', lw=0.5)
    ax3, = ax[3].plot(x_axis,y_axis, c = 'w', lw=0.5)
    ax4, = ax[4].plot(x_axis,y_axis, c = 'c', lw=0.5)
    ax5, = ax[5].plot(x_axis,y_axis, c = 'm', lw=0.5)
    ax6, = ax[6].plot(x_axis,y_axis, c = 'y', lw=0.5)
    ax7, = ax[7].plot(x_axis,y_axis, c = 'w', lw=0.5)

    canvas = FigureCanvasTkAgg(fig, master = frame1)  
    canvas.get_tk_widget().grid(row=0,column=0)

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
                tdt.connectToCard(card, port, dtype=np.float32) #, sort_codes=4, bits_per_bin=1)
            else:
                # second port receives floats.
                tdt.connectToCard(card, port)

            index = card * tdt.PORTS_PER_CARD + port
            if (tdt.isNull(index)):
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
        # print(numSamples)
        time.sleep(0.05)

    stoppedCount = 0;
    print('Stream started')
    # for port in range(portCount):
    #     tdt.getStreamInfo(port)
    #     print('port {} has {} channels of {} byte data'.format(port, tdt.nchannels[port], tdt.sample_size[port]))

    # waitCount = [0 for i in range(portCount)]
    loopCt = 0
    skipCt = 0
    lastPrintTime = time.perf_counter()
    tic = time.time()
    toc = time.time()

    while stoppedCount < portCount:


        # time.sleep(1)
        # if portCount == 1:
        #     tdt.waitForDataReady(0)
        
        printMsg = ''

    # for port in range(portCount):
        port = 0
        # print(tdt.startCollecting(0, 1))

        numSamples = 0
        while numSamples < time_chunk*tdt_fs: 
            numSamples = tdt.samplesReady(0)
            # if tdt.isStopped(port)==True:
            #     break
        if numSamples == time_chunk*tdt_fs:
            tdt.readBlock(port, numSamples)
            if tdt.status[port] == 0:
                break
            else:
                buffer_time = time.time()-toc
                print(f'time from last buffer: {buffer_time}')
                elapsed_times.append(buffer_time)
                toc = time.time()
                total_elapsed_time = toc-tic
                total_elapsed_times.append(total_elapsed_time)
                print(f'total time elapsed: {total_elapsed_time}s')
                tdt.flushBufferedData(port, numSamples, 0)
                m1_R_SSC = tdt.data[0][2]
                m1_L_SSC = tdt.data[0][3]
                m1_R_EMG = tdt.data[0][4]
                m1_L_EMG = tdt.data[0][5]

                m2_R_SSC = tdt.data[0][10]
                m2_L_SSC = tdt.data[0][11]
                m2_R_EMG = tdt.data[0][12]
                m2_L_EMG = tdt.data[0][13]

                m1_signals = np.stack([m1_R_SSC, m1_L_SSC, m1_R_EMG, m1_L_EMG])
                channel_names = ['R_SSC', 'L_SSC', 'R_EMG', 'L_EMG']
                info = mne.create_info(channel_names, tdt_fs, ch_types=['eeg', 'eeg', 'emg', 'emg'],verbose=False)
                m1_raw = mne.io.RawArray(m1_signals, info, first_samp=0, copy='auto', verbose=False)
                m1_raw.resample(fs)

                m2_signals = np.stack([m2_R_SSC, m2_L_SSC, m2_R_EMG, m2_L_EMG])
                channel_names = ['R_SSC', 'L_SSC', 'R_EMG', 'L_EMG']
                info = mne.create_info(channel_names, tdt_fs, ch_types=['eeg', 'eeg', 'emg', 'emg'],verbose=False)
                m2_raw = mne.io.RawArray(m2_signals, info, first_samp=0, copy='auto', verbose=False)
                m2_raw.resample(fs)

                m1_y_hat = []
                m2_y_hat = []

                m1_scaled_CH1 = scaler_CH1.transform(m1_raw._data[0].reshape(1,1280))
                m1_scaled_CH2 = scaler_CH2.transform(m1_raw._data[1].reshape(1,1280))
                m1_scaled_CH3 = scaler_CH3.transform(m1_raw._data[2].reshape(1,1280))
                m1_scaled_CH4 = scaler_CH4.transform(m1_raw._data[3].reshape(1,1280))

                m2_scaled_CH1 = scaler_CH1.transform(m2_raw._data[0].reshape(1,1280))
                m2_scaled_CH2 = scaler_CH2.transform(m2_raw._data[1].reshape(1,1280))
                m2_scaled_CH3 = scaler_CH3.transform(m2_raw._data[2].reshape(1,1280))
                m2_scaled_CH4 = scaler_CH4.transform(m2_raw._data[3].reshape(1,1280))

                corresponding_sigs.append([m1_scaled_CH1, m1_scaled_CH2, m1_scaled_CH3, m1_scaled_CH4, m2_scaled_CH1, m2_scaled_CH2, m2_scaled_CH3, m2_scaled_CH4])

                ax0.set_xdata(x_axis)
                ax1.set_xdata(x_axis)
                ax2.set_xdata(x_axis)
                ax3.set_xdata(x_axis)
                ax4.set_xdata(x_axis)
                ax5.set_xdata(x_axis)
                ax6.set_xdata(x_axis)
                ax7.set_xdata(x_axis)

                ax0.set_ydata(m1_scaled_CH1)
                ax1.set_ydata(m1_scaled_CH2)
                ax2.set_ydata(m1_scaled_CH3)
                ax3.set_ydata(m1_scaled_CH4)
                ax4.set_ydata(m2_scaled_CH1)
                ax5.set_ydata(m2_scaled_CH2)
                ax6.set_ydata(m2_scaled_CH3)
                ax7.set_ydata(m2_scaled_CH4)

                ax[0].set_ylabel('R EEG (norm)')
                ax[1].set_ylabel('L EEG (norm)')
                ax[2].set_ylabel('R EMG (norm)')
                ax[3].set_ylabel('L EMG (norm)')
                ax[4].set_ylabel('R EEG (norm)')
                ax[5].set_ylabel('L EEG (norm)')
                ax[6].set_ylabel('R EMG (norm)')
                ax[7].set_ylabel('L EMG (norm)')
                ax[7].set_xlabel('time (s)')
                for x in range(8):
                    ax[x].set_xlim(0,time_chunk)
                    ax[x].set_xticks([0,time_chunk])
                ax[0].set_ylim(-3.5,3.5)
                ax[1].set_ylim(-3.5,3.5)
                ax[2].set_ylim(-0.1,0.1)
                ax[3].set_ylim(-0.1,0.1)
                ax[4].set_ylim(-3.5,3.5)
                ax[5].set_ylim(-3.5,3.5)
                ax[6].set_ylim(-0.1,0.1)
                ax[7].set_ylim(-0.1,0.1)
                plt.tight_layout()

                m1_X_data = np.vstack([m1_scaled_CH1,m1_scaled_CH2,m1_scaled_CH3,m1_scaled_CH4])

                tensor_sample = torch.tensor(m1_X_data, dtype=torch.float32).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():  # Disable gradient computation for inference
                    outputs = model(tensor_sample)
                _, predicted_label = torch.max(outputs, 1)
                m1_y_hat = predicted_label.item()

                m2_X_data = np.vstack([m2_scaled_CH1,m2_scaled_CH2,m2_scaled_CH3,m2_scaled_CH4])

                tensor_sample = torch.tensor(m2_X_data, dtype=torch.float32).unsqueeze(0).to(device)
                model.eval()
                with torch.no_grad():  # Disable gradient computation for inference
                    outputs = model(tensor_sample)
                _, predicted_label = torch.max(outputs, 1)
                m2_y_hat = predicted_label.item()

                if m1_y_hat==0:
                    m1_pred_label = 'wake'
                    print('m1: wake')
                    
                elif m1_y_hat==1:
                    m1_pred_label = 'NREM'
                    print('m1: NREM')
                    
                elif m1_y_hat==2:
                    m1_pred_label = 'REM'
                    print('m1: REM')

                if m2_y_hat==0:
                    m2_pred_label = 'wake'
                    print('m2: wake')
                    
                elif m2_y_hat==1:
                    m2_pred_label = 'NREM'
                    print('m2: NREM')
                    
                elif m2_y_hat==2:
                    m2_pred_label = 'REM'
                    print('m2: REM')

                if total_elapsed_time <= 28800 and m1_y_hat == 2:
                    syn.setParameterValue('PulseGen1', 'Enable', 1)
                elif 28800 < total_elapsed_time <= 57600 and m1_y_hat == 0:
                    syn.setParameterValue('PulseGen1', 'Enable', 1)
                elif 57600 < total_elapsed_time <= 86400 and m1_y_hat == 1:
                    syn.setParameterValue('PulseGen1', 'Enable', 1)
                if total_elapsed_time <= 28800 and m2_y_hat == 2:
                    syn.setParameterValue('PulseGen2', 'Enable', 1)
                elif 28800 < total_elapsed_time <= 57600 and m2_y_hat == 0:
                    syn.setParameterValue('PulseGen2', 'Enable', 1)
                elif 57600 < total_elapsed_time <= 86400 and m2_y_hat == 1:
                    syn.setParameterValue('PulseGen2', 'Enable', 1)

                sleep_scores.append([m1_y_hat, m2_y_hat])

                prediction_label = Label(master = frame2, text = f"m1 prediction: {m1_pred_label}, m2 prediction: {m2_pred_label}", font = ("Arial", 20), borderwidth = 3, relief = 'solid')
                prediction_label.pack()

                canvas.draw()
                root.update()
                canvas.flush_events()
                prediction_label.destroy()
                np.save(mouse_num+'_'+date+'_'+'sleep_scores.npy', sleep_scores)
                # np.save(mouse_num+'_'+date+'_'+'corresponding_sigs.npy', corresponding_sigs)
                # np.save(mouse_num+'_'+date+'_'+'elapsed_times.npy', elapsed_times)
                # np.save(mouse_num+'_'+date+'_'+'total_elapsed_times.npy', total_elapsed_times)

        else:
            tdt.flushBufferedData(port, numSamples, 0)
            print('SKIPPED')
            buffer_time = time.time()-toc
            print(f'time from last buffer: {buffer_time}')
            elapsed_times.append(buffer_time)
            toc = time.time()
            total_elapsed_time = toc-tic
            total_elapsed_times.append(total_elapsed_time)
            print(f'time elapsed: {total_elapsed_time}s')

            sleep_scores.append([255, 255])
            corresponding_sigs.append([np.zeros((1,1280)), np.zeros((1,1280)), np.zeros((1,1280)), np.zeros((1,1280)),
                np.zeros((1,1280)), np.zeros((1,1280)), np.zeros((1,1280)), np.zeros((1,1280))])
            skipCt += 1
            # print(numSamples)
            # waitCount[port] += 1
            np.save(mouse_num+'_'+date+'_'+'sleep_scores.npy', sleep_scores)
            # np.save(mouse_num+'_'+date+'_'+'corresponding_sigs.npy', corresponding_sigs)
            # np.save(mouse_num+'_'+date+'_'+'elapsed_times.npy', elapsed_times)
            # np.save(mouse_num+'_'+date+'_'+'total_elapsed_times.npy', total_elapsed_times)


        loopCt += 1
        if time.perf_counter() - toc > 10:
            stoppedCount += 1
            print('Disconnected')
        # if tdt.isStopped(port)==True:
        #     print('Disconnected')
        #     # stoppedCount += 1
        if tdt.status[port] == 0:
            print('Disconnected_2')
            stoppedCount += 1

        # print(loopCt)
        # print('{:>.3f} ms per loop'.format(1000*(time.perf_counter()-lastPrintTime)))

    for port in range(portCount):
        print('Releasing port', port)
        tdt.releaseCard(port)
        print('{:>.3f} s total time'.format((time.perf_counter()-lastPrintTime)))
        print('total num loops: {}'.format(loopCt))
        print('total num skips: {}'.format(skipCt))

    root.mainloop()





