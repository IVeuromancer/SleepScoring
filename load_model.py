import tensorflow as tf
from tensorflow import keras
import numpy as np
# import data_conditioning

test_CH1 = np.load('test_data/test_CH1.npy')
test_CH2 = np.load('test_data/test_CH2.npy')
test_CH3 = np.load('test_data/test_CH3.npy')
test_scores = np.load('test_data/test_scores.npy')

model = keras.models.load_model('test_model')

EEG1 = []
for array in test_CH1:
    max_abs = max(abs(array))
    EEG1.append(array/max_abs)
X_EEG1 = np.array(EEG1)

EEG2 = []
for array in test_CH2:
    max_abs = max(abs(array))
    EEG2.append(array/max_abs)
X_EEG2 = np.array(EEG2)

EMG = []
for array in test_CH3:
    max_abs = max(abs(array))
    EMG.append(array/max_abs)
X_EMG = np.array(EMG)

y = np.array(test_scores)

test_scores = model.evaluate(x=[X_EEG1,X_EEG2,X_EMG],
                            y=y, 
                           verbose = 2
                           )

