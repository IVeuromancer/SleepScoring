import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

test_CH1 = np.load('test_data/test_CH1.npy')
test_CH2 = np.load('test_data/test_CH2.npy')
test_CH3 = np.load('test_data/test_CH3.npy')
test_scores = np.load('test_data/test_scores.npy')

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
    EMG.append(array/0.0005621231996092901) #max_abs
X_EMG = np.array(EMG)

y = np.array(test_scores)

X_EEG1_ten = X_EEG1[0:2]
X_EEG2_ten = X_EEG2[0:2]
X_EMG_ten = X_EMG[0:2]
y_ten = y[0:2]

# model = keras.models.load_model('test_model')


plt.style.use('dark_background')
plt.ion()
fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (6, 4))
t = np.linspace(0,10,len(X_EEG1[0]))
ax0, = ax[0].plot(t, X_EEG1_ten[0], color = 'w', lw = 0.5)
ax1, = ax[1].plot(t, X_EEG2_ten[0], color = 'w', lw = 0.5)
ax2, = ax[2].plot(t, X_EMG_ten[0], color = 'grey', lw = 0.5)
for data1, data2, data3, datay in zip(X_EEG1_ten, X_EEG2_ten, X_EMG_ten, y_ten):
	ax0.set_xdata(t)
	ax0.set_ydata(data1)
	ax1.set_xdata(t)
	ax1.set_ydata(data2)
	ax2.set_xdata(t)
	ax2.set_ydata(data3)
	# for x in range(3):
	# 	ax[x].set_ylim(-2,2)
	# 	ax[x].set_yticks([-1,0,1])
	# 	ax[x].set_ylabel('norm V')
	# 	ax[x].set_xlim(0,10)
	# 	ax[x].set_xticks([0,10])
	# 	ax[x].set_xlabel('time (s)')
	fig.canvas.draw()
	fig.canvas.flush_events()

	time.sleep(2)



# test_scores = model.evaluate(x=[X_EEG1,X_EEG2,X_EMG],
#                             y=y, 
#                            verbose = 2
#                            )

