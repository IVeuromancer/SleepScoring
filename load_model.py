import tensorflow as tf
from tensorflow import keras
import numpy as np
import tkinter as tk
from tkinter import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import time

test_CH1 = np.load('test_data/test_CH1.npy')
test_CH2 = np.load('test_data/test_CH2.npy')
test_CH3 = np.load('test_data/test_CH3.npy')
test_scores = np.load('test_data/test_scores.npy')

factor = 0.0006
# EEG1 = []
# for array in test_CH1:
#     max_abs = max(abs(array))
#     EEG1.append(array/max_abs)
X_EEG1 = np.array(test_CH1)/factor

# EEG2 = []
# for array in test_CH2:
#     max_abs = max(abs(array))
#     EEG2.append(array/max_abs)
X_EEG2 = np.array(test_CH2)/factor

# EMG = []
# factor = np.max(np.array(test_CH3))
# for array in test_CH3:
#     max_abs = max(abs(array))
#     EMG.append(array/factor) #max_abs
X_EMG = np.array(test_CH3)/factor

X_EMG_plot = test_CH3*1000000

y = np.array(test_scores)

# X_EEG1_ten = X_EEG1[0:3]
# X_EEG2_ten = X_EEG2[0:3]
# X_EMG_ten = X_EMG_plot[0:3]
# y_ten = y[0:3]
y_hat = []

model = keras.models.load_model('test_model')

root = tk.Tk()
root.title("real-time score (10s bin)")
# root.geometry("600x600")
root.resizable(width=False, height=False)
plt.style.use('dark_background')
fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (8, 4), sharex = True)
frame1=Frame(root, bg = 'black')
frame1.grid(row=0, column=0)
t = np.linspace(0,10,len(X_EEG1[0]))
ax0, = ax[0].plot(t, X_EEG1[0], color = 'c', lw = 0.5)
ax1, = ax[1].plot(t, X_EEG2[0], color = 'm', lw = 0.5)
ax2, = ax[2].plot(t, X_EMG_plot[0], color = 'y', lw = 0.5)
for data1, data2, data3_plot, data3, datay in zip(X_EEG1, X_EEG2, X_EMG_plot, X_EMG, y):
	ax0.set_xdata(t)
	ax0.set_ydata(data1)
	ax1.set_xdata(t)
	ax1.set_ydata(data2)
	ax2.set_xdata(t)
	ax2.set_ydata(data3_plot)
	ax[0].set_ylabel('EEG1 (norm)')
	ax[1].set_ylabel('EEG2 (norm)')
	ax[2].set_ylabel('EMG (uV)')
	ax[2].set_xlabel('time (s)')
	ax[2].set_ylim(-600,600)
	ax[2].set_yticks([-600,600])
	ax[2].set_xlim(0,10)
	ax[2].set_xticks([0,10])
	for x in range(2):
		ax[x].set_ylim(-1,1)
		ax[x].set_yticks([-1,0,1])
		ax[x].set_xlim(0,10)
		ax[x].set_xticks([0,10])
	plt.tight_layout()

	canvas = FigureCanvasTkAgg(fig, master = frame1)  
	canvas.get_tk_widget().grid(row=0,column=0)
	prediction = model.predict([data1.reshape(1,1280), data2.reshape(1,1280), data3.reshape(1,1280)])
	prediction_p = tf.nn.softmax(prediction)
	y_hat = (np.argmax(prediction_p))
	if y_hat==0:
		pred_label = 'wake'
	elif y_hat==1:
		pred_label = 'NREM'
	elif y_hat==2:
		pred_label = 'REM'
	else:
		pred_label = 'unscored'

	frame2=Frame(root)
	frame2.grid(row=1, column=0, pady=10)
	prediction_label = Label(master = frame2, text = f"prediction: {pred_label}", font = ("Arial", 20), borderwidth = 3, relief = 'solid')
	prediction_label.pack()

	if datay==0:
		sc_label = 'wake'
	elif datay==1:
		sc_label = 'NREM'
	elif datay==2:
		sc_label = 'REM'
	else:
		sc_label = 'unscored'

	frame3=Frame(root)
	frame3.grid(row=2, column=0, pady = 10)
	score_label = Label(master = frame3, text = f"score: {sc_label}", font = ("Arial", 20), borderwidth = 3, relief = 'solid')
	score_label.pack()


	###############    TOOLBAR    ###############
	# toolbarFrame = Frame(master=window)
	# toolbarFrame.grid(row=1,column=0)
	# toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

	canvas.draw()
	root.update()
	canvas.flush_events()
	prediction_label.destroy()
	score_label.destroy()

	time.sleep(5)

# yhat = np.array(y_hat).reshape(len(y_hat),1)
root.mainloop()


# for sleep scoring code -2 = Unscored, -1 = Artefact / Movement, 0 = Wake, 1 = N1 sleep, 1 = N2 sleep, 1 = N3 sleep, 2 = REM sleep





