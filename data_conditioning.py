import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange, sample


def load_edf():
	# Load data
	import mne
	raw = mne.io.read_raw_edf('../EEG_recordings/346/Default_2022-09-21_10_00_27_export_346.edf', preload=True)
	#downsample
	raw.resample(128)
	sf = raw.info['sfreq']
	print('Chan =', raw.ch_names)
	print('Sampling frequency =', sf)
	print('Data shape (channels, times) =', raw._data.shape)
	# (128*60*60*23)+(128*60*30)+(128*20) #23hrs + 30min + 20sec
	#load sleep scores
	scores = pd.read_csv('d:/makinson_lab/sleep/EEG_recordings/346/Default_2022-09-21_10_00_27_export_346_scores.tsv', sep='\t')
	# in sirenia sleep, 255 unscored, 1 wake, 2 NREM, 3 REM, 4 "parameters"
	# in Yasa -2 = Unscored, -1 = Artefact / Movement, 0 = Wake, 1 = N1 sleep, 2 = N2 sleep, 3 = N3 sleep, 4 = REM sleep
	# for sleep scoring code -2 = Unscored, -1 = Artefact / Movement, 0 = Wake, 1 = N1 sleep, 1 = N2 sleep, 1 = N3 sleep, 2 = REM sleep
	scores_array = scores['scores']
	scores_converted = []
	for x in range(scores_array.shape[0]):
	    if scores_array[x]==1: # wake
	        scores_converted.append(0)
	    elif scores_array[x]==2: # NREM
	        scores_converted.append(1)
	    elif scores_array[x]==3: # REM
	        scores_converted.append(2)
	    elif scores_array[x]==255: # unscored
	        scores_converted.append(-2)
	all_scores = scores_converted[0:4384] # 4384: total number of epochs
	array_scores = np.array(all_scores).reshape(4384,1)

	CH1 = []
	CH2 = []
	CH3 = []
	for i in range(len(all_scores)): # chunking into 10 second arrays
	    CH1.append(raw._data[0][i*1280:(i+1)*1280])
	    CH2.append(raw._data[1][i*1280:(i+1)*1280])
	    CH3.append(raw._data[2][i*1280:(i+1)*1280])
	scores_list = array_scores.tolist()

	# next three cells are to randomize 60/20/20% 10s chunks into training, dev and test sets
	train_CH1 = []
	train_CH2 = []
	train_CH3 = []
	train_scores = []
	list_of_numbers = list(range(4384))
	for i in range(2630): #60% of 4384
	    num = list_of_numbers.pop(randrange(len(list_of_numbers)))
	    train_CH1.append(CH1[num])
	    train_CH2.append(CH2[num])
	    train_CH3.append(CH3[num])
	    train_scores.append(scores_list[num])

	dev_CH1 = []
	dev_CH2 = []
	dev_CH3 = []
	dev_scores = []
	for i in range(877): #20% of 4384
	    num = list_of_numbers.pop(randrange(len(list_of_numbers)))
	    dev_CH1.append(CH1[num])
	    dev_CH2.append(CH2[num])
	    dev_CH3.append(CH3[num])
	    dev_scores.append(scores_list[num])

	test_CH1 = []
	test_CH2 = []
	test_CH3 = []
	test_scores = []
	for i in range(877): #20% of 4384
	    num = list_of_numbers.pop(randrange(len(list_of_numbers)))
	    test_CH1.append(CH1[num])
	    test_CH2.append(CH2[num])
	    test_CH3.append(CH3[num])
	    test_scores.append(scores_list[num])

	np.save('test_data/train_CH1.npy', train_CH1)
	np.save('test_data/train_CH2.npy', train_CH2)
	np.save('test_data/train_CH3.npy', train_CH3)
	np.save('test_data/train_scores.npy', train_scores)

	np.save('test_data/dev_CH1.npy', dev_CH1)
	np.save('test_data/dev_CH2.npy', dev_CH2)
	np.save('test_data/dev_CH3.npy', dev_CH3)
	np.save('test_data/dev_scores.npy', dev_scores)

	np.save('test_data/test_CH1.npy', test_CH1)
	np.save('test_data/test_CH2.npy', test_CH2)
	np.save('test_data/test_CH3.npy', test_CH3)
	np.save('test_data/test_scores.npy', test_scores)

	return [train_CH1, train_CH2, train_CH3, train_scores, dev_CH1, dev_CH2, dev_CH3, dev_scores, 
	test_CH1, test_CH2, test_CH3, test_scores]



