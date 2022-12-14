import tensorflow as tf
from tensorflow.keras import Input, Model, models, layers
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import data_conditioning

def get_uncompiled_model():
    input_EEG1 = Input(shape=(1280,), name = 'EEG1')
    input_EEG2 = Input(shape=(1280,), name = 'EEG2')
    input_EMG = Input(shape=(1280,), name = 'EMG')
    inputs = layers.concatenate([input_EEG1, input_EEG2, input_EMG], name = 'combined_inputs')
    x = layers.Dense(9, activation = "relu", name = 'layer_1')(inputs)
    x = layers.Dense(3, activation = "relu", name = 'layer_2')(x)
    outputs = layers.Dense(3, activation = "linear", name = 'outputs')(x)
    model = Model(inputs = [input_EEG1, input_EEG2, input_EMG], outputs = [outputs], name = 'sleep_scoring_model')
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"],
    )
    return model

def train():


	# [train_CH1, train_CH2, train_CH3, train_scores, dev_CH1, dev_CH2, dev_CH3, dev_scores, 
	# test_CH1, test_CH2, test_CH3, test_scores] = data_conditioning.load_edf()

	train_CH1 = np.load('test_data/train_CH1.npy')
	train_CH2 = np.load('test_data/train_CH2.npy')
	train_CH3 = np.load('test_data/train_CH3.npy')
	train_scores = np.load('test_data/train_scores.npy')

	dev_CH1 = np.load('test_data/dev_CH1.npy')
	dev_CH2 = np.load('test_data/dev_CH2.npy')
	dev_CH3 = np.load('test_data/dev_CH3.npy')
	dev_scores = np.load('test_data/dev_scores.npy')

	model = get_compiled_model()

	factor = 0.0006
	# EEG1 = []
	# for array in train_CH1:
	#     max_abs = max(abs(array))
	#     EEG1.append(array/max_abs)
	X_EEG1 = np.array(train_CH1)/factor

	# EEG2 = []
	# for array in train_CH2:
	#     max_abs = max(abs(array))
	#     EEG2.append(array/max_abs)
	X_EEG2 = np.array(train_CH2)/factor

	# EMG = []
	# factor = np.max(np.array(train_CH3))
	# for array in train_CH3:
	#     max_abs = max(abs(array))
	#     EMG.append(array/factor) #max_abs
	X_EMG = np.array(train_CH3)/factor

	y = np.array(train_scores)


	history = model.fit({'EEG1':X_EEG1, 'EEG2':X_EEG2, 'EMG': X_EMG},
	                    {'outputs': y},
	    epochs=200
	)
	fig = plt.figure(figsize = [10,5])
	# plt.style.use('dark_background')
	plt.plot(history.history['loss'])
	plt.plot(history.history['accuracy'])
	plt.xlabel('epoch')
	plt.ylabel('loss/accuracy')
	plt.title('Learning Curve')
	plt.show()

	EEG1 = []
	for array in dev_CH1:
	    max_abs = max(abs(array))
	    EEG1.append(array/max_abs)
	X_EEG1 = np.array(EEG1)

	EEG2 = []
	for array in dev_CH2:
	    max_abs = max(abs(array))
	    EEG2.append(array/max_abs)
	X_EEG2 = np.array(EEG2)

	EMG = []
	for array in dev_CH3:
	    max_abs = max(abs(array))
	    EMG.append(array/max_abs)
	X_EMG = np.array(EMG)

	y = np.array(dev_scores)

	dev_scores = model.evaluate(x=[X_EEG1,X_EEG2,X_EMG],
	                            y=y, 
	                           verbose = 2
	                           )
	return model

# model = train()