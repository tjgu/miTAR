import numpy as np
sdnum=1122
np.random.seed(sdnum)

import random as rn
rn.seed(sdnum)

import tensorflow as tf
tf.set_random_seed(sdnum)

import h5py
import scipy.io

from keras.preprocessing import sequence
from keras.layers.core import Dropout, Activation
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras import optimizers
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import plot_model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from utils import formatDeepMirTar2, padding, convert3D, flatten

from sklearn.model_selection import train_test_split


# prepare the input data
inputf = "data/data_DeepMirTar_removeMisMissing_remained_seed1122.txt"
seqs, label =  formatDeepMirTar2(inputf)

x = [x[0] for x in seqs]
x = padding(x)
y = [int(y) for y in label]

x_2 = x.reshape(x.shape[0], x.shape[1])
y_2 = np.array(y).reshape(len(y), 1)

percT = 0.2
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=sdnum)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=percT, random_state=sdnum)
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

epochs = 1000
batches = [10, 30, 50, 100, 200]
learning_rate = [0.2, 0.1, 0.05, 0.01, 0.005, 0.001] 
dropout = [0.1, 0.2, 0.3, 0.4, 0.5] 
fils = 320
ksize = 12

acc = 0
for batch in batches:
	for lr in learning_rate:
		for dout in dropout:
			model = Sequential()
			model.add(Embedding(input_dim=5, output_dim=5, input_length=x.shape[1]))
			model.add(Conv1D(filters=fils, kernel_size=ksize, activation='relu'))
			model.add(Dropout(dout))
			model.add(MaxPooling1D(pool_size=2))
			model.add(Dropout(dout))
			model.add(Bidirectional(LSTM(32, activation='relu')))
			model.add(Dropout(dout))
			model.add(Dense(16, activation='relu'))
			model.add(Dropout(dout))
			model.add(Dense(1, activation='sigmoid'))

			model.summary()
			adam = optimizers.Adam(lr)
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

			es = EarlyStopping(monitor='val_acc', mode='max', min_delta=0.001, verbose=1, patience=100)
			mcp = ModelCheckpoint(filepath='results/miTAR1_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)
			lstm_CNN_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, y_valid), verbose=2, callbacks=[es, mcp]).history
			
			bestModel = load_model('results/miTAR1_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '.h5')

			scores = bestModel.evaluate(X_test, y_test, verbose=0)
			print("Accuracy: %.2f%%" % (scores[1]*100))

			if scores[1] > acc:
				acc = scores[1]
				paras = [batch, lr, dout]
				print("best so far, acc=", acc, " paras=", paras)

			print("finish paras at: batch=", batch, " lr=", lr, " dout=", dout)
