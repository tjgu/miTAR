import numpy as np
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

from utils import formatDeepMirTar2, padding, convert3D, flatten, evals

from sklearn.model_selection import train_test_split

epochs = 1000
batch = 50
lr = 0.005
dout = 0.5
fils = 320
ksize = 26

scores = []
vals = []
acc = 0
seeds = [1111, 2222, 3333, 7777, 55, 888, 111, 9999, 666, 1010]
for seed in seeds:
	inputf = 'data/data_miRaw_DeepMirTar_equal_training_noRepeats_3folds_seed' + str(seed) + '.txt'
	seqs, label =  formatDeepMirTar2(inputf)
	x = [x[0] for x in seqs]
	x = padding(x)
	y = [int(y) for y in label]
	
	x_2 = x.reshape(x.shape[0], x.shape[1])
	y_2 = np.array(y).reshape(len(y), 1)

	percT = 0.2
	X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=seed)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=percT, random_state=seed)

	np.random.seed(1122)
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
	mcp = ModelCheckpoint(filepath='results/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '_seed' + str(seed) + '_1122.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)
	lstm_CNN_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, validation_data=(X_valid, y_valid), verbose=2, callbacks=[es, mcp]).history
	
	bestModel = load_model('results/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '_seed' + str(seed) + '_1122.h5')

	score = bestModel.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (score[1]*100))
	scores.append(score[1]*100)

	y_pred = bestModel.predict_proba(X_test)
	posthr = 0.5
	negthr = 0.5
	rm = 0

	oneacc, sen, spe, Fmeasure, PPV, NPV = evals(y_test, y_pred, posthr, negthr, rm)
	vals.append([oneacc, sen, spe, Fmeasure, PPV, NPV])

	if score[1] > acc:
		acc = score[1]
		paras = [seed]
		print("best so far, acc=", acc, " paras=", paras)

	print("finish paras at: seed=", seed)


from statistics import mean 
aveScore = mean(scores)
print("the average accuracy is: ", aveScore)

aveEvals = []
for i in range(6):
	aveEvals.append(mean([vals[j][i] for j in range(len(vals))]))

print("The average evaluations are: ", aveEvals)

outfName = 'results/miTAR_CNN_BiRNN_b' + str(batch) + '_lr' + str(lr) + '_dout' + str(dout) + '_seeds_evals.txt'
with open(outfName, 'w+', encoding='utf-8') as outf:
	for i in range(len(vals)):
		outf.write("\t".join(str(vals[i][j]) for j in range(6)))
		outf.write('\n')
	outf.write("\t".join(str(aveEvals[i]) for i in range(6)))
	outf.write('\n')
	outf.write("\t".join(str(scores[i]) for i in range(10)))
	outf.write('\n')
	outf.write(str(aveScore))

outf.close()

