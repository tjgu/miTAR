import numpy as np
seed = 1122
np.random.seed(seed)

import h5py
import scipy.io
from keras.models import Sequential, load_model
from utils import formatDeepMirTar2, padding_len, convert3D, flatten, evals
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss


# prepare the input data
inputf = "./data/data_DeepMirTar_miRAW_noRepeats_3folds.txt"
seqs, label =  formatDeepMirTar2(inputf)
x = [x[0] for x in seqs]
x = padding_len(x, 79)
y = [int(y) for y in label]
x = np.array(x)

x_2 = x.reshape(x.shape[0], x.shape[1])
y_2 = np.array(y).reshape(len(y), 1)

percT = 0.2
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size=percT, random_state=seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=percT, random_state=seed)

bestModel = load_model('results/miTAR_CNN_BiRNN_b100_lr0.005_dout0.2.h5')
scores = bestModel.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred = bestModel.predict(X_test)
posthr = 0.5
negthr = 0.5
rm = 0

acc, sen, spe, Fmeasure, PPV, NPV = evals(y_test, y_pred, posthr, negthr, rm)
brierScore = brier_score_loss(y_test, y_pred)

print("Accuracy: " + str(acc) + " Sensitivity: " + str(sen) + " Specificity: " + str(spe) + " F-measure: " + str(Fmeasure) + " Positive predictive value: " + str(PPV) + " Negative predictive value: " + str(NPV) + "Brier score: " + str(brierScore))


inputf = "./data/data_miRaw_IndTest_noRepeats_3folds.txt"
#inputf = "./data/data_DeepMirTar_IndTest_noRepeats_3folds.txt"
seqs, label =  formatDeepMirTar2(inputf)
x = [x[0] for x in seqs]
x = padding_len(x, 79)
y = [int(y) for y in label]
x = np.array(x)

x_2 = x.reshape(x.shape[0], x.shape[1])
y_2 = np.array(y).reshape(len(y), 1)

scores = bestModel.evaluate(x_2, y_2, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

y_pred = bestModel.predict(x_2)
posthr = 0.5
negthr = 0.5
rm = 0

acc, sen, spe, Fmeasure, PPV, NPV = evals(y_2, y_pred, posthr, negthr, rm)
brierScore = brier_score_loss(y_2, y_pred)

print("Accuracy: " + str(acc) + " Sensitivity: " + str(sen) + " Specificity: " + str(spe) + " F-measure: " + str(Fmeasure) + " Positive predictive value: " + str(PPV) + " Negative predictive value: " + str(NPV) + "Brier score: " + str(brierScore))



