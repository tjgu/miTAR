import re
import math
import random
import numpy as np
import keras.backend as K
from six.moves import cPickle as pk

def formatRNA(filename):
	"""
	function:
		tokenize the sequences into 0,1,2,3,4 and saved the token into a list.

	inputs:
		filename - the input file name. The format is as following:
		miRNA gene_name EnsemblId Positive_Negative Mature_mirna_transcript 3UTR_transcript
	returns:
		A list containing the sequences that combine both miRNA and 3UTR, and the gene names
	"""
	seqs = list()
	with open(filename, 'r') as infl:
		for line in infl:
			values = line.split("\t")
			seq2 = values[4] + values[5]
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			encode = dict(zip('NAUCG', range(5)))
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[1] + "_" + values[2]
			seqs.append((token, name))

	return seqs

def toFasta(filename):
	"""
	function:
		convert tab separated file format to fasta format

	inputs:
		filename - the input file name. The format is as following:
		each line contains the headers and the sequence separated by tab; the last element is the sequence
	returns:
		a list containing the sequences in fasta format
	"""
	seqs = list()
	with open(filename, 'r') as infl:
		for line in infl:
			values = line.split("\t")
			num = len(values)
			seq = values[num-1].rstrip('\r\n')
			name = ">" + "_".join(values[0:(num-2)])
			seqs.append((name, seq))

	return seqs

def formatLmRNA(filename, maxlenmir, step, fraglen, maxlen, miloci, mloci, rev):
	"""
	function:
		extract the sequences and labels from tab separated files; then tokenize the sequences into 0,1,2,3,4 and saved the token into a list.

	inputs:
		filename - the input file name.
		maxlenmir - the max length of the miRNA sequence in the existing model.
		step - the length of the overlaps between each two fragments.
		fraglen - the length of the fragments that will be split to.
		maxlen - the max length of the mRNA in the existing model.
		miloci - the location of miRNA sequences in the input file.
		mloci - the location of the mRNA sequences in the input file.
		rev - whether to reverse the miRNA sequences. 1: reverse.
	returns:
		A list containing the sequences that combine both miRNA and 3UTR, and the gene names
	"""
	seqs = list()
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			mrna = re.sub('T', 'U', values[mloci].rstrip('\r\n'))

			if rev == 1:
				mirna = "".join(reversed(values[miloci]))
			else:
				mirna =values[miloci]
			mirnalen = len(mirna)

			if mirnalen < maxlenmir:
				mirna = mirna + 'N' * (maxlenmir - mirnalen)
				mirnalen = maxlenmir

			if len(mrna) < maxlen:
				mrnaseqs = mrna + 'N' * (maxlen - len(mrna))
				mrnaseqs = [mrnaseqs]
			else:
				mrnaseqs = split_seqs(mrna, step, fraglen)

			#print (mrnaseqs)	
			mirnaseqs = [mirna] * len(mrnaseqs)
			seq2 = [i + j for i, j in zip(mirnaseqs, mrnaseqs)]
			encode = dict(zip('NAUCG', range(5)))
			token = [[encode[ch] for ch in s] for s in seq2]
			name = values[0] + "_" + values[1] + "_" + values[2]
			seqs.append((token, name))

	return seqs

def formatLmRNALabel(filename, maxlenmir, step, fraglen, maxlen, miloci, mloci, rev, lloci):
	"""
	function:
		extract the sequences and labels from tab separated files; then tokenize the sequences into 0,1,2,3,4 and saved the token into a list.

	inputs:
		filename - the input file name.
		maxlenmir - the max length of the miRNA sequence in the existing model.
		step - the length of the overlaps between each two fragments.
		fraglen - the length of the fragments that will be split to.
		maxlen - the max length of the mRNA in the existing model.
		miloci - the location of miRNA sequences in the input file.
		mloci - the location of the mRNA sequences in the input file.
		rev - whether to reverse the miRNA sequences. 1: reverse.
		lloci - the location of the label in the input file. 
	returns:
		A list containing the sequences that combine both miRNA and 3UTR, and the gene names
	"""
	seqs = list()
	labels = list()
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			mrna = re.sub('T', 'U', values[mloci].rstrip('\r\n'))

			if rev == 1:
				mirna = "".join(reversed(values[miloci]))
			else:
				mirna =values[miloci]
			mirnalen = len(mirna)

			if mirnalen < maxlenmir:
				mirna = mirna + 'N' * (maxlenmir - mirnalen)
				mirnalen = maxlenmir

			if len(mrna) < maxlen:
				mrnaseqs = mrna + 'N' * (maxlen - len(mrna))
				mrnaseqs = [mrnaseqs]
			else:
				mrnaseqs = split_seqs(mrna, step, fraglen)

			#print (mrnaseqs)	
			mirnaseqs = [mirna] * len(mrnaseqs)
			seq2 = [i + j for i, j in zip(mirnaseqs, mrnaseqs)]
			encode = dict(zip('NAUCG', range(5)))
			token = [[encode[ch] for ch in s] for s in seq2]
			name = values[0] + "_" + values[1] + "_" + values[2]
			l = values[lloci]
			seqs.append((token, name))
			labels.append(l)

	return seqs, labels

def formatLmiRAW(filename, maxlenmir, maxlen, miloci, mloci, rev):
	"""
	function:
		extract the sequences and labels from miRAW database; then tokenize the sequences into 0,1,2,3,4 and saved the token into a list.

	inputs:
		filename - the input file name.
		maxlenmir - the max length of the miRNA sequence in the existing model.
		maxlen - the max length of the mRNA in the existing model.
		miloci - the location of miRNA sequences in the input file.
		mloci - the location of the mRNA sequences in the input file.
		rev - whether to reverse the miRNA sequences. 1: reverse.
	returns:
		A list containing the sequences that combine both miRNA and 3UTR, and the gene names
	"""
	seqs = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			mrna = re.sub('T', 'U', values[mloci].rstrip('\r\n'))
			mrna = re.sub('L', 'N', mrna)
			mrna = re.sub('X', 'N', mrna)
			mirna = re.sub('L', 'N', values[miloci].rstrip('\r\n'))
			mirna = re.sub('X', 'N', mirna)
			mirna = re.sub('T', 'U', mirna)

			if rev == 1:
				mirna = "".join(reversed(mirna))

			mirnalen = len(mirna)

			if mirnalen < maxlenmir:
				mirnaseq = mirna + 'N' * (maxlenmir - mirnalen)
				mirnalen = maxlenmir
			else:
				mirnaseq = mirna

			if len(mrna) + mirnalen < maxlen:
				mrnaseq = mrna + 'N' * (maxlen - mirnalen - len(mrna))
			elif len(mrna) + mirnalen > maxlen:
				mrnalen = maxlen - mirnalen
				mrnaseq = mrna[0:mrnalen]

			#print (mrnaseq)	
			seq2 = mirnaseq + mrnaseq
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[1] + "_" + values[2]
			seqs.append((token, name))

	return seqs

def formatDeepMirTar(filename):
	"""
	function:
		format the DeepMirTar data: tokenize the sequences into 0,1,2,3,4 and saved the token into a list.

	inputs:
		filename - the input file name. The format is as following:
		miRNA Mature_mirna_transcript_reversed gene_Id 3UTR_transcript label
	returns:
		seqs - A list containing the sequences that combine both miRNA and 3UTR, and the gene names.
		l - A list containing the labels.
	"""
	seqs = list()
	l = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			seq2 = values[1] + values[3]
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[2]
			seqs.append((token, name))
			l.append(values[4].rstrip('\r\n'))

	return seqs, l

def formatDeepMirTar2(filename):
	"""
	function:
		format the DeepMirTar data: tokenize the sequences into 0,1,2,3,4 and saved the token into a list.
		this function will pad the miRNA sequences with N if they are less than 26.

	inputs:
		filename - the input file name. The format is as following:
		miRNA Mature_mirna_transcript_reversed gene_Id 3UTR_transcript label
	returns:
		seqs - A list containing the sequences that combine both miRNA and 3UTR, and the gene names
		l - A list containing the labels.
	"""
	seqs = list()
	l = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			if len(values[1]) < 26:
				values[1] = values[1] + 'N' * (26 - len(values[1]))
			seq2 = values[1] + values[3]
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			#print('seq=' + str(seq2))
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[2]
			seqs.append((token, name))
			l.append(values[4].rstrip('\r\n'))

	return seqs, l

def formatDeepMirTarRev(filename):
	"""
	function:
		format the DeepMirTar data: tokenize the sequences into 0,1,2,3,4 and saved the token into a list.
		this function will pad the miRNA sequences with N if they are less than 26.

	inputs:
		filename - the input file name. The format is as following:
		miRNA Mature_mirna_transcript_reversed gene_Id 3UTR_transcript label
	returns:
		seqs - A list containing the sequences that combine both miRNA and 3UTR, and the gene names
		l - A list containing the labels.
	"""
	seqs = list()
	l = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			values[1] = "".join(reversed(values[1]))
			if len(values[1]) < 26:
				values[1] = values[1] + 'N' * (26 - len(values[1]))
			values[3] = "".join(reversed(values[3]))
			seq2 = values[1] + values[3]
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[2]
			seqs.append((token, name))
			l.append(values[4].rstrip('\r\n'))

	return seqs, l

def formatmiRAW(filename):
	"""
	function:
		format the miRAW data: tokenize the sequences into 0,1,2,3,4 and saved the token into a list.
		this function will reverse and pad the miRNA seqeuences if they are less than 26.

	inputs:
		filename - the input file name. The format is as following:
		miRNA gene_name EnsemblId Positive_Negative Mature_mirna_transcript 3UTR_transcript
	returns:
		seqs - A list containing the sequences that combine both miRNA and 3UTR, and the gene names
		l - A list containing the labels.
	"""
	seqs = list()
	l = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		next(infl)
		for line in infl:
			values = line.split("\t")
			if len(values[1]) < 26:
				values[1] = "".join(reversed(values[1])) + 'N' * (26 - len(values[1]))
			seq2 = values[1] + values[3]
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			token = [encode[s] for s in seq2.upper()]
			name = values[0] + "_" + values[2]
			seqs.append((token, name))
			l.append(values[4].rstrip('\r\n'))

	return seqs, l

def formatUnderScore(filename):
	"""
	function:
		tokenize the sequences into 0,1,2,3,4 and saved the token into a list.
		the headers will contain both the miRNA and mRNA gene names and other information separated by |.

	inputs:
		filename - the input file name. The miRNA and mRNA seqeuences are connected to the headers by _._ 
	returns:
		seqs - A list containing the sequences that combine both miRNA and 3UTR, and the gene names.
		l - A list containing the labels.
	"""
	seqs = list()
	l = list()
	encode = dict(zip('NAUCG', range(5)))
	with open(filename, 'r') as infl:
		for line in infl:
			values = line.split("\t")
			mirna = values[0].split("_")
			mrna = values[1].split("_")
			if len(mirna[1]) < 26:
				mirna[1] = mirna[1] + 'N' * (26 - len(mirna[1]))
#			print(mirna[1])
			seq2 = mirna[1] + mrna[(len(mrna)-1)]
#			print(seq2)
			seq2 = re.sub('T', 'U', seq2.rstrip('\r\n'))
			token = [encode[s] for s in seq2.upper()]
			name = mirna[0] + "_" + "_".join(mrna[0:(len(mrna)-1)]) + "|" +  "_".join(values[2:6])
			seqs.append((token, name))
			l.append(mirna[2].rstrip('\r\n'))

	return seqs, l

def evals(y, y_pred, posthr, negthr, rm):
	"""
	function:
		evaluate the model

	inputs:
		y - Input labels 
		y_pred - Predicted labels
		posthr - threshold for converting probability into 1: higher than posthr will be 1
		negthr - threshold for converting probability into 0: lower than negthr will be 0
		rm - values for handling probability between posthr and negthr: 0, remove these cases; 1, convert to 1; 2, convert to 0
	returns:
		Accuracy, sensititvity, specificity, F-measure, Positive predictive value, Negative predictive value
	"""
	new_y_pred = []
	for one in y_pred:
		if one > posthr:
			new_y_pred.append(1)
		elif one < negthr:
			new_y_pred.append(0)
		else:
			new_y_pred.append(2)
	inds = [i for i, x in enumerate(new_y_pred) if x == 2]
	if rm == 0:
		new_y = [i for j, i in enumerate(y) if j not in inds]
		new_y_pred = [i for j, i in enumerate(new_y_pred) if j not in inds]
		y = np.array(new_y)
	elif rm == 1:
		new_y_pred = [1 for j, i in enumerate(new_y_pred) if j in inds]
	elif rm == 2:
		new_y_pred = [0 for j, i in enumerate(new_y_pred) if j in inds]

	if y.shape[0] > 1:
		y = y.reshape(1, y.shape[0])
	y_pred = np.array(new_y_pred)
	if y_pred.shape[0] > 1:
		y_pred = y_pred.reshape(1, y_pred.shape[0])

	neg_y = 1 - y
	neg_y_pred = 1 - y_pred
	tp = ((y * y_pred) == 1).sum()
	fp = ((neg_y * y_pred) == 1).sum()
	fn = ((neg_y_pred * y) == 1).sum()
	tn = ((neg_y * neg_y_pred) == 1).sum()
	acc = (tp + tn) / (tp + tn + fp + fn + K.epsilon())
	sensitivity = tp / (tp + fn + K.epsilon())
	specificity = tn / (tn + fp + K.epsilon())
	Fmeasure = 2*tp / (2*tp + fp + fn + K.epsilon())
	PPV = tp / (tp + fp + K.epsilon())
	NPV = tn / (tn + fn + K.epsilon())

	return acc, sensitivity, specificity, Fmeasure, PPV, NPV

def longest_len(seqs):
	"""
	function:
		get the length for the longest sequence

	inputs:
		seqs - a set of sequences
	output: 
		the length for the longest sequence
	"""
	return len(max(seqs, key=len))

def split_seqs(seq, step, fraglen):
	"""
	function:
		split long sequence into short fragments

	inputs:
		seq - the long sequence
		step - how much can be overlapped between fragments
		fraglen - the length for the fragments
	output:
		seqs - a set of fragments
	"""
	lseq = len(seq.rstrip('\r\n'))
	st = 0
	end = fraglen
	seqs = list()
	while end < lseq:
		seqs.append(seq[st:end])
		st = st + step
		end = end + step
	if end >= lseq and st < lseq:
		st = lseq - fraglen
		seqs.append(seq[st:lseq])
	return seqs

def split_seqs_padding(seq, step, fraglen):
	"""
	function:
		split one sequence into short fragments; if the length of the sequence is less than the fragment, the sequence will be padded to the fragment length
	inputs:
		seq - the long sequence
		step - how much can be overlapped between fragments
		fraglen - the length for the fragments
	output:
		seqs - a set of fragments
	"""
	lseq = len(seq.rstrip('\r\n'))
	st = 0
	end = fraglen
	seqs = list()
	if end >= lseq:
		seq = seq + '0' * (fraglen-lseq) #input is string, such as 'CTACCAGTTCCTTGGGATGCTTTGGTGTTCTGCAAAGGCATCCTTAGTCT'
		seqs.append(seq)
	elif end < lseq:
		while end < lseq:
			seqs.append(seq[st:end])
			st = st + step
			end = end + step
		if end >= lseq and st < lseq and lseq :
			st = lseq - fraglen
			seqs.append(seq[st:lseq])
	return seqs

def splitPercent(seqs, percV, percT, seed):
	"""
	function:
		split the data into training, validation and test datasets based on the percentage

	inputs:
		seqs - the sequences that need to be split into training, validation and test sets.
		percV - the percentage for the validation set
		percT - the percentage for the test set in the total of the validation and test sets
		seed - the seed for shuffling the input sequences
	returns:
		seqsTr - the sequences for training
		seqsVa - the sequences for validation
		seqsTe - the sequences for test
	"""
	random.seed(seed)
	random.shuffle(seqs)
	tot = len(seqs)
	Tend = math.floor(tot * (1-percV))
	Vend = Tend + math.floor(tot * percV * (1-percT))

	seqsTr = seqs[0:Tend]
	seqsVa = seqs[Tend:Vend]
	seqsTe = seqs[Vend:tot]
	
	return seqsTr, seqsVa, seqsTe

def padding(seqs):
	"""
	function:
		add 0 to the end of the short arrays/list

	inputs:
		seqs - a list of sequences
	outputs:
		a padded array
	"""
	maxL = len(max(seqs, key=len))
	lens = [len(seq) for seq in seqs]
	seqsP = []
	for seq, seq_len in zip(seqs, lens):
		gap = maxL - seq_len
		seq = seq + [0] * gap
		seqsP.append(seq)

	return np.atleast_2d(seqsP)

def padding_string(seqs, maxl, chrs='0'):
	"""
	function:
		add 0 to the end of the short arrays/list

	inputs:
		seqs - a list of sequences
		maxl - the max length of the sequences
	outputs:
		a padded array
	"""
	lens = [len(seq) for seq in seqs]
	seqsP = []
	for seq, seq_len in zip(seqs, lens):
		gap = maxl - seq_len
		seq = seq + '0' * gap
		seqsP.append(seq)

	return seqsP

def padding_len(seqs, maxl):
	"""
	function:
		add 0 to the end of the short arrays/list

	inputs:
		seqs - a list of sequences
		maxl - the max length of the sequences
	outputs:
		a padded array
	"""
	lens = [len(seq) for seq in seqs]
	seqsP = []
	for seq, seq_len in zip(seqs, lens):
		gap = maxl - seq_len
		seq = seq + [0] * gap
		seqsP.append(seq)

	return seqsP

def split_seqs_padding(seq, step, fraglen):
	"""
	function:
		split one sequence into short fragments; if the sequence is less than the fragment, the sequence will be padded to the fragment length
	inputs:
		seq - the long sequence
		step - how much can be overlapped between fragments
		fraglen - the length for the fragments
	output:
		seqs - a set of fragments
	"""
	lseq = len(seq.rstrip('\r\n'))
	st = 0
	end = fraglen
	seqs = list()
	if end >= lseq:
		seq = seq + 'N' * (fraglen-lseq) #input is string: 'CTACCAGTTCCTTGGGATGCTTTGGTGTTCTGCAAAGGCATCCTTAGTCT'
		seqs.append(seq)
	elif end < lseq:
		while end < lseq:
			seqs.append(seq[st:end])
			st = st + int(step)
			end = end + int(step)
		if end >= lseq and st < lseq and lseq :
			st = lseq - fraglen
			seqs.append(seq[st:lseq])
	return seqs


def convert3D(x, y, timesteps):
	"""
	function:
		convert the data from 2D to 3D

	inputs:
		x - input x, a two dimensional array
		y - input y, a tuple
	returns:
		output_x - 3D array with timesteps. 
		output_y - not sure how to much x and y.
	"""
	output_x = []
	output_y = []
	for i in range(len(x) - timesteps - 1):
		t = []
		for j in range(1, timesteps + 1):
			t.append(x[i+j+1])
		output_x.append(t)
		output_y.append(y[i+timesteps+1])

	return np.array(output_x), np.array(output_y)

def prepareDatasets(posf, negf, percV, percT, seed):
	"""
	function:
		prepare the data for traning and testing

	Input:
		posf - the positive sequences that need to be split into training, validation and test sets.
		negf - the negative sequences that need to be split into training, validation and test sets.
		percV - the percentage for the validation set
		percT - the percentage for the test set in the total of the validation and test sets
		seed - the seed for shuffling the input sequences
	returns:
		TrX - the sequences for training
		TrY - the labels for training
		VaX - the sequences for validation
		VaY - the labels for validation
		TeX - the sequences for validation
		TeY - the labels for validation
	"""
	posTr, posVa, posTe = splitPercent(posf, percV, percT, seed)
	negTr, negVa, negTe = splitPercent(posf, percV, percT, seed)

	TrX = [x[0] for x in posTr] + [x[0] for x in negTr]
	TrY = list([1 for x in range(len(posTr))]) + list([0 for x in range(len(negTr))])
	VaX = [x[0] for x in posVa] + [x[0] for x in negVa]
	VaY = list([1 for x in range(len(posVa))]) + list([0 for x in range(len(negVa))])
	TeX = [x[0] for x in posTe] + [x[0] for x in negTe]
	TeY = list([1 for x in range(len(posTe))]) + list([0 for x in range(len(negTe))])

	return TrX, TrY, VaX, VaY, TeX, TeY

def flatten(X):
	"""
	function:
		Flatten a 3D array.

	Input:
		X - A 3D array for lstm, where the array is sample x timesteps x features.
	Output:
		flattened_X - A 2D array, sample x features.
	"""
	flattened_X = np.empty((X.shape[0], X.shape[2])) 
	for i in range(X.shape[0]):
		flattened_X[i] = X[i, (X.shape[1]-1), :]

	return flattened_X

def filterLen(x, y, ln):
	"""
	function:
		Filter sequences longer than ln.

	Input:
		x - a list of sequences
		y - a tuple that containing the labels
		ln - the length that will be used to filter the sequences
	Output:
		x_fl - sequences with no sequence longer than ln
		y_fl - labels without the label for the sequences longer than ln
	"""
	x_fl, y_fl = zip(*((x0, y0) for x0,y0 in zip(x, y) if len(x0) <= ln))
	
	return x_fl, y_fl

def loadDataDeepMirTar():
	"""
	function:
		load the data from DeepMirTar_SdA software and return the values and keys separately for the positive and negative cases.

	"""
	dir='/media/sf_D_DRIVE/Projects/2019/CancerCenter/miRNA_target/previous_work/DeepMirTar/DeepMirTar_SdA/data/'
	f = open(dir + 'Mark_features_dic_all.pkl', 'rb')
	Mark_features_dic = pk.load(f)
	f.close()

	f = open(dir + 'Mark_N_features_dic_all.pkl', 'rb')
	Mark_N_features_dic = pk.load(f)
	f.close()

	x_p_keys = Mark_features_dic.keys()
	x_p_values = Mark_features_dic.values()

	x_n_keys = Mark_N_features_dic.keys()
	x_n_values = Mark_N_features_dic.values()

	return x_p_keys, x_p_values, x_n_keys, x_n_values
    	
def read_fasta(fasta_name):
	"""
	function:
		read the fasta file and convert to two lists with one being the headers and one being the sequences.

	Input:
		fasta_name - input file name
	Output:
		headers - the headers of the input fasta file.
		seqs - the sequences of the input fasta file.
	"""
	from itertools import groupby
	fh = open(fasta_name)
	fas = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
	headers = []
	seqs = []
	for header in fas:
		headers.append(header.__next__()[1:].strip())
		seqs.append("".join(s.strip() for s in fas.__next__()))
	
	return headers, seqs

def revs_comple(onestr):
	"""
	function:
		reverse complementary the input sequence.

	Input:
		onestr - the input sequence.
	Output:
		chrs_re - the reverse complementary of the input sequence.
	"""
	chrs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'U': 'A'}
	chrs_re = "".join(chrs.get(ch, ch) for ch in reversed(onestr))

	return chrs_re

