"""
function:
	to predict one miRNA targests from a set of mRNAs
	the input miRNA is one seuence
	the input mRNAs are in fasta format
usage:

python predict_multimiRmultimRNA.py -i1 data/mrna_multiple_mixedLongShort.fa -i2 data/mirna_multiple.fa -o results/miRNAs_predictedTargets.fa

how to reimport a function:
from importlib import reload
import utils
reload(utils)
from utils import split_seqs_padding

"""

import numpy as np
import re
seed = 1122
np.random.seed(seed)

from utils import split_seqs_padding, padding_len, evals, read_fasta, revs_comple
from keras.models import load_model

import argparse

parser = argparse.ArgumentParser(description="To predict the targets for multiple miRNAs")

parser.add_argument("--input1", "-i1", help="the mRNA input file")
parser.add_argument("--input2", "-i2", help="the miRNA input file")
parser.add_argument("--output", "-o", help="the output file")
parser.add_argument("--step", "-s", default=15, help="the length between each two fragments")
parser.add_argument("--flen", "-fl", default=53, help="the length for each fragment")
parser.add_argument("--prob", "-p", default=0.5, help="the calling probability")
parser.add_argument("--numsite", "-ns", default=0, help="the number of called target sites")

args = parser.parse_args()

fasta_mrna = args.input1
fasta_mirna = args.input2
outf = args.output
step = args.step
fraglen = args.flen
prob = args.prob
nums = args.numsite

hmrna, mrnaseqs = read_fasta(fasta_mrna)
hmirna, mirnaseqs = read_fasta(fasta_mirna)

# prepare the input data
remirnas = []
for one in mirnaseqs:
	oneremirna = "".join(reversed(one))
	remirnas.append(oneremirna.upper())

print(mirnaseqs[0]) 
print(remirnas[0]) 
mirnaseqs = remirnas

maxlen = 79
maxlenmir = 26
mirnalen = len(min(mirnaseqs, key=len))
mirnalenm = len(max(mirnaseqs, key=len))

if mirnalenm > maxlenmir:
	print ("the max length of miRNAs > max len mir! need to remove the following mirna!")
	print (max(mirnaseqs, key=len))

if mirnalen < maxlenmir:
	mirnasp = []
	for one in mirnaseqs:
		onelen = len(one)
		onemirnaseq = one + 'N' * (maxlenmir - onelen)
		mirnasp.append(onemirnaseq)
	mirnalen = maxlenmir
	mirnaseqs = mirnasp

if mirnalen + fraglen > maxlen:
	print ("the length of the miRNA and mRNA > maxlen!! reset the fraglen!")
	fraglen = maxlen - mirnalen
	print ("the new length for the mRNA fragment is: " + str(fraglen))

xs = []
locimi = 0
for onemirna in mirnaseqs:
	locim = 0
	for onemrna in mrnaseqs:
		mrnasplits = split_seqs_padding(onemrna, step, fraglen)
		mirnasplits = [onemirna] * len(mrnasplits)
		splitsboth = [i + j for i, j in zip(mirnasplits, mrnasplits)]
		splitsboth = [ch.replace('T', 'U') for ch in splitsboth]
		encode = dict(zip('NAUCG', range(5)))
		token = [[encode[ch] for ch in s] for s in splitsboth]
		xs.append((token, splitsboth, "|".join((hmirna[locimi], hmrna[locim]))))
		locim = locim + 1
	locimi = locimi + 1

if mirnalen + fraglen < maxlen:
	print ("mirna len + fragment len < max len! padding will be added to mRNA")
	xsp = []
	for onet, ones, oneid in xs:
		onep = padding_len(onet, maxlen)
		xsp.append((onep, ones, oneid))
	xs = xsp


model = load_model('results/miTAR_CNN_BiRNN_b50_lr0.005_dout0.5.h5')

res = []
resseq = []
for onemirna in xs: # the same miRNA
	ys = []
	locinum = 0
	for onesplit in onemirna[0]: # the same gene
		y_pre = model.predict(np.array(onesplit).reshape(1, maxlen))
		ys.append((onemirna[2], y_pre[0][0], onemirna[1][locinum], locinum))
		locinum = locinum + 1
	mk = 0
	loci = []
	lociseq = []
	for y in ys:
		if y[1] > float(prob):
			mk = mk + 1
			loci.append([y[0], y[3], y[1]])
			lociseq.append((y[0], y[3], y[1], y[2]))
	if mk > int(nums):
		res.append([loci, mk])
		resseq.append((lociseq, mk))

print("finish analyzing and will save the results")
print("")

if res == []:
	print("no target gene!")
else:
	print("here is the summary of the results:")
	for oneres in res: 
		print(oneres)
	with open(outf, 'w+') as ofl:
		for one in resseq: # for gene
			for i in range(one[1]):
				onegnum = one[1]
				fasta = ">" + one[0][i][0] + "|" + str(onegnum) + "|" + str(one[0][i][1]) + "|" + str(one[0][i][2]) + "\n" + str(one[0][i][3]) + "\n"
				ofl.write(fasta)
	ofl.close()

