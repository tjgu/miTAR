"""
function:
	to predict one miRNA targests from a set of mRNAs
	the input miRNA is one seuence
	the input mRNAs are in fasta format
usage:

python predict_onemiRmultimRNA.py -i1 data/mrna_multiple_mixedLongShort.fa -i2 data/mirna-30e-5p.txt -o results/mirna-30e-3p_predictedTargets.fasta

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

parser = argparse.ArgumentParser(description="To predict the targets for one miRNA")

parser.add_argument("--input1", "-i1", help="the mRNA input file")
parser.add_argument("--input2", "-i2", help="the miRNA input file")
parser.add_argument("--output", "-o", help="the output file")
parser.add_argument("--step", "-s", default=15, help="the length between each two fragments")
parser.add_argument("--flen", "-fl", default=53, help="the length for each fragment")
parser.add_argument("--prob", "-p", default=0.5, help="the probability for selecting a target site")
parser.add_argument("--numsite", "-ns", default=0, help="the number of target sites")

args = parser.parse_args()

fasta_name = args.input1
onemirna = args.input2
outf = args.output
step = args.step
fraglen = args.flen
prob = args.prob
nums = args.numsite

headers, mrnaseqs = read_fasta(fasta_name)

# prepare the input data
mirnaf = open(onemirna, 'r')
mirnaseq = mirnaf.read().rstrip('\r\n').upper()

print(mirnaseq)
remirna = "".join(reversed(mirnaseq))
mirnaseq = remirna
print(remirna) 

maxlen = 79
maxlenmir = 26
mirnalen = len(mirnaseq)

if mirnalen < maxlenmir:
	mirnaseq = mirnaseq + 'N' * (maxlenmir - mirnalen)
	mirnalen = maxlenmir

if mirnalen + fraglen > maxlen:
	print ("the length of the miRNA and mRNA > maxlen!! reset the fraglen!")
	fraglen = maxlen - mirnalen
	print ("the new length for the mRNA fragment is: " + str(fraglen))

xs = []
for onemrna in mrnaseqs:
	mrnasplits = split_seqs_padding(onemrna, step, fraglen)
	mirnasplits = [mirnaseq] * len(mrnasplits)
	splitsboth = [i + j for i, j in zip(mirnasplits, mrnasplits)]
	splitsboth = [ch.replace('T', 'U') for ch in splitsboth]
	encode = dict(zip('NAUCG', range(5)))
	token = [[encode[ch] for ch in s] for s in splitsboth]
	xs.append((token, splitsboth))

if mirnalen + fraglen < maxlen:
	print ("mirna len + fragment len < max len! padding will be added to mRNA")
	xsp = []
	for onet, ones in xs:
		onep = padding_len(onet, maxlen)
		xsp.append((onep, ones))
	xs = xsp


model = load_model('results/miTAR_CNN_BiRNN_b50_lr0.005_dout0.5.h5')

res = []
resseq = []
count = 0
for onemirna in xs: # the same miRNA
	ys = []
	for onesplit in onemirna[0]: # the same gene
		y_pre = model.predict(np.array(onesplit).reshape(1, maxlen))
		ys.append(y_pre[0][0])
	mk = 0
	loci = []
	lociseq = []
	locinum = 0
	for y in ys:
		if y > float(prob):
			mk = mk + 1
			loci.append([locinum, y])
			lociseq.append((headers[count], locinum, y, onemirna[1][locinum]))
		locinum = locinum + 1
	if mk > int(nums):
		res.append([headers[count], mk, loci])
		resseq.append((lociseq, mk))
	count = count + 1

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

