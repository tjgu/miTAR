# random sampling 10 sets of samples for testing the performance of the best model
import re
import math
import random
import numpy as np

pos = []
neg = []
inputf = './data/data_miRaw_noL_noMisMissing_remained_seed1122.txt'
with open(inputf, 'r') as inf:
	for one in inf:
		one = one.rstrip('\r\n')
		values = one.split("\t")
		values[1] = "".join(values[1])
		one = "\t".join(values[:])
		if values[4] == '1':
			pos.append(one)
		elif values[4] == '0':
			neg.append(one)

posD = []
negD = []
inputfD = './data/data_DeepMirTar_removeMisMissing_remained_seed1122.txt'
with open(inputfD, 'r') as infD:
	next(infD)
	for one in infD:
		one = one.rstrip('\r\n')
		values = one.split("\t")
		if values[4] == '1':
			posD.append(one)
		elif values[4] == '0':
			negD.append(one)

seeds = [1234, 2345, 3456, 4567, 5678, 6789, 7890, 123, 234, 345, 456, 567, 678, 890, 11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777]

for rand_seed in seeds:
	np.random.seed(rand_seed)
	p = np.random.permutation(len(pos))
	newpos = [pos[t] for t in p[0:(3465*3)]]
	newpos2 = [pos[t] for t in p[(3465*3):]]
	
	p = np.random.permutation(len(neg))
	newneg = [neg[t] for t in p[0:(3465*3)]]
	newneg2 = [neg[t] for t in p[(3465*3):]]
	
	outfName = './data/sampling/data_DeepMirTar_miRAW_noRepeats_3folds_seed' + str(rand_seed) + '.txt'
	outfName2 = './data/sampling/data_miRAW_IndTest_noRepeats_3folds_seed' + str(rand_seed) + '.txt'
	with open(outfName, 'w+') as outf:
		for onepos in newpos:
			outf.write(onepos)
			outf.write('\n')
		for oneneg in newneg:
			outf.write(oneneg)
			outf.write('\n')
	
	outf.close()
	
	with open(outfName2, 'w+') as outf2:
		for onepos in newpos2:
			outf2.write(onepos)
			outf2.write('\n')
		for oneneg in newneg2:
			outf2.write(oneneg)
			outf2.write('\n')
	
	outf2.close()
	
	
	p = np.random.permutation(len(posD))
	newposD = [posD[t] for t in p[0:3465]]
	newpos2D = [posD[t] for t in p[3465:]]
	p = np.random.permutation(len(negD))
	newnegD = [negD[t] for t in p[0:3465]]
	newneg2D = [negD[t] for t in p[3465:]]
	
	outfName = './data/sampling/data_DeepMirTar_miRAW_noRepeats_3folds_seed' + str(rand_seed) + '.txt'
	outfName2 = './data/sampling/data_DeepMirTar_IndTest_noRepeats_3folds_seed' + str(rand_seed) + '.txt'
	with open(outfName, 'a+') as outf:
		for onepos in newposD:
			outf.write(onepos)
			outf.write('\n')
		for oneneg in newnegD:
			outf.write(oneneg)
			outf.write('\n')
	
	outf.close()
	
	with open(outfName2, 'w+') as outf2:
		for onepos in newpos2D:
			outf2.write(onepos)
			outf2.write('\n')
		for oneneg in newneg2D:
			outf2.write(oneneg)
			outf2.write('\n')
	
	outf2.close()
	
	
