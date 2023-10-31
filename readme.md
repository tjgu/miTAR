# miRNA target prediction
We developed a new hybrid deep learning approach to predict targets of miRNAs by integrating CNNs and RNNs. We applied the approach on two independent datasets obtained from two published studies, DeepMirTar (Ming Wen et al., 2018; https://academic.oup.com/bioinformatics/article/34/22/3781/5026656) and miRAW (Albert Pla et al., 2018; https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006185) and trained three models: miTAR1 was trained based on DeepMirTar dataset; miTAR2 was trained based on miRAW dataset; and miTAR was trained by combining both datasets. We recommended to use miTAR.

## Process for identifying the optimal parameters for each model
To obtain the best hyperparameters for each model, multiple hyperparameters were tested at a wide range: the learning rates at 0.2, 0.1, 0.05, 0.01, 0.005, and 0.001; the dropout rates at 0.1, 0.2, 0.3, 0.4, and 0.5; and the batch sizes at 10, 30, 50, 100, and 200. The program, parameter_sel_CNN_BiRNN_DeepMirTar_miTAR1.py, was used to identify the best parameters for miTAR1; parameter_sel_CNN_BiRNN_miRAW_miTAR2.py was used to identify the best parameters for miTAR2; and parameter_sel_CNN_BiRNN_MirTarRAW_miTAR.py was used to identify the best parameters for miTAR. After obtaining the best hyperparameters, each dataset was randomly split 30 times into training, validation and test datasets and the performance of each model was reported using the average accuracy with 95% confidence interval. evals_CNN_BiRNN_DeepMirTar_miTAR1_sampling.py, evals_CNN_BiRNN_miRAW_miTAR2_sampling.py, and evals_CNN_BiRNN_MirTarRAW_miTAR_sampling.py were used to perform 30 times' training, validation and test separately for each model. The three models with the best parameters were saved in results folder. They are miTAR1_CNN_BiRNN_b30_lr0.005_dout0.2.h5, miTAR2_CNN_BiRNN_b200_lr0.1_dout0.4.h5, and miTAR_CNN_BiRNN_b100_lr0.005_dout0.2.h5.

## Tools for predicting miRNA targets using trained models
predict_onemiRmultimRNA.py can be used to predict the targets for one miRNA. An example is shown here:

    python predict_onemiRmultimRNA.py -i1 data/tests/mrna_multiple_mixedLongShort.fa -i2 data/tests/mirna_hsa-miR-139-5p.txt -o results/tests/mir139-5p_predictedTar.fa -p 0.8 -ns 1 -s 22 
The meaning of the parameters can be obtained by python predict_onemiRmultimRNA.py -h

predict_multimiRmultimRNA.py can be used to predict the targets for multiple miRNAs. An example is shown here: 

    python predict_multimiRmultimRNA.py -i1 data/tests/mrna_multiple_mixedLongShort.fa -i2 data/tests/mirna_multiple.fa -o results/tests/mirna_multiple_predictedTar.fa -s 22 -p 0.8 -ns 1
The meaning of the parameters are the same as predict_onemiRmultimRNA.py and can be otained by python predict_multimiRmultimRNA.py -h

**The model used in the predict_multimiRmultimRNA.py and predict_onemiRmultimRNA.py is miTAR. If you would like to use a different model, please replace the model with another one, and change the fragment length to match the model input.**

## Understanding the output
The results from predict_onemiRmultimRNA.py and predict_multimiRmultimRNA.py were saved to the file following -o, such as results/tests/mirna_multiple_predictedTar.fa as shown in the above example. In the fasta file, each line following > contains the miRNA/gene information. For example, the first line in the results/tests/mirna_multiple_predictedTar.fa file: >hsa-miR-22-3p|1_long|2|4|0.9964378. 
hsa-miR-22-3p is the miRNAID; 1_long is the geneID; 2 is the total number of target site identified as the target location in one gene ( in this case, two miRNA target sites were identified in 1_long gene); 4 is the number of target site tested starting from 0 (in this case, the 4th target site); 0.9964378 is the probability of the prediction for this specific target site.

The results summary has the same meaning: [miRNAID|targetID, the target site number, the prediction probability for this target site], the total number of target sites have been found. 

## Requirements
The following packages should be installed before running the scripts

python: >=3.6

tensorflow: 1.14.0

keras: 2.2.4

h5py: 2.9.0

numpy

matplotlib

scipy

sklearn

The input of our tools are: whole sequence/sequences of miRNA/miRNAs from 5'->3'; gene sequence/sequences from 5'->3'. For longer gene sequence, the tools will automatically split into overlapped short sequences that matches the model requirement. 

## conda
This directory contains a Conda package with all the packages and dependencies installed. We supplied the package for 11 platforms. The package is renamed as mitar.
###### How to use it:
Download the package and install it on your local machine by 
`conda install --use-local mitar`

## mitar
This directory contains the package of miTAR, which renamed: mitar-0.0.1.tar.gz but without dependencies installed.
###### How to use it:
Download the package, uncompress it and use the scripts directly or install by 
`python setup.py install`

## scripts_data_models
This directory contains all the raw scripts, the data used for generating the models (data directory), and the trained models (in results directory).


## Report issues:
If you have any questions or suggestions or meet any issues, please contact Tongjun Gu: tgu at ufl.edu.
   
