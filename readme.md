# miRNA target prediction
A new hybrid deep learning algorithm was developed to predict the targets of miRNAs by integrating CNN and RNN. Since two independent datasets was download from two published studies: DeepMirTar (Ming Wen et al., 2018; https://academic.oup.com/bioinformatics/article/34/22/3781/5026656) and miRAW (Albert Pla et al., 2018; https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006185), three models were trained: miTAR1 was trained based on DeepMirTar dataset; miTAR2 was trained based on miRAW dataset; and miTAR was trained using both datasets.


To get the best hyperparameters for each model, multiple hyperparameters were tested at a wide range: the learning rates at 0.2, 0.1, 0.05, 0.01, 0.005, and 0.001; the dropout rates at 0.1, 0.2, 0.3, 0.4, and 0.5; and the batch sizes at 10, 30, 50, 100, and 200. The program, parameter_sel_CNN_BiRNN_DeepMirTar_miTAR1.py, was used to identify the best parameters for miTAR1; parameter_sel_CNN_BiRNN_miRAW_miTAR2.py was used to identify the best parameters for miTAR2; and parameter_sel_CNN_BiRNN_MirTarRAW_miTAR.py was used to identify the best parameters for miTAR. After obtaining the best parameters, eacho dataset was randomly split 30 times into training, validation and test datasets and the performance of each model was reported using the average accuracy with 95% confidence interval. evals_CNN_BiRNN_DeepMirTar_miTAR1_sampling.py, evals_CNN_BiRNN_miRAW_miTAR2_sampling.py, and evals_CNN_BiRNN_MirTarRAW_miTAR_sampling.py were used to perform 30 times' training, validation and test separately for each model. The three models with the best parameters were saved in results folder. They are miTAR1_CNN_BiRNN_b30_lr0.005_dout0.2.h5, miTAR2_CNN_BiRNN_b200_lr0.1_dout0.4.h5, and miTAR_CNN_BiRNN_b100_lr0.005_dout0.2.h5.


python/3.6.5 and Keras with TensorFlow as backend were used to implement the models and perform all the analysis.
A set of functions were developed to assist the analysis and all are in utils.py

To predict the targets for one miRNA, predict_onemiRmultimRNA.py can be used. An example is shown here:

    python predict_onemiRmultimRNA.py -i1 data/tests/mrna_multiple_mixedLongShort.fa -i2 data/tests/mirna_hsa-miR-139-5p.txt -o results/tests/mir139-5p_predictedTar.fa -p 0.8 -ns 1 -s 22 
The meaning of the parameters can be obtained by python predict_onemiRmultimRNA.py -h

To predict the targets for multiple miRNAs, predict_multimiRmultimRNA.py can be used. An example is shown here: 

    python predict_multimiRmultimRNA.py -i1 data/tests/mrna_multiple_mixedLongShort.fa -i2 data/tests/mirna_multiple.fa -o results/tests/mirna_multiple_predictedTar.fa -s 22 -p 0.8 -ns 1
The meaning of the parameters are the same as predict_onemiRmultimRNA.py and can be otained by python predict_multimiRmultimRNA.py -h

The model used in the predict_multimiRmultimRNA.py and predict_onemiRmultimRNA.py is miTAR. If you would like to use a different model, please replace the model with another one, and change the fragment length to match the model input.
   
