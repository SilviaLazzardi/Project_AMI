# Traineeship_Project_AMI
Traineeship code repository

#TODO: Project description

#Prerequisites
Python libraries:
* numpy
* pathlib2
* pandas
* sklearn
* time

#Python script
-project.py

#At the moment this file is composed by a a list of functions:
* read_dataset -> inputs: paths of the training and the testing datasets. 
                 returns: the train and test data loaded with pandas 
* train_var -> input: train_data
              returns: separated data (id, text,...)
* test_var -> input: test_data
              returns: separated data (id, text) 
* define_methods -> inputs: feature_extraction_method_name* = type(str), method used to convert text into feature vector
                           reducing_method_name** = type(str), method to reduce the feature vector (could be also 'None')
                           clustering_method_name*** = type(str), method to cluster data
                           n_comp = type(int), number of components of the reduced feature vector
                           n_clust = type(int), number of clusters
                   returns: the methods
* data_processing -> returns: feat_vec, reduced_vec, explained_variance
                             clustering_parameters=[hom, completeness, v_meas, rand_index, sil]
                             (homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, silhouette_score)
* save_data -> returns: 3 results files .npy in a new directory: ".\results\date\usedmethods\"
                       -'feature_vector.npy'
                       -'red_feat_vector_n_comp_{}.npy'.format(n_comp)
                       -'clustering_parameters_n_clust.npy'.format(n_clust)

*   *Now: tf-idf. If you want you can easly add more or change parameters
*  **Now: svd and pca
*  ***Now: k-means

#Extension to do 
* setting a limit-parameter on the explained_variance and choose the number of components with it
* automatic run of the code for a range of explained_variance values
* saving file txt or excel with ordered results with the parameters changment

#Bash file
* main.sh: bash file to run project.py

#Code instructions:
* download the repository and open main.sh
* set the current date and the parameters you want to try
* from your shell go to the directory containing the files:
* first time: chmod u+x main.sh
* then: ./main.sh
*see the results 
