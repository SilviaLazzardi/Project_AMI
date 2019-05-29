#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:00:50 2019

@author: silvia
"""
from pathlib2 import Path
import pandas as pd

import numpy as np
#import scipy as sp

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn import metrics


import time


def read_dataset(path_in_training, path_in_testing):
    train_data = pd.read_csv(path_in_training,sep='\t', header=(0))
    test_data = pd.read_csv(path_in_testing,sep='\t', header=(0))
    return train_data, test_data

############################### TRAINING VARIABLES ############################

def train_var(train_data):
    # X TRAIN
    train_id = train_data['id']
    train_text = train_data['text']
    
    # Y TRAIN
    train_misogynous = train_data['misogynous']
    train_category = train_data['misogyny_category'] 
    train_target = train_data['target']
    
    return train_id, train_text, train_misogynous, train_category, train_target

############################### TESTING  VARIABLES ############################

def test_var(test_data):
    # X TEST
    test_id = test_data['id']
    test_text = test_data['text']
    
    # Y TEEST unknown yet
    
    return test_id, test_text

###############################      CHECK      ###############################

def check_var(trainortest, train_data, test_data, train_misogynous, train_category, train_target, test_id, test_text):
    # to check if the data was correctly loaded just use this function with the 
    # argument 'train' to check the training dataset, 'test' for the testing one
    if trainortest == 'train':
        print("CHECK LOADED DATA:")
        print("Training set:")
        print(train_data[:5])
        print('misogynous:: ')
        print(train_misogynous[:5])
        print('category:: ')
        print(train_category[:5])
        print('target:: ')
        print(train_target[:5])
    else:
        print("Testing set:")
        print(test_data[:5])
        print('id:: ', test_id[:5])
        print('text:: ', test_text[:5])


############################## DEFINING  METHODS ##############################

def define_methods(feature_extraction_method_name, reducing_method_name, clustering_method_name, n_comp, n_clust):
    n_comp=int(n_comp)
    n_clust=int(n_clust)
    if feature_extraction_method_name=='tf_idf':
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
    else:
        print("This method not exists yet to the list. Please add it to the function 'define_methods' in project.py.")
    
    if reducing_method_name=='svd':
        reducing_method=TruncatedSVD(int(n_comp))
        
    if reducing_method_name=='pca':
        reducing_method=PCA(int(n_comp))
        
    else:
       print("This method not exists yet to the list. Please add it to the function 'define_methods' in project.py.") 
    
    if clustering_method_name=='kmeans':
       clustering_method=KMeans(n_clusters=n_clust, init='k-means++', max_iter=100, n_init=1)
        
    else:
       print("This method not exists yet to the list. Please add it to the function 'define_methods' in project.py.") 
    print('')
    print('Choosen methods of analysis:')
    print(feature_extraction_method_name, reducing_method_name, clustering_method_name)
    print('n_components: ', n_comp)
    print('n_clusters: ', n_clust)
    return vectorizer, reducing_method, clustering_method

################################ DATA ANALYSIS ################################

def data_processing(train_text, labels, vectorizer, reducing_method, clustering_method):
    feat_vec = vectorizer.fit_transform(train_text).toarray() # features shape: 4000x11267
    print('SHAPE FEATURES VECTOR: ', feat_vec.shape)
    reduced_vec = reducing_method.fit_transform(feat_vec)
    print('')
    print('REDUCED FEATURES VECTOR SHAPE: ' ,reduced_vec.shape)
    explained_variance = reducing_method.explained_variance_ratio_.sum()
    print('')
    print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
    print('')
    
    clustering_method.fit(reduced_vec)

    hom = metrics.homogeneity_score(labels, clustering_method.labels_)
    completeness = metrics.completeness_score(labels, clustering_method.labels_)
    v_meas = metrics.v_measure_score(labels, clustering_method.labels_)
    rand_index = metrics.adjusted_rand_score(labels, clustering_method.labels_)
    sil = metrics.silhouette_score(reduced_vec, clustering_method.labels_, sample_size=1000)

    clustering_parameters=[hom, completeness, v_meas, rand_index, sil]
    
    print('CLUSTERING PARAMETERS: ')
    print('')
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, clustering_method.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, clustering_method.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, clustering_method.labels_))
    print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, clustering_method.labels_))
    print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(reduced_vec, clustering_method.labels_, sample_size=1000))

    return feat_vec, reduced_vec, explained_variance, clustering_parameters

################################## SAVE DATA ##################################

def save_data(outfolder, reducing_method, clustering_method, n_components, n_clust, feat_vec, reduced_vec, clustering_parameters):
    path_out=Path(outfolder)/'{}_{}'.format(reducing_method, clustering_method)
    if not path_out.is_dir():
        path_out.mkdir()
    np.save(str(Path(path_out)/'feat_vector.npy'), feat_vec)
    np.save(str(Path(path_out)/'red_feat_vector_n_comp_{}.npy'.format(n_components)), reduced_vec)
    np.save(str(Path(path_out)/'clustering_parameters_n_clust_{}.npy'.format(n_clust)), clustering_parameters)

###############################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
             Clustering texts to see if there are any important structure in the dataset
             '''.strip())

    # System parameters:

    parser.add_argument('--path_train')
    parser.add_argument('--path_test')
    parser.add_argument('--save', action='store_true', default='true')
    parser.add_argument('--date')
    parser.add_argument('--outfolder',
                        help="Where to store the .txt file of results.")

    # PREPROCESSING DATA OPTIONS

    preprocessing = parser.add_argument_group('Preprocessing options')
    
    # Choose the method to extract features 
    preprocessing.add_argument('--vectorizer_name', default='tf_idf')
    # Choose if reduce or not the features vectors
    g = preprocessing.add_mutually_exclusive_group()
    g.add_argument('--reduce_feat_vec', action='store_true',
                   default='true',
                   help="Store true to reduce the feature vectores before the clustering operation")
    g.add_argument('--not_reduce_feat_vec', action='store_true',
                   default='false',
                   help="Store true to not reduce the feature vectores before the clustering operation")
    preprocessing.add_argument('--n_components')
    # Choose the method to reduce the feature vector
    preprocessing.add_argument('--reducing_method_name', default='svd')
    
    # CLUSTERING DATA OPTIONS 
    
    # Choose the Clustering method and the number of clusters
    clusters = parser.add_argument_group('Clusterization options')
    clusters.add_argument('--n_clust', default=None)
    clusters.add_argument('--clustering_method_name', default='k_means')
 
    args = parser.parse_args()

    path_in_training=args.path_train
    path_in_testing=args.path_test
    outfolder=args.outfolder
    save = args.save
    
    vectorizer_name = args.vectorizer_name
    
    reduce_feat_vec = args.reduce_feat_vec
    n_comp=args.n_components
    reducing_method_name = args.reducing_method_name
    
    n_clust=args.n_clust 
    clustering_method_name = args.clustering_method_name
    
    train_data, test_data = read_dataset(path_in_training, path_in_testing)
    
    train_id, train_text, train_misogynous, train_category, train_target = train_var(train_data)
    test_id, test_text = test_var(test_data)
    
    print('DATA DETAILS:')
    print('')
    print('Training data shape: ', train_data.shape)
    print('')
    print('Testing data shape: ', test_data.shape)
    print('')
    print("The input data are the 'id' and the 'text'")
    print("The output indeed are the 'misogynous', the 'misogynous_category' and the 'target'")
    print("'misogynous' -> 0 if the text is cathegorized as NOT-misogynous, 1 otherwise")
    print("'misogynous_category' could be: ", np.unique(train_category))
    print('')
    print('STARTING THE DATA ANALYSIS.')
    t0 = time.time()
    print('')
    print('Method to covert the text into feature vectors: ', vectorizer_name)
    if n_clust==None:
        n_clust = np.unique(train_category).shape[0]
    if reduce_feat_vec:
        vectorizer, reducing_method, clustering_method = define_methods(vectorizer_name, reducing_method_name, clustering_method_name, n_comp, n_clust)
        feat_vec, reduced_vec, explained_variance, clustering_parameters = data_processing(train_text, train_category, vectorizer, reducing_method, clustering_method)
        if save:
            print('')
            print('SAVING..')
            save_data(outfolder, reducing_method_name, clustering_method_name, n_comp, n_clust, feat_vec, reduced_vec, clustering_parameters)
            print('')
            t1 = time.time()
            print('Elapsed time: ', t1-t0)
    else:
        print('To be continued')
    
main()




