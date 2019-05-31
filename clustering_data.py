#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:14:22 2019

@author: silvia
"""
import numpy as np
import pandas as pd
from pathlib2 import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn import metrics

import time

def load_feature_vector(path_save_feat_vec):
    path_feat_vec=Path(path_save_feat_vec)/'feat_vector.npy'
    feat_vec = np.load(str(path_feat_vec))
    return feat_vec

def reduce_feat_vector(feat_vec, reducing_method_name, expl_var_percentage):
    print(expl_var_percentage)
    expl_var_percentage=float(expl_var_percentage)
    if reducing_method_name=='svd':
        reducing_method=reducing_method=TruncatedSVD(n_components=(feat_vec.shape[0]-1))
    elif reducing_method_name=='pca':
        reducing_method=PCA(n_components=(feat_vec.shape[0]-1))
    else:
        print('Error: method not yet added to the list')
    t0=time.time()
    reduced_vec = reducing_method.fit_transform(feat_vec)
    t1=time.time()
    print('time to apply ',reducing_method_name,' : ', t1-t0)
    print('')
    print('REDUCED FEATURES VECTOR SHAPE: ' ,reduced_vec.shape)
    variance = np.cumsum(reducing_method.explained_variance_ratio_)
    #print('Variance::::::: ', variance.shape)
    #print(variance[-1])
    #print('argwhere::::::', np.argwhere(variance >= expl_var_percentage))
    n_components = np.maximum(2, np.argwhere(variance >= expl_var_percentage)[0])
    reduced_vec = reduced_vec[:, 0:n_components[0]]
    print('')
    print("Explained variance of the SVD step: {}%".format(int(expl_var_percentage * 100)))
    print('')
    
    return reduced_vec, expl_var_percentage, n_components

def clustering_data(feat_vec, clustering_method_name, n_clust, labels):
    n_clust=int(n_clust)
    if clustering_method_name=='kmeans':
        clustering_method=KMeans(n_clusters=n_clust, init='k-means++', max_iter=100, n_init=1)
        
    clustering_method.fit(feat_vec)

    hom = metrics.homogeneity_score(labels, clustering_method.labels_)
    completeness = metrics.completeness_score(labels, clustering_method.labels_)
    v_meas = metrics.v_measure_score(labels, clustering_method.labels_)
    rand_index = metrics.adjusted_rand_score(labels, clustering_method.labels_)
    sil = metrics.silhouette_score(feat_vec, clustering_method.labels_, sample_size=1000)

    clustering_parameters=[hom, completeness, v_meas, rand_index, sil]
    
    print('CLUSTERING PARAMETERS: ')
    print('')
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, clustering_method.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, clustering_method.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, clustering_method.labels_))
    print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(labels, clustering_method.labels_))
    print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(feat_vec, clustering_method.labels_, sample_size=1000))

    return clustering_parameters

def save_data(saving_path, n_components, n_clust, reduced_vec, clustering_parameters):
    path_out=saving_path
    path_out_red_feat_vec=path_out/'reduced_feat_vec'
    path_out_clust_params=path_out/'clustering_parameters'
    if not path_out_red_feat_vec.is_dir():
        path_out_red_feat_vec.mkdir()
    if not path_out_clust_params.is_dir():
        path_out_clust_params.mkdir()
    np.save(str(Path(path_out_red_feat_vec)/'red_feat_vector_n_comp_{}.npy'.format(n_components)), reduced_vec)
    np.save(str(Path(path_out_clust_params)/'clustering_parameters_n_clust_{}.npy'.format(n_clust)), clustering_parameters)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='''
             Clustering texts to see if there are any important structure in the dataset
             '''.strip())

    ###########################################################################
    
    parser.add_argument('--path_train_data')
    parser.add_argument('--path_feat_vec')
    parser.add_argument('--saving_path')

    # Choose if reduce or not the features vectors
    parser.add_argument('--reduce_feat_vec', action='store_true',
                   default='true',
                   help="Store true to reduce the feature vectores before the clustering operation")
    parser.add_argument('--expl_variance_percentage')
    parser.add_argument('--reducing_method_name', default='svd')
    
    parser.add_argument('--save', action='store_true', default='true')
    
    # CLUSTERING DATA OPTIONS 
    
    clusters = parser.add_argument_group('Clusterization options')
    clusters.add_argument('--n_clust', default=None)
    clusters.add_argument('--clustering_method_name', default='k_means')
 
    args = parser.parse_args()

    ###########################################################################
    
    path_train=args.path_train_data
    path_save_feat_vec=args.path_feat_vec
    saving_path=args.saving_path
    reduce_feat_vec = args.reduce_feat_vec
    #expl_var_percentage=args.expl_variance_percentage
    reducing_method_name = args.reducing_method_name
    
    n_clust=args.n_clust 
    clustering_method_name = args.clustering_method_name
    
    ###########################################################################

    print('STARTING THE DATA ANALYSIS...')
    
    train_data=pd.read_csv(path_train,sep='\t', header=(0))
    train_category=train_data['misogyny_category'] 
    feat_vec=load_feature_vector(path_save_feat_vec)
    
    if reduce_feat_vec:
        t0 = time.time()
        feat_vec, explained_variance, n_components = reduce_feat_vector(feat_vec, reducing_method_name, expl_var_percentage=0.8)
        t1= time.time()
        print('')
        print('time to reduce feature vector: ', t1-t0)
    if n_clust==None:
        n_clust = np.unique(train_category).shape[0]    
    print('')
    t0=time.time()    
    clustering_parameters = clustering_data(feat_vec, clustering_method_name, n_clust, train_category)
    t1=time.time()
    print('time to do the clustering: ', t1-t0)
    if args.save:
        print('')
        print('SAVING..')
        t0=time.time()
        save_data(saving_path, n_components, n_clust, feat_vec, clustering_parameters)
        t1=time.time()
        print('')
        print('time to save data: ', t1-t0)
###############################################################################    
    
main()





















