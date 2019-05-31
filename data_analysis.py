#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:02:53 2019

@author: silvia
"""
import numpy as np
from pathlib2 import Path
import xlsxwriter
import os


import matplotlib.pyplot as plt
#import scipy as sp

from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette

'''

The aim of this script is to analize the data about the feature vector clustering. 


'''
 
def create_excel_file(path_save_clustering_data, path_clust_experiments):
    path_clust_experiments=Path(path_clust_experiments)
    path_save=Path(path_save_clustering_data)
    path_save = Path(path_save)/'union_results.xlsx'
    j = 0
    workbook  = xlsxwriter.Workbook(str(path_save))
    ws = workbook.add_worksheet()
    parameters_list=['homogeneity_score', 'completeness_score', 'v_measure_score', 'adjusted_rand_score', 'silhouette_score']
    for k in range(len(parameters_list)):
        ws.write(0, k, parameters_list[k])
    listdir = os.listdir(str(path_clust_experiments))
    print('listdir 1 ', listdir)
    for folder in listdir:
        path_in = path_clust_experiments/folder
        listdir = os.listdir(str(path_in))
        listdir.remove('feat_vector.npy')
        print('listdir 2 ', listdir)
        for folder in listdir:
            path_in_1 = path_in/folder
            listdir = os.listdir(str(path_in_1))
            listdir.remove('reduced_feat_vec')
            print('listdir 3 ', listdir)
            for folder in listdir:
                path_in_2 = path_in_1/'clustering_parameters'
                listdir = os.listdir(str(path_in_2))
                print('listdir 4 ', listdir)
                for file in listdir:
                    path_in_3=path_in_2/file
                    j = j+1
                    data = np.load(str(path_in_3))
                    for i in range(len(data)):
                        print(parameters_list[i],' of file number ',j,' : ', data[i])
                        ws.write(j, i, data[i])
                workbook.close()
    
def create_dendogram(vectorizer_name, reducing_method_name, clustering_method_name, path_save_dendogram, feat_vec, linkage_method='ward', metric='euclidean', truncate_mode='level', p=10, show=True):
    path_save_dendogram=Path(path_save_dendogram)  
    Z = linkage(feat_vec, linkage_method, metric=metric)
    
    fig = plt.figure(figsize = (10,10), dpi=300)
    set_link_color_palette(['m', 'c', 'y', 'k'])
    dn = dendrogram(Z, p, show_leaf_counts=True, above_threshold_color='y')
    title = '{}_{}{}dendogram_{}_linkmethod_{}_truncatemode_{}_p.png'.format(vectorizer_name, reducing_method_name, clustering_method_name, linkage_method, truncate_mode,p)
    path_fig=path_save_dendogram/title
    fig.savefig(str(path_fig))
    if show:
        plt.show()
    plt.close()
    
    return dn, Z    


def main():
    import argparse
    parser = argparse.ArgumentParser()

    ###########################################################################
    
    #parser.add_argument('--path_train_data')
    parser.add_argument('--vectorizer_name', default='tf_idf')
    parser.add_argument('--reducing_method_name', default='svd')
    parser.add_argument('--clustering_method_name', default='kmeans')
    parser.add_argument('--path_feat_vec')
    parser.add_argument('--path_clust_experiments')
    parser.add_argument('--path_save_dendogram')
    parser.add_argument('--path_save_clustering_data')
    parser.add_argument('--path_analysis')
 
    args = parser.parse_args()

    ###########################################################################
    
    vectorizer_name=args.vectorizer_name
    reducing_method_name=args.reducing_method_name
    clustering_method_name=args.clustering_method_name
    path_save_feat_vec=args.path_feat_vec
    path_clust_experiments=args.path_clust_experiments
    path_save_dendogram=args.path_save_dendogram
    path_save_clustering_data=args.path_save_clustering_data
    
    
    
    feat_vec=Path(path_save_feat_vec)/'feat_vector.npy'
    feat_vec=np.load(str(feat_vec))
    create_excel_file(path_save_clustering_data, path_clust_experiments)
    create_dendogram(vectorizer_name, reducing_method_name, clustering_method_name, path_save_dendogram, feat_vec, linkage_method='ward', metric='euclidean', truncate_mode='level', p=10, show=False)
    
main()    





