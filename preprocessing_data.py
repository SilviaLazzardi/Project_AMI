#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:31:14 2019

@author: silvia
"""
import numpy as np
import pandas as pd

from pathlib2 import Path

from sklearn.feature_extraction.text import TfidfVectorizer

import time

'''

This script has the aim to extract the feature vector from the training text data.
It uses tf-idf.
If you want to add onother method please add it to the function "feature_extraction"


'''

################################## LOAD DATA ##################################

def read_dataset(path_in_training, path_in_testing):
    train_data = pd.read_csv(path_in_training,sep='\t', header=(0))
    test_data = pd.read_csv(path_in_testing,sep='\t', header=(0))
    return train_data, test_data

############################## TRAINING VARIABLES #############################

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

############################# FEATURE  EXTRACTION #############################

def feature_extraction(train_text, vectorizer_name):
    if vectorizer_name=='tf_idf':
        vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
    else:
        print("This method not exists yet to the list. Please add it to the function 'feature_extraction' in preprocessing_data.py.")
    feat_vec = vectorizer.fit_transform(train_text).toarray()
    
    return feat_vec
 
############################# SAVE FEATURE VECTOR #############################

def save_data(outfolder, feat_vec):
    np.save(str(Path(outfolder)/'feat_vector.npy'), feat_vec)
    

###############################################################################
#                                   MAIN 
###############################################################################
    
def main():
    import argparse
    parser = argparse.ArgumentParser()

    ###########################################################################

    # System parameters:

    parser.add_argument('--path_train')
    parser.add_argument('--path_test')
    parser.add_argument('--save', action='store_true', default='true')
    parser.add_argument('--outfolder_feat_vec',
                        help="Where to store the .npy file of results.")
    parser.add_argument('--vectorizer_name', default='tf_idf')
 
    args = parser.parse_args()

    ###########################################################################

    path_in_training=args.path_train
    path_in_testing=args.path_test
    outfolder=args.outfolder_feat_vec
    save = args.save    
    vectorizer_name = args.vectorizer_name
   
    ###########################################################################
    
    train_data, test_data = read_dataset(path_in_training, path_in_testing)
    
    train_id, train_text, train_misogynous, train_category, train_target = train_var(train_data)
    test_id, test_text = test_var(test_data)
    
    ########################################################################### 
    
    print('DATA DETAILS:')
    print('')
    print('Training data shape: ', train_data.shape)
    print('')
    print('Testing data shape: ', test_data.shape)
    print('')
    print("Input data:  'id' and 'text'")
    print('')
    print("Output: 'misogynous','misogynous_category', 'target'")
    print('')
    print("'misogynous' -> 0 if the text is cathegorized as NOT-misogynous, 1 otherwise")
    print('')
    print("'misogynous_category' could be: ")
    print(np.unique(train_category))
    print('')
    
    ###########################################################################    

    print('Method to covert the text into feature vectors: ', vectorizer_name)
    t0=time.time()
    feat_vec = feature_extraction(train_text, vectorizer_name)
    t1=time.time()
    elapsed_time=t1-t0
    print('')
    print('Time to extract features: ', elapsed_time)
    if save:
        print('')
        t0=time.time()
        print('SAVING FEATURE VECTOR..')
        save_data(outfolder, feat_vec)
        t1=time.time()
        print('Time to save: ', t1-t0)
 
###############################################################################
#                               MAIN EXECUTING
###############################################################################
    
main()


