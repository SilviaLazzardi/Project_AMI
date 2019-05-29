#!/usr/bin/env bash

PATH_TRAIN=/home/silvia/Desktop/TRAINEESHIP/Project_python/AMI_data/new/it_training.tsv
PATH_TEST=/home/silvia/Desktop/TRAINEESHIP/Project_python/AMI_data/new/it_testing.tsv

date='29_05_2019'
mkdir results
mkdir results/${date}

PATH_OUT=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/${date}
echo path_out=${PATH_OUT}

vectorizer_name='tf_idf'
n_comp=1000
reducing_method_name='svd'
n_clust=6
clustering_method_name='kmeans'


python -u project.py \
          --path_train=${PATH_TRAIN} \
	      --path_test=${PATH_TEST} \
	      --outfolder=${PATH_OUT} \
          --save \
          --date=${date} \
          --vectorizer_name=${vectorizer_name} \
          --reduce_feat_vec \
          --n_comp=${n_comp} --reducing_method_name=${reducing_method_name} \
          --n_clust=${n_clust}  --clustering_method_name=${clustering_method_name} \
