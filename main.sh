#!/usr/bin/env bash

vectorizer_name='tf_idf'
expl_var_percentage=0.8
reducing_method_name='svd'
n_clust=6
clustering_method_name='kmeans'

# ANALYSIS FOLDER CREATION
mkdir results

# CLUSTERING EXPERIMENT FOLDER CREATION
mkdir results/experiments
mkdir results/experiments/clustering
mkdir results/experiments/clustering/${vectorizer_name}
mkdir results/experiments/clustering/${vectorizer_name}/${reducing_method_name}_${clustering_method_name}
mkdir results/experiments/clustering/${vectorizer_name}/${reducing_method_name}_${clustering_method_name}/reduced_feat_vec
mkdir results/experiments/clustering/${vectorizer_name}/${reducing_method_name}_${clustering_method_name}/clustering_parameters

mkdir results/analysis
mkdir results/analysis/clustering
mkdir results/analysis/dendograms

PATH_TRAIN=/home/silvia/Desktop/TRAINEESHIP/Project_python/AMI_data/new/it_training.tsv
PATH_TEST=/home/silvia/Desktop/TRAINEESHIP/Project_python/AMI_data/new/it_testing.tsv

PATH_OUT_EXPERIMENTS=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/experiments/clustering/${vectorizer_name}/${reducing_method_name}_${clustering_method_name}
PATH_OUT_ANALYSIS=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/analysis
PATH_SAVE_FEAT_VEC=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/experiments/clustering/${vectorizer_name}

PATH_SAVE_DENDOGRAM=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/analysis/dendograms
PATH_SAVE_CLUSTERING_DATA=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/analysis/clustering

PATH_CLUST_EXP=/home/silvia/Desktop/TRAINEESHIP/Project_python/results/experiments/clustering

echo path_out_experiments=${PATH_OUT_EXPERIMENTS}
echo path_out_analysis=${PATH_OUT_ANALYSIS}


#python -u preprocessing_data.py \
#         --path_train=${PATH_TRAIN} \
#	      --path_test=${PATH_TEST} \
#	      --outfolder_feat_vec=${PATH_SAVE_FEAT_VEC} \
#	      --save \
#	      --vectorizer_name=${vectorizer_name}


#python -u clustering_data.py \
#          --path_train=${PATH_TRAIN} \
#          --path_feat_vec=${PATH_SAVE_FEAT_VEC}\
#	       --saving_path=${PATH_OUT_EXPERIMENTS} \
#	       --reduce_feat_vec \
#          --save \
#          --expl_variance_percentage={expl_var_percentage} --reducing_method_name=${reducing_method_name} \
#          --n_clust=${n_clust}  --clustering_method_name=${clustering_method_name} \

    
python -u data_analysis.py \
          --vectorizer_name=${vectorizer_name} --reducing_method_name=${reducing_method_name} --clustering_method_name=${clustering_method_name} \
          --path_feat_vec=${PATH_SAVE_FEAT_VEC}\
	      --path_save_dendogram=${PATH_SAVE_DENDOGRAM} \
	      --path_save_clustering_data=${PATH_SAVE_CLUSTERING_DATA} \
          --path_clust_experiments=${PATH_CLUST_EXP}
