#!/bin/sh

################
# PBS settings #
################

#PBS -N ProTherm_attack
#PBS -l nodes=1:ppn=12
#PBS -l walltime=30:00:00
#PBS -q gpu
#PBS -j oe
#PBS -o pbs.out

###################
# script settings #
###################

MIN_FILTER=30
MODEL="ESM_MSA" # "ESM", "ESM_MSA"
TOKEN_SELECTION="max_attention" # 'min_entropy' 'max_entropy'
TARGET_ATTENTION='last_layer' # 'all_layers' 'last_layer'
LOSS_METHOD="max_masked_prob"
DEVICE="cuda"
LOAD=False

###########
# modules #
###########

eval "$(conda shell.bash hook)"
conda activate esm 

###############
# directories #
###############

WORKING_DIR="/u/external/gcarbone/adversarial-protein-sequences/src"
OUT_DIR="/fast/external/gcarbone/adversarial-protein-sequences_out/"
DATA_DIR="/scratch/external/gcarbone/"
	
cd $WORKING_DIR
PBS_O_WORKDIR=$(pwd)
LOGS_DIR="${OUT_DIR}logs/"
mkdir -p $LOGS_DIR
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
OUT="${LOGS_DIR}${DATE}_${TIME}_out.txt"

#######
# run #
#######

python attack_protherm.py --data_dir=$DATA_DIR --out_dir=$OUT_DIR --model=$MODEL --loss_method=$LOSS_METHOD \
	--token_selection=$TOKEN_SELECTION --device=$DEVICE --load=$LOAD \
	--target_attention=$TARGET_ATTENTION --min_filter=$MIN_FILTER >> $OUT 2>&1

conda deactivate

