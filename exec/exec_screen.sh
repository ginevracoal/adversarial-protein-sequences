#!/bin/bash

###################
# script settings #
###################

TARGET_ATTENTION='last_layer' # last_layer, all_layers
ALIGN=True
MAX_TOKENS=500
N_SEQUENCES=None
DEVICE="cuda"
LOAD=False
DATASET="fastaPF00001" # fastaPF00001, fastaPF00004

###############
# directories #
###############

WORKING_DIR="/u/external/gcarbone/adversarial-protein-sequences/src"
OUT_DIR="/fast/external/gcarbone/adversarial-protein-sequences_out/"
DATA_DIR="/scratch/external/gcarbone/msa/"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS_DIR="${OUT_DIR}logs/"
mkdir -p $LOGS_DIR
OUT="${LOGS_DIR}${DATE}_${TIME}_out.txt"

###########
# modules #
###########

source ../venv/bin/activate

#######
# run #
#######

for N_SUBSTITUTIONS in 3 10
do
	python pfam_test.py  --data_dir=$DATA_DIR --out_dir=$OUT_DIR \
		--dataset=$DATASET --align=$ALIGN --max_tokens=$MAX_TOKENS --n_sequences=$N_SEQUENCES \
		--n_substitutions=$N_SUBSTITUTIONS --device=$DEVICE --load=$LOAD \
		--target_attention=$TARGET_ATTENTION >> $OUT 2>&1

done

deactivate

