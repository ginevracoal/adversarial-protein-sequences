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

###############
# directories #
###############

DATA_DIR="../data"
OUT_DIR="../out"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="${OUT_DIR}/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"

###########
# modules #
###########

source ../venv/bin/activate

#######
# run #
#######

for DATASET in "fastaPF00001" "fastaPF00004" 
do
	for N_SUBSTITUTIONS in 3 10 20
	do
		python pfam_test.py  --data_dir=$DATA_DIR --out_dir=$OUT_DIR \
			--dataset=$DATASET --align=$ALIGN --max_tokens=$MAX_TOKENS --n_sequences=$N_SEQUENCES \
			--n_substitutions=$N_SUBSTITUTIONS --device=$DEVICE --load=$LOAD \
			--target_attention=$TARGET_ATTENTION >> $OUT 2>&1

	done
done

deactivate

