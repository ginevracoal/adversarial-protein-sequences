#!/bin/bash

TARGET_ATTENTION='last_layer' # last_layer, all_layers
ALIGN=True
MAX_TOKENS=200
N_SEQUENCES=500
DEVICE="cuda"
LOAD=False

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="../out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


for DATASET in "fastaPF00001" "fastaPF00004" 
do
	for N_SUBSTITUTIONS in 3 10 #20
	do
		python pfam_test.py --dataset=$DATASET --align=$ALIGN --max_tokens=$MAX_TOKENS --n_sequences=$N_SEQUENCES \
							--n_substitutions=$N_SUBSTITUTIONS --device=$DEVICE --load=$LOAD \
							--target_attention=$TARGET_ATTENTION >> $OUT 2>&1

	done
done

deactivate

