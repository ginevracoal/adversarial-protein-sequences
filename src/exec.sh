#!/bin/bash

MAX_TOKENS=120
N_SEQUENCES=500
N_SUBSTITUTIONS=3
DEVICE="cuda"

source ../venv/bin/activate

DATE=$(date +%Y-%m-%d)
TIME=$(date +%H:%M:%S)
LOGS="data/out/logs/"
mkdir -p $LOGS
OUT="${LOGS}${DATE}_${TIME}_out.txt"


for DATASET in "fastaPF00001" "fastaPF00004"
do
	python pfam_test.py --dataset=$DATASET --max_tokens=$MAX_TOKENS --n_sequences=$N_SEQUENCES \
						--n_substitutions=$N_SUBSTITUTIONS --device=$DEVICE >> $OUT

done

deactivate

