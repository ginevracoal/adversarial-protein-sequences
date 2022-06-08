#!/bin/bash

DATASET="PF00533"
N_SEQUENCES=100
FILTER_SIZE=100

### set paths

IN_FILENAME="/scratch/external/gcarbone/msa/seqs${DATASET}"
OUT_PATH="/scratch/external/gcarbone/hhfiltered/hhfiltered_${DATASET}_seqs=${N_SEQUENCES}_filter=${FILTER_SIZE}/"

mkdir -p $OUT_PATH
module load conda/4.9.2
conda activate esm

### select top N_SEQUENCES in the MSA 

OUT_NAME="${DATASET}_top_${N_SEQUENCES}_seqs"

# sed -n 1,2p $IN_FILENAME

hhfilter -diff $N_SEQUENCES -i $IN_FILENAME -o $OUT_PATH$OUT_NAME

cat $OUT_PATH$OUT_NAME | awk 'NR % 2 == 1' > "${OUT_PATH}names"
cat $OUT_PATH$OUT_NAME | awk 'NR % 2 == 0' > "${OUT_PATH}full_sequences"


### for each sequence select columns without gaps and build a filtered MSA of minimum size FILTER_SIZE

seq_count=1
for current_seq in $(cat "${OUT_PATH}full_sequences"); do

	current_seq_name=$(sed -n ${seq_count}p "${OUT_PATH}names")
	OUT_NO_GAPS="${DATASET}_seq_idx=${seq_count}_no_gaps"
	cat "${OUT_PATH}full_sequences" > "${OUT_PATH}tmp"

	### remove all columns where current sequence has gaps 

	printf "\n\n"
	echo name: "$current_seq_name" 
	echo seq: "$current_seq"

	char_count=1
	for char in $(sed 's/./&\n/g' <(printf "$current_seq")); do

		if [ "$char" = "-" ]; then

			cat "${OUT_PATH}tmp" | cut --complement -c$char_count > "${OUT_PATH}swp"
			cat "${OUT_PATH}swp" > "${OUT_PATH}tmp"

			### DEBUG
			# printf "\n"
			# echo removing col $char_count with char $char
			# printf "\n"
			# cat "${OUT_PATH}tmp"

			char_count=$((char_count-1))
		fi

		char_count=$((char_count+1))

	done

	printf "\n"
	cat "${OUT_PATH}tmp"

	### switch back to fasta type dataset

	row_count=1

	for row_seq_name in $(cat "${OUT_PATH}names"); do

		echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS
		sed -n ${row_count}p "${OUT_PATH}tmp" >> $OUT_PATH$OUT_NO_GAPS

		row_count=$((row_count+1))

	done

	printf "\n"
	cat $OUT_PATH$OUT_NO_GAPS

	### select top FILTER_SIZE seqs in the new msa

	OUT_NO_GAPS_FILTERED="${DATASET}_seq_idx=${seq_count}_no_gaps_filter=${FILTER_SIZE}"

	hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED

	seq_count=$((seq_count+1))

	### delete temporary files

	rm "${OUT_PATH}tmp"
	rm "${OUT_PATH}swp"
	rm $OUT_PATH$OUT_NO_GAPS

done

rm "${OUT_PATH}names"
rm "${OUT_PATH}full_sequences"