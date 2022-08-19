#!/bin/bash

# 1. Filter the MSA from the chosen protein family and select N_SEQUENCES by minimum sequence identity.
# 2. For each selected sequence build and MSA of size FILTER_SIZE by minimum sequence identity.

### args

DATASET=$1 # PF00533, PF00627, PF00240
N_SEQUENCES=100
FILTER_SIZE=100

### set paths

MSA_PATH="/scratch/external/gcarbone/msa/"
FASTA="${MSA_PATH}fasta/${DATASET}.fasta"
FASTA_ONELINE="${MSA_PATH}fasta_oneline/oneline_${DATASET}.fasta"
OUT_PATH="${MSA_PATH}hhfiltered/hhfiltered_${DATASET}_filter=${FILTER_SIZE}/"

mkdir -p "${MSA_PATH}fasta_oneline/"
mkdir -p $OUT_PATH
eval "$(conda shell.bash hook)"
# module load conda/4.9.2
conda activate esm

### MSA to single line

cat $FASTA | awk 'BEGIN{FS=""}{if($1==">"){if(NR==1)print $0; else {printf "\n";print $0;}}else printf toupper($0)}' > $FASTA_ONELINE

### Select N_SEQUENCES in the MSA 

OUT_NAME="${DATASET}_top_${N_SEQUENCES}_seqs"

hhfilter -diff $N_SEQUENCES -i $FASTA_ONELINE -o $OUT_PATH$OUT_NAME

cat $OUT_PATH$OUT_NAME | awk 'NR % 2 == 1' > "${OUT_PATH}names"
cat $OUT_PATH$OUT_NAME | awk 'NR % 2 == 0' > "${OUT_PATH}full_sequences"

# head "${OUT_PATH}names"
# head "${OUT_PATH}full_sequences"

### For each sequence select columns without gaps and build a filtered MSA of minimum size FILTER_SIZE

seq_count=1
for current_seq in $(cat "${OUT_PATH}full_sequences"); do

	current_seq_name=$(sed -n ${seq_count}p "${OUT_PATH}names")
	current_seq_filename=$(echo $current_seq_name | sed 's|[>,]||g' | sed 's/\//_/g')

	OUT_NO_GAPS="${DATASET}_${current_seq_filename}_no_gaps"

	# cat "${OUT_PATH}full_sequences" > "${OUT_PATH}tmp" 
	cat $FASTA_ONELINE | awk 'NR % 2 == 1' > "${OUT_PATH}tmp_names"
	cat $FASTA_ONELINE | awk 'NR % 2 == 0' > "${OUT_PATH}tmp_msa"

	printf "\n"
	echo name: "$current_seq_name" 
	echo seq: "$current_seq"
	printf "\n"

	echo "Removing columns from the MSA where current sequence has gaps"

	char_count=1
	for char in $(sed 's/./&\n/g' <(printf '%s\n' "$current_seq")); do

		if [ "$char" = "-" ]; then

			cat "${OUT_PATH}tmp_msa" | cut --complement -c$char_count > "${OUT_PATH}swp_msa"
			cat "${OUT_PATH}swp_msa" > "${OUT_PATH}tmp_msa"

			### DEBUG
			printf "."
			# echo removing col $char_count with char $char

			char_count=$((char_count-1))
		fi

		char_count=$((char_count+1))

	done
	
	# printf "\n"
	# cat "${OUT_PATH}tmp_msa"

	echo "Building new fasta file for current sequence"

	row_count=1

	for row_seq_name in $(cat "${OUT_PATH}tmp_names"); do

		echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS
		sed -n ${row_count}p "${OUT_PATH}tmp_msa" >> $OUT_PATH$OUT_NO_GAPS

		row_count=$((row_count+1))

	done

	echo ""
	head $OUT_PATH$OUT_NO_GAPS

	### select top FILTER_SIZE seqs in the new msa

	echo "Filtering the new MSA"

	OUT_NO_GAPS_FILTERED="${DATASET}_${current_seq_filename}_no_gaps_filter=${FILTER_SIZE}"

	hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED

	seq_count=$((seq_count+1))

	### delete temporary files

	rm "${OUT_PATH}tmp_names"
	rm "${OUT_PATH}tmp_msa"
	rm "${OUT_PATH}swp_msa"
	rm $OUT_PATH$OUT_NO_GAPS

done

rm "${OUT_PATH}names"
rm "${OUT_PATH}full_sequences"