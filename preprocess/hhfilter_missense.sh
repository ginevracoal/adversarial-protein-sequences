#!/bin/bash

### args

DATASET="PF00533"
FILTER_SIZE=100

### set paths

IN_FILENAME="/scratch/external/gcarbone/msa/seqs${DATASET}"
OUT_PATH="/scratch/external/gcarbone/missense/hhfiltered/hhfiltered_${DATASET}_filter=${FILTER_SIZE}/"

mkdir -p $OUT_PATH
eval "$(conda shell.bash hook)"
# module load conda/4.9.2
conda activate esm

### split names and sequences

cat $IN_FILENAME | awk 'NR % 2 == 1' > "${OUT_PATH}names"
cat $IN_FILENAME | awk 'NR % 2 == 0' > "${OUT_PATH}full_sequences"

######################################
# First sequence in missense dataset #
######################################

### select columns without gaps and build a filtered MSA of minimum size FILTER_SIZE

sequence="TPEEFMLVYKFARKHHITLTNLITEETTHVVMKTDAEFVCERTLKYFLGIAGGKWVVSYFWVTQSI"
sequence_filename="P38398_1658-1723"
OUT_NO_GAPS="${DATASET}_${sequence_filename}_no_gaps"
cat "${OUT_PATH}full_sequences" > "${OUT_PATH}tmp"

### remove all columns where the sequence has gaps 

char_count=1
for char in $(sed 's/./&\n/g' <(printf "$sequence")); do
	
	if [ "$char" = "-" ]; then

		cat "${OUT_PATH}tmp" | cut --complement -c$char_count > "${OUT_PATH}swp"
		cat "${OUT_PATH}swp" > "${OUT_PATH}tmp"
		char_count=$((char_count-1))
	fi

	char_count=$((char_count+1))

done

# printf "\n"
# cat "${OUT_PATH}tmp"

### switch back to fasta type

row_count=1

for row_seq_name in $(cat "${OUT_PATH}names"); do

	echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS
	sed -n ${row_count}p "${OUT_PATH}tmp" >> $OUT_PATH$OUT_NO_GAPS

	row_count=$((row_count+1))

done

# printf "\n"
# cat $OUT_PATH$OUT_NO_GAPS

### select top FILTER_SIZE seqs in the new msa

OUT_NO_GAPS_FILTERED="${DATASET}_${sequence_filename}_no_gaps_filter=${FILTER_SIZE}"

hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED

seq_count=$((seq_count+1))

### delete temporary files

rm "${OUT_PATH}tmp"
rm "${OUT_PATH}swp"
rm $OUT_PATH$OUT_NO_GAPS

#######################################
# Second sequence in missense dataset #
#######################################

sequence="PTDQL---EWMVQLCGASVSFTLGTGVHPIVVVQPD-AWTEDNGFHAIGQMCEAPVVTREWVLDS-"
sequence_filename="P38398_1757-1841"
OUT_NO_GAPS="${DATASET}_${sequence_filename}_no_gaps"
cat "${OUT_PATH}full_sequences" > "${OUT_PATH}tmp"

### remove all columns where the sequence has gaps 

char_count=1
for char in $(sed 's/./&\n/g' <(printf "$sequence")); do

	if [ "$char" = "-" ]; then

		cat "${OUT_PATH}tmp" | cut --complement -c$char_count > "${OUT_PATH}swp"
		cat "${OUT_PATH}swp" > "${OUT_PATH}tmp"
		char_count=$((char_count-1))
	fi

	char_count=$((char_count+1))

done

### switch back to fasta type

row_count=1

for row_seq_name in $(cat "${OUT_PATH}names"); do

	echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS
	sed -n ${row_count}p "${OUT_PATH}tmp" >> $OUT_PATH$OUT_NO_GAPS

	row_count=$((row_count+1))

done

### select top FILTER_SIZE seqs in the new msa

OUT_NO_GAPS_FILTERED="${DATASET}_${sequence_filename}_no_gaps_filter=${FILTER_SIZE}"

hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED

seq_count=$((seq_count+1))

### delete temporary files

rm "${OUT_PATH}tmp"
rm "${OUT_PATH}swp"
rm $OUT_PATH$OUT_NO_GAPS

rm "${OUT_PATH}names"
rm "${OUT_PATH}full_sequences"