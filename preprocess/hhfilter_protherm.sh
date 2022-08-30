#!/bin/bash

### args

FILTER_SIZE=30

### set paths

FILEPATH="/scratch/external/gcarbone/"
MSA_PATH="${FILEPATH}msa/"
PROCESSED_PROTHERM="${FILEPATH}ProTherm/processed_single_mutation.csv"
PROTHERM_SEQUENCES="${FILEPATH}ProTherm/sequences_single_mutation.csv"
OUT_LOGS="/fast/external/gcarbone/adversarial-protein-sequences_out/logs/"

mkdir -p "${MSA_PATH}fasta_oneline/"
mkdir -p $OUT_LOGS
eval "$(conda shell.bash hook)"
conda activate esm


printf "\n=== Filter MSA for ProTherm sequences ===\n\n"

sed 1d $PROTHERM_SEQUENCES | while read -r line; do

   pfam_id=$(echo $line | awk -F ";" '{print $1}')
   fasta_name=$(echo $line | awk -F ";" '{print $3}')
   pfam_sequence=$(echo $line | awk -F ";" '{print $5}')

   printf "\n"
   echo name: "$fasta_name" 
   echo seq: "$pfam_sequence"
   printf "\n"

   OUT_PATH="${MSA_PATH}hhfiltered/hhfiltered_${pfam_id}_filter=${FILTER_SIZE}/"
   current_seq_filename=$(echo $fasta_name | sed 's|[>,]||g' | sed 's/\//_/g')
   # OUT_NO_GAPS="${pfam_id}_${current_seq_filename}_no_gaps"
   OUT_NO_GAPS_FILTERED="${pfam_id}_${current_seq_filename}_no_gaps_filter=${FILTER_SIZE}"
   mkdir -p $OUT_PATH

   printf "\nSelect top ${FILTER_SIZE} seqs in ${pfam_id}"

   FASTA_ONELINE="${MSA_PATH}fasta_oneline/oneline_${pfam_id}.fasta"
   FASTA_ONELINE_FILTERED="${MSA_PATH}fasta_oneline/oneline_${pfam_id}_filter=${FILTER_SIZE}.fasta"
   hhfilter -diff $FILTER_SIZE -i $FASTA_ONELINE -o $FASTA_ONELINE_FILTERED

   echo
   head -n 2 $FASTA_ONELINE_FILTERED

   printf "\nRemove columns from the MSA where sequence ${fasta_name} has gaps"

   cat $FASTA_ONELINE_FILTERED | awk 'NR % 2 == 1' > "${OUT_PATH}tmp_names"
   cat $FASTA_ONELINE_FILTERED | awk 'NR % 2 == 0' > "${OUT_PATH}tmp_msa"

   char_count=1
   for char in $(sed 's/./&\n/g' <(printf '%s\n' "$pfam_sequence")); do

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
   # head "${OUT_PATH}tmp_names"
   # head "${OUT_PATH}tmp_msa"

   echo "Building new fasta file for current sequence"

   row_count=1

   for row_seq_name in $(cat "${OUT_PATH}tmp_names"); do

      echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS_FILTERED
      sed -n ${row_count}p "${OUT_PATH}tmp_msa" >> $OUT_PATH$OUT_NO_GAPS_FILTERED

      row_count=$((row_count+1))

   done

   echo
   head -n 2 $OUT_PATH$OUT_NO_GAPS_FILTERED

   # hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED
   # head $OUT_PATH$OUT_NO_GAPS_FILTERED 

   ### delete temporary files

   rm "${OUT_PATH}tmp_names"
   rm "${OUT_PATH}tmp_msa"
   rm "${OUT_PATH}swp_msa"
   # rm $OUT_PATH$OUT_NO_GAPS

done 