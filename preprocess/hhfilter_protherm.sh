#!/bin/bash

### args

FILTER_SIZE=100

### set paths

FILEPATH="/scratch/external/gcarbone/"
MSA_PATH="${FILEPATH}msa/"
PROCESSED_PROTHERM="${FILEPATH}ProTherm/processed_single_mutation.csv"
PROTHERM_SEQUENCES="${FILEPATH}ProTherm/pfam_uniprot_fasta_seq.csv"
# FULL_PROTHERM="${FILEPATH}ProTherm/full_single_mutation.csv"
OUT_PATH="${MSA_PATH}hhfiltered/hhfiltered_${DATASET}_seqs=${N_SEQUENCES}_filter=${FILTER_SIZE}/"
OUT_LOGS="/fast/external/gcarbone/adversarial-protein-sequences_out/logs/"

mkdir -p "${MSA_PATH}fasta_oneline/"
mkdir -p $OUT_PATH
mkdir -p $OUT_LOGS
eval "$(conda shell.bash hook)"
conda activate esm

printf "\n=== Build ProTherm sequences csv ===\n\n"

### select unique pfam_id, uniprot_id couples

# cat $PROCESSED_PROTHERM | awk -F ";" '{print $6";"$9}' | uniq > "${FILEPATH}ProTherm/tmp.csv"
# echo $(head -n 1 $PROCESSED_PROTHERM | awk -F ";" '{print $6";"$9}' )";FASTA;SEQUENCE;PFAM_SEQUENCE" > $PROTHERM_SEQUENCES
# cat $PROTHERM_SEQUENCES

# sed 1d "${FILEPATH}ProTherm/tmp.csv" | while read -r line; do

#    pfam_id=$(echo $line | awk -F ";" '{print $1}')
#    uniprot_id=$(echo $line | awk -F ";" '{print $2}')

#    ### Search fasta id from uniprot id

#    fasta_id=$(grep $uniprot_id $MSA_PATH"stockholm/"$pfam_id".sto" | tail -n 1 | awk -F " " '{print $2}')

#    if [[ $fasta_id != "" ]]; then

#       ### MSA to single line

#       FASTA_ONELINE="${MSA_PATH}fasta_oneline/oneline_${pfam_id}.fasta"

#       cat $MSA_PATH"fasta/"$pfam_id".fasta" | awk 'BEGIN{FS=""}{if($1==">"){if(NR==1)print $0; else {printf "\n";print $0;}}else printf toupper($0)}' > $FASTA_ONELINE

#       ### Search sequence in the MSA 

#       pfam_sequence=$(grep $fasta_id -A 1 $FASTA_ONELINE | sed 1d)
#       sequence=$(echo "${pfam_sequence//-}")

#       new_line=$(echo "${line};${fasta_id};${sequence};${pfam_sequence}")
#       echo
#       echo $new_line
#       echo $new_line >> $PROTHERM_SEQUENCES

#    else

#       echo "missing uniprot ID ${uniprot_id} in ${pfam_id}.sto"
#       echo "missing uniprot ID ${uniprot_id} in ${pfam_id}.sto" >> "${OUT_LOGS}missing_uniprot_stockholm.txt"

#    fi

# done

# rm "${FILEPATH}ProTherm/tmp.csv"


printf "\n=== Filter MSA ===\n\n"

sed 1d $PROTHERM_SEQUENCES | while read -r line; do

   pfam_id=$(echo $line | awk -F ";" '{print $1}')
   fasta_name=$(echo $line | awk -F ";" '{print $3}')
   pfam_sequence=$(echo $line | awk -F ";" '{print $5}')

   printf "\n"
   echo name: "$fasta_name" 
   echo seq: "$pfam_sequence"
   printf "\n"

   echo "Removing columns from the MSA where current sequence has gaps"

   FASTA_ONELINE="${MSA_PATH}fasta_oneline/oneline_${pfam_id}.fasta"
   cat $FASTA_ONELINE | awk 'NR % 2 == 0' > "${OUT_PATH}tmp_msa"

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
   # cat "${OUT_PATH}tmp_msa"

   echo "Building new fasta file for current sequence"

   row_count=1

   for row_seq_name in $(cat "${OUT_PATH}tmp_names"); do

      echo "$row_seq_name" >> $OUT_PATH$OUT_NO_GAPS
      sed -n ${row_count}p "${OUT_PATH}tmp_msa" >> $OUT_PATH$OUT_NO_GAPS

      row_count=$((row_count+1))

   done

   current_seq_filename=$(echo $fasta_name | sed 's|[>,]||g' | sed 's/\//_/g')
   echo current_seq_filename
   exit 0
   OUT_NO_GAPS="${pfam_id}_${current_seq_filename}_no_gaps"
   OUT_NO_GAPS_FILTERED="${pfam_id}_${current_seq_filename}_no_gaps_filter=${FILTER_SIZE}"

   echo ""
   head $OUT_PATH$OUT_NO_GAPS

   ### select top FILTER_SIZE seqs in the new msa

   echo "Filtering the new MSA"
   hhfilter -diff $FILTER_SIZE -i $OUT_PATH$OUT_NO_GAPS -o $OUT_PATH$OUT_NO_GAPS_FILTERED

   ### delete temporary files

   rm "${OUT_PATH}tmp_names"
   rm "${OUT_PATH}tmp_msa"
   rm "${OUT_PATH}swp_msa"
   rm $OUT_PATH$OUT_NO_GAPS

done 