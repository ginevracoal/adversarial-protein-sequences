#!/bin/bash

### Extract PFAM IDs and UNIPROT IDs from ProTherm database.

FILEPATH="/scratch/external/gcarbone/ProTherm/"
INP_FILE="${FILEPATH}single_mutation.csv"
OUT_PDB="${FILEPATH}pdb_ids.txt"
OUT_PFAM="${FILEPATH}pfam_ids.txt"
OUT_UNIPROT="${FILEPATH}uniprot_ids.txt"

cat $INP_FILE | grep "pdb" | awk '{sub(/.pdb/, " "); print $1}' | uniq > $OUT_PDB

while read -r pdb_id; do

   printf "\n=== PDB $pdb_id ===\n"
   PDB_ID=$( echo $pdb_id | sed 's/.$//' | sed 's/.*/\L&/g' )
   CHAIN=$( echo "${pdb_id: -1}" )

   printf "\nPFAM:\n"
   sed -n "2p" "/scratch/external/gcarbone/mappings/pdb_pfam_mapping.csv"
   cat "/scratch/external/gcarbone/mappings/pdb_pfam_mapping.csv" | grep "$PDB_ID,$CHAIN"
   cat "/scratch/external/gcarbone/mappings/pdb_pfam_mapping.csv" | grep "$PDB_ID,$CHAIN" | awk -F, '{print $5}' | uniq >> $OUT_PFAM
   
   # printf "\nUNIPROT:\n"
   # sed -n "2p" "/scratch/external/gcarbone/mappings/pdb_chain_uniprot.csv"
   # cat "/scratch/external/gcarbone/mappings/pdb_chain_uniprot.csv" | grep "$PDB_ID,$CHAIN"
   # cat "/scratch/external/gcarbone/mappings/pdb_chain_uniprot.csv" | grep "$PDB_ID,$CHAIN" | awk -F, '{print $3}' | uniq >> $OUT_UNIPROT

done < $OUT_PDB

### Get PFAM alignments in stockholm format

PFAM_FILE='/scratch/external/gcarbone/Pfam-A.full'
OUT_PATH='/scratch/external/gcarbone/msa/stockholm/'
mkdir -p $OUT_PATH

while read -r family_name; do

   line=$(grep -m 1 $family_name -B 2 -n $PFAM_FILE | grep 'STOCKHOLM')
   line="${line%-*}"
   awk -v l=$line '{if(NR>=l)if($1!="//")print $0; else {print $0; exit}}' $PFAM_FILE > $OUT_PATH$family_name".sto"
   echo $OUT_PATH$family_name".sto"

done < $OUT_PFAM

### Stockholm to fasta

eval "$(conda shell.bash hook)"
conda activate esm
python stockholm_to_fasta.py

### Protherm hhfilter

PFAM_IDS="${FILEPATH}pfam_ids.txt"

while read -r pfam_id; do

   ./hhfilter.sh $pfam_id

done < $PFAM_IDS
