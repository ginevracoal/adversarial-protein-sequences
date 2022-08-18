#!/bin/bash

FILEPATH="/scratch/external/gcarbone/"
PROTHERM="${FILEPATH}ProTherm/single_mutation.csv"
OUT_CSV="${FILEPATH}ProTherm/processed_single_mutation.csv"
FULL_PFAM="${FILEPATH}Pfam-A.full"
INP_PDB='/scratch/area/cuturellof/proteins_databases/pdb/'
PDB_PFAM_MAPPING="${FILEPATH}mappings/pdb_pfam_mapping.csv" 
PDB_UNIPROT_MAPPING="${FILEPATH}mappings/pdb_chain_uniprot.csv" 
PFAM_IDS="${FILEPATH}ProTherm/pfam_ids.txt"
PDB_IDS="${FILEPATH}ProTherm/pdb_ids.txt"
UNIPROT_IDS="${FILEPATH}ProTherm/uniprot_ids.txt"
OUT_PDB="${FILEPATH}pdb/"
MSA_PATH="${FILEPATH}msa/"


printf "\n=== Build new ProTherm csv ===\n\n"

rm $OUT_CSV
echo "PDB;CHAIN;POSITION;WILD_TYPE;MUTANT;PFAM;PDB_START;PDB_END;UNIPROT" >> $OUT_CSV
head $OUT_CSV

sed 1d $PROTHERM | while read -r line; do

   pdb_chain=$(echo $line | grep "pdb" | awk '{sub(/.pdb/, " "); print $1}') 

   pdb=$( echo $pdb_chain | sed 's/.$//' | sed 's/.*/\L&/g' )
   wild_type=$(echo $line | awk -F ";" '{print $2}')
   chain=$(echo $line | awk -F ";" '{print $3}')
   position=$(echo $line | awk -F ";" '{print $4}')
   mutant=$(echo $line | awk -F ";" '{print $5}')

   cat $PDB_PFAM_MAPPING | grep "$pdb,$chain" > "pfam_matches"
   pfam=""
   while read -r match; do
      pdb_start=$(echo $match | awk -F "," '{print $3}')
      pdb_end=$(echo $match |  awk -F "," '{print $4}')

      if (($pdb_start <= $position && $position <= $pdb_end)); then
         # echo $match
         pfam=$(echo $match | grep "$pdb,$chain" | awk -F, '{print $5}')         
      fi
   done < "pfam_matches"
   rm "pfam_matches"

   uniprot=$(cat $PDB_UNIPROT_MAPPING | grep "$pdb,$chain" | awk -F, '{print $3}')

   if [[ $pfam != "" ]]; then
      new_line=$(echo "$pdb;$chain;$position;$wild_type;$mutant;$pfam;$pdb_start;$pdb_end;$uniprot")
      echo $new_line
      echo $new_line >> $OUT_CSV
   fi

done

exit 0


printf "\n=== Get PFAM alignments ===\n\n"

eval "$(conda shell.bash hook)"
conda activate esm

cat $OUT_CSV | sed 1d | awk -F ";" '{print $6}' | uniq > $PFAM_IDS

OUT_STOCK="${MSA_PATH}stockholm/"
OUT_FASTA="${MSA_PATH}fasta/"
mkdir -p $OUT_STOCK
mkdir -p $OUT_FASTA

while read -r pfam_id; do

   ### Get alignments in stockholm format

   echo $pfam_id
   line=$(grep -m 1 $pfam_id -B 2 -n $FULL_PFAM | grep 'STOCKHOLM')
   line="${line%-*}"
   printf "\n$line"

   awk -v l=$line '{if(NR>=l)if($1!="//")print $0; else {print $0; exit}}' $FULL_PFAM > $OUT_STOCK$pfam_id".sto"
   echo $pfam_id".sto"
   head $OUT_STOCK$pfam_id".sto"

   ### Check sto file exists

   if [ ! -f $OUT_STOCK$pfam_id".sto" ]
   then
       echo "File $pfam_id".sto" does not exist"
   fi

   ### Convert stockholm alignments to fasta format

   seqmagick convert $OUT_STOCK$pfam_id".sto" $OUT_FASTA$pfam_id".fasta"
   echo
   echo $pfam_id".fasta"
   head $OUT_FASTA$pfam_id".fasta"

done < $PFAM_IDS

# printf "\n=== Convert stockholm alignments to fasta format ===\n\n"

# eval "$(conda shell.bash hook)"
# conda activate esm

# python stockholm_to_fasta.py --pfam_ids_filepath=$PFAM_IDS --msa_path="${FILEPATH}msa/"

printf "\n=== Get ProTherm PDBs ===\n\n"

cat $PROTHERM | grep "pdb" | awk '{sub(/.pdb/, " "); print $1}' | uniq > $PDB_IDS

mkdir -p $OUT_PDB
OLD_WDIR=$(pwd)
cd $OUT_PDB

while read -r pdb_id; do

   pdb=$( echo $pdb_id | sed 's/.$//' | sed 's/.*/\L&/g' )
   pdb_folder=${pdb:1:2}

   file=$INP_PDB$pdb_folder'/pdb'$pdb'.ent.gz'
   cp $file $OUT_PDB
   target='pdb'$pdb'.ent'
   if [ -f "$target" ]; then
      ok=1
   else
      gunzip 'pdb'$pdb'.ent.gz'
   fi

   echo $target

done < $PDB_IDS
rm *.gz

cd $OLD_WDIR