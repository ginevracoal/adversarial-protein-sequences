#!/bin/bash

# FILEPATH='/scratch/external/gcarbone/msa/fasta/'
FILEPATH='/scratch/area/cuturellof/all_pfam/MSA_rp15/'
PDB_IDS="/scratch/external/gcarbone/ProTherm/pdb_ids.txt"

# per ogni pdb id in  protherm cerco la famiglia in pdb pfam mapping e filtro

### loop through PDB IDs

while read -r pdb_id; do

   printf "\n=== PDB $pdb_id ===\n"
   PDB_ID=$( echo $pdb_id | sed 's/.$//' | sed 's/.*/\L&/g' )

   ### search for unique 
   cat "/scratch/external/gcarbone/mappings/pdb_pfam_mapping.csv" | grep "$PDB_ID,$CHAIN" #| awk -F, '{print $5}' #| uniq

done < $PDB_IDS


# while read -r pfam_id; do

#    printf "\n=== $pfam_id ===\n"

#    ### Check .fasta file exists

#    if [ ! -f $FILEPATH$pfam_id".fasta" ]
#    then
#        echo "File $pfam_id".fasta" does not exist"
#    fi

#    ### filter MSA

#    ./hhfilter.sh $pfam_id

# done < $PFAM_IDS
