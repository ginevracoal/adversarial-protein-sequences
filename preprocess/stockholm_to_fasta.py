import os
from Bio import SeqIO

PFAM_IDS_FILE="/scratch/external/gcarbone/ProTherm/pfam_ids.txt"
MSA_PATH='/scratch/external/gcarbone/msa/'
OUT_PATH=MSA_PATH+'fasta/'
os.makedirs(OUT_PATH, exist_ok=True)

with open(PFAM_IDS_FILE) as pfam_ids:
    for line in pfam_ids:

        family_name = line.rstrip('\n')
        records = SeqIO.parse(MSA_PATH+'stockholm/'+family_name+".sto", "stockholm")
        count = SeqIO.write(records, OUT_PATH+family_name+".fasta", "fasta")
        print(f"Converted {count} records from family {family_name}")
