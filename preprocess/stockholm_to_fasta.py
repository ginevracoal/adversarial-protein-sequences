import os
import argparse
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument('--pfam_ids_filepath', default='/scratch/external/gcarbone/ProTherm/pfam_ids.txt')
parser.add_argument('--msa_path', default='/scratch/external/gcarbone/msa/')
args = parser.parse_args()

OUT_PATH=args.msa_path+'fasta/'
os.makedirs(OUT_PATH, exist_ok=True)

with open(args.pfam_ids_filepath) as pfam_ids:
    for line in pfam_ids:

        try:
            family_name = line.rstrip('\n')
            records = SeqIO.parse(args.msa_path+'stockholm/'+family_name+".sto", "stockholm")
            count = SeqIO.write(records, OUT_PATH+family_name+".fasta", "fasta")
            print(f"Converted {count} records from family {family_name}")

        except:
            print(f"File not found for {family_name}")
            pass
