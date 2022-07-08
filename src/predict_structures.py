#!/usr/bin/python 

import os
import json
import socket
import random
import os.path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.protein_structures import *


random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/msa/', type=str, 
# parser.add_argument("--data_dir", default='data/', type=str, 
    help="Datasets path. Choose `msa/` or `msa/hhfiltered/`.")
parser.add_argument("--dataset", default='PF00533', type=str, help="Dataset name")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
# parser.add_argument("--out_dir", default='out/', type=str, 
    help="Output data path.")
parser.add_argument("--max_tokens", default=None, type=eval, 
    help="Optionally cut sequences to maximum number of tokens. None does not cut sequences.")
parser.add_argument("--n_sequences", default=100, type=eval, 
    help="Number of sequences from the chosen dataset. None loads all sequences.")
parser.add_argument("--min_filter", default=100, type=eval, help="Minimum number of sequences selected for the filtered MSA.")

parser.add_argument("--n_substitutions", default=3, type=int, help="Number of token substitutions in the original sequence.")

parser.add_argument("--token_selection", default='max_attention', type=str, 
    help="Method used to select most relevant token idxs. Choose 'max_attention', 'max_entropy' or 'min_entropy'.")
parser.add_argument("--target_attention", default='last_layer', type=str, 
    help="Attention matrices used to choose target token idxs. Set to 'last_layer' or 'all_layers'. \
    Used only when `token_selection`=`max_attention")

parser.add_argument("--loss_method", default='max_tokens_repr', type=str, 
    help="Loss function used to compute gradients in the first embedding space. Choose 'max_prob' or 'max_tokens_repr'.")

parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
parser.add_argument("--verbose", default=True, type=eval)
args = parser.parse_args()
print("\n", args)

filename = f"msa_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}"
out_dir = 'out/' if socket.gethostname()=="dmgdottorati" else args.out_dir
out_structures_dir = os.path.join(out_dir, "structures/", filename)
os.makedirs(os.path.dirname(out_structures_dir), exist_ok=True)

perturbations_keys = ['original','max_cos','min_dist','max_dist'] 
atk_df_dir = "data/" if socket.gethostname()=="dmgdottorati" else os.path.join(args.data_dir, filename+"/data/")
atk_df = pd.read_csv(os.path.join(atk_df_dir, filename+"_atk.csv"), index_col=[0])

atk_df = atk_df[['name']+[f'{key}_sequence' for key in perturbations_keys]]
atk_df = atk_df.drop_duplicates()
atk_df = atk_df.reset_index()

out_df = pd.DataFrame()

for row_idx, row in atk_df.iterrows():

    coordinates_dict = {}
    for key in perturbations_keys:

        print(f'{key}_sequence', "=", row[f'{key}_sequence'])
        predict_structure(f'{key}_{row_idx}', row[f'{key}_sequence'], savedir=out_dir, filename=filename)

        structure_id = key+"_"+row['name']
        filepath = os.path.join(out_structures_dir, f"{key}_{row_idx}_unrelaxed_rank_1_model_1.pdb")
        coordinates_dict[f'{key}_coordinates'] = get_coordinates(structure_id, filepath)

    if np.all([len(coordinates_dict[f'{key}_coordinates'])==len(row['original_sequence']) for key in perturbations_keys]):

        for key in ['max_cos','min_dist','max_dist']:

            ##############
            # Prediction #
            # confidence # 
            #   scores   # 
            ##############

            filepath = os.path.join(out_structures_dir, f'{key}_{row_idx}_unrelaxed_rank_1_model_1_scores.json')
            f = open(filepath, "r")
            data = json.loads(f.read())
            row[f'{key}_plddt'] = np.mean(data['plddt'])
            row[f'{key}_ptm'] = data['ptm']
            
            ########
            # RMSD #
            ########  

            row[f'{key}_rmsd'] = get_RMSD(coordinates_dict['original_coordinates'], coordinates_dict[f'{key}_coordinates'])
            
            ########
            # LDDT #
            ########

            true_dmap = get_dmap(cb_coordinates=coordinates_dict['original_coordinates'])
            pred_dmap = get_dmap(cb_coordinates=coordinates_dict[f'{key}_coordinates'])
            row[f'{key}_lddt'] = get_LDDT(true_dmap, pred_dmap)

            ############
            # TM-score #
            ############

            original_filepath = os.path.join(out_structures_dir, f"original_{row_idx}_unrelaxed_rank_1_model_1.pdb")
            pert_filepath = os.path.join(out_structures_dir, f"{key}_{row_idx}_unrelaxed_rank_1_model_1.pdb")

            original_coordinates, pert_coordinates = get_corresponding_residues_coordinates(
                f"original_{row['name']}", original_filepath,
                f"{key}_{row['name']}", pert_filepath)
            # original_coordinates = coordinates_dict[f'original_coordinates']
            # pert_coordinates = coordinates_dict[f'{key}_coordinates']

            row[f'{key}_tm'] = get_TM_score(len(row['original_sequence']), original_coordinates, pert_coordinates)

            print(f"{key}   PTM = {row[f'{key}_ptm']}\tpLDDT = {row[f'{key}_plddt']:.2f}\tRMSD = {row[f'{key}_rmsd']:.2f}\t\tLDDT = {row[f'{key}_lddt']:.2f}\tTM-score = {row[f'{key}_tm']:.2f}")

        out_df = out_df.append(row, ignore_index=False)

    else:
        print("\nPart of the 3d structure is unknown")

print(f"\nSaving: {out_dir}{filename}_structure_prediction.csv")
out_df.to_csv(os.path.join(out_dir, filename+"_structure_prediction.csv"))
