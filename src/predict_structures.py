#!/usr/bin/python 

import os
import gc 
import json
import socket
import random
import os.path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.protein_structures import *

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/msa/', type=str, 
    help="Datasets path. Choose `msa/` or `msa/hhfiltered/`.")
parser.add_argument("--dataset", default='PF00533', type=str, help="Dataset name")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
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
parser.add_argument("--load", default=False, type=eval, help='If True load else compute.')
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

    gc.collect()

    print(f"\n=== Sequence {row_idx} ===\n")

    coordinates_dict = {}
    for key in perturbations_keys:

        print(f'{key}_sequence', "=", row[f'{key}_sequence'])

        if args.load is False:
            predict_structure(f'{key}_{row_idx}', row[f'{key}_sequence'], savedir=out_dir, filename=filename)

        structure_id = key+"_"+row['name']
        filepath = os.path.join(out_structures_dir, f"{key}_{row_idx}_unrelaxed_rank_1_model_1.pdb")
        coordinates_dict[f'{key}_coordinates'] = get_coordinates(structure_id, filepath)

    print("\nScores:\n")

    if np.all([len(coordinates_dict[f'{key}_coordinates'])==len(row['original_sequence']) for key in perturbations_keys]):

        for key in ['max_cos','min_dist','max_dist']:

            original_coordinates = coordinates_dict['original_coordinates']
            pert_coordinates = coordinates_dict[f'{key}_coordinates']

            original_filepath = os.path.join(out_structures_dir, f"original_{row_idx}_unrelaxed_rank_1_model_1.pdb")
            pert_filepath = os.path.join(out_structures_dir, f"{key}_{row_idx}_unrelaxed_rank_1_model_1.pdb")
            same_residues_original_coordinates, same_residues_pert_coordinates = \
                get_corresponding_residues_coordinates(f"original_{row['name']}", original_filepath,
                    f"{key}_{row['name']}", pert_filepath)

            ##############
            # Prediction #
            # confidence # 
            #   scores   # 
            ##############

            filepath = os.path.join(out_structures_dir, f'{key}_{row_idx}_unrelaxed_rank_1_model_1_scores.json')
            f = open(filepath, "r")
            data = json.loads(f.read())
            plddt = np.mean(data['plddt'])
            ptm = data['ptm']
            
            ########
            # LDDT #
            ########

            true_dmap = get_dmap(cb_coordinates=original_coordinates)
            pert_dmap = get_dmap(cb_coordinates=pert_coordinates)
            lddt = get_LDDT(true_dmap, pert_dmap)

            ############
            # TM-score #
            ############

            tm = get_TM_score(len(row['original_sequence']), 
                same_residues_original_coordinates, same_residues_pert_coordinates)

            ########
            # RMSD #
            ########  

            rmsd = get_RMSD(original_coordinates, pert_coordinates)

            print(f"{key}    PTM = {ptm}\tpLDDT = {plddt:.2f}\t\tRMSD = {rmsd:.2f}\tLDDT = {lddt:.2f}\tTM-score = {tm:.2f}")

            row_dict = {'seq_idx':row_idx, 'perturbation':key, 'pLDDT':plddt, 'PTM':ptm, 'LDDT':lddt, 'TM-score':tm, 'RMSD':rmsd}
            out_df = out_df.append(row_dict, ignore_index=True)

        del data, coordinates_dict

    else:
        print("Part of the 3d structure is unknown")


print(f"\nSaving: {out_dir}{filename}_structure_prediction.csv")
out_df.to_csv(os.path.join(out_dir, filename+"_structure_prediction.csv"))

########
# plot #
########

matplotlib.rc('font', **{'size': 13})
sns.set_style("darkgrid")
keys = ['PTM','pLDDT','LDDT','RMSD','TM-score']
plot = sns.pairplot(out_df, x_vars=keys, y_vars=keys, hue="perturbation", palette='rocket', corner=True)
plt.savefig(os.path.join(out_dir, filename+f"_structure_prediction_pairplot.png"))
plt.close()