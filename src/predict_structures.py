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
parser.add_argument("--target_attention", default='all_layers', type=str, 
    help="Attention matrices used to choose target token idxs. Set to 'last_layer' or 'all_layers'. \
    Used only when `token_selection`=`max_attention")

parser.add_argument("--loss_method", default='max_masked_ce', type=str, 
    help="Loss function used to compute gradients in the first embedding space. Choose 'max_masked_ce', max_prob' \
    or 'max_tokens_repr'.")

parser.add_argument("--min", default=0, type=int) 
parser.add_argument("--max", default=100, type=int) 
parser.add_argument("--plddt_ths", default=70, type=int) 
parser.add_argument("--ptm_ths", default=0.4, type=float) 

parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute.')
parser.add_argument("--verbose", default=True, type=eval)
args = parser.parse_args()
print("\n", args)

filename = f"msa_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}_{args.loss_method}_attn={args.target_attention}"
out_dir = 'out/' if socket.gethostname()=="dmgdottorati" else args.out_dir
out_structures_dir = os.path.join(out_dir, "structures/", filename)
os.makedirs(os.path.dirname(out_structures_dir), exist_ok=True)

perturbations_keys = ['min_dist','max_dist','max_cmap_dist','max_cos'] 
all_keys = ['original']+perturbations_keys
atk_df_dir = "data/" if socket.gethostname()=="dmgdottorati" else os.path.join(args.data_dir, filename+"/data/")
atk_df = pd.read_csv(os.path.join(atk_df_dir, filename+"_atk.csv"), index_col=[0])

atk_df.rename(columns = {'orig_token':'original_token'}, inplace = True)

new_df = atk_df[['name']+[f'{key}_sequence' for key in all_keys]]
new_df = new_df.drop_duplicates()
new_df = new_df.reset_index()

out_df = pd.DataFrame(columns=['seq_idx','perturbation','target_token_idxs','pert_tokens',\
        'pLDDT','PTM','LDDT','TM-score','RMSD'])   

for seq_idx, row in new_df.iterrows():

    gc.collect()

    if seq_idx>=args.min and seq_idx<=args.max:

        print(f"\n=== Sequence {seq_idx} ===\n")

        target_token_idxs = list(atk_df[atk_df['name']==row['name']]['target_token_idx'])
        print(f"target_token_idxs = {target_token_idxs}\n")

        coordinates_dict = {}
        for key in all_keys:

            if args.load is False:
                predict_structure(f'{key}_{seq_idx}', row[f'{key}_sequence'], savedir=out_dir, filename=filename)

            structure_id = key+"_"+row['name']
            filepath = os.path.join(out_structures_dir, f"{key}_{seq_idx}_unrelaxed_rank_1_model_1.pdb")
            coordinates_dict[f'{key}_coordinates'], coordinates_dict[f'{key}_missing_residues'] = get_coordinates(structure_id, filepath)

        print("\nScores:")

        same_pdb_length = np.all([len(coordinates_dict[f'{key}_coordinates'])==len(row['original_sequence']) for key in perturbations_keys])

        if same_pdb_length:

            filepath = os.path.join(out_structures_dir, f'original_{seq_idx}_unrelaxed_rank_1_model_1_scores.json')
            f = open(filepath, "r")
            data = json.loads(f.read())
            original_plddt = np.mean(data['plddt'])
            original_ptm = data['ptm']

            original_tokens = list(atk_df[atk_df['name']==row['name']][f'original_token'])
            print(f"\noriginal_sequence\ttokens = {original_tokens}", end='\t')
            print(f"\tPTM = {original_ptm}\tpLDDT = {original_plddt:.2f}")

            for key in perturbations_keys:

                #######################
                # collect coordinates #
                #######################

                pert_tokens = list(atk_df[atk_df['name']==row['name']][f'{key}_token'])
                print(f'\n{key}_sequence tokens = {pert_tokens}', end='\t')

                original_coordinates = coordinates_dict['original_coordinates']
                pert_coordinates = coordinates_dict[f'{key}_coordinates']

                original_filepath = os.path.join(out_structures_dir, f"original_{seq_idx}_unrelaxed_rank_1_model_1.pdb")
                pert_filepath = os.path.join(out_structures_dir, f"{key}_{seq_idx}_unrelaxed_rank_1_model_1.pdb")

                corr_original_coordinates, corr_pert_coordinates = \
                    get_corresponding_residues_coordinates("original", original_filepath, key, pert_filepath)

                ##############
                # Prediction #
                # confidence # 
                #   scores   # 
                ##############

                filepath = os.path.join(out_structures_dir, f'{key}_{seq_idx}_unrelaxed_rank_1_model_1_scores.json')
                f = open(filepath, "r")
                data = json.loads(f.read())
                plddt = np.mean(data['plddt'])
                ptm = data['ptm']

                print(f"\t\tPTM = {ptm}\tpLDDT = {plddt:.2f}", end="\t")

                ########
                # RMSD #
                ########  

                rmsd = get_RMSD(original_coordinates, pert_coordinates)
                
                ########
                # LDDT #
                ########

                true_dmap = get_dmap(cb_coordinates=original_coordinates)
                pert_dmap = get_dmap(cb_coordinates=pert_coordinates)
                lddt = get_LDDT(true_dmap, pert_dmap)

                ############
                # TM-score #
                ############

                tm = get_TM_score(len(row['original_sequence']), corr_original_coordinates, corr_pert_coordinates)

                print(f"\t\tRMSD = {rmsd:.2f}\tLDDT = {lddt:.2f}\tTM-score = {tm:.2f}")

                if original_ptm>=args.ptm_ths and original_plddt>=args.plddt_ths and plddt>=args.plddt_ths:

                    row_dict = {'seq_idx':seq_idx, 'perturbation':key, 'target_token_idxs':target_token_idxs,
                        'pert_tokens':pert_tokens, 'pLDDT':plddt, 'PTM':ptm, 'LDDT':lddt, 'TM-score':tm, 'RMSD':rmsd}                  
                    out_df = out_df.append(row_dict, ignore_index=True)

                else:
                    out_df = out_df.drop(out_df[out_df.seq_idx==seq_idx].index)

        else:

            print("\tPart of the 3d structure is unknown")

out_dir = os.path.join(out_dir, 'structures/')

print(f"\nSaving: {out_dir}{filename}_structure_prediction.csv")
out_df.to_csv(os.path.join(out_dir, filename+"_structure_prediction.csv"))

########
# plot #
########

print(f"\ndf size = {len(out_df)}\t n_sequences = {len(list(out_df['seq_idx'].unique()))}")

matplotlib.rc('font', **{'size': 13})
sns.set_style("darkgrid")
keys = ['PTM','pLDDT','LDDT','RMSD','TM-score']
plot = sns.pairplot(out_df, x_vars=keys, y_vars=keys, hue="perturbation", corner=True, palette='rocket', 
    hue_order=perturbations_keys)
plt.savefig(os.path.join(out_dir, filename+f"_structure_prediction_pairplot.png"))
plt.close()