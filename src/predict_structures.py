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
parser.add_argument("--data_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/data/msa/', type=str, 
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

parser.add_argument("--loss_method", default='max_masked_prob', type=str, 
    help="Loss function used to compute gradients in the first embedding space. Choose 'max_masked_ce', max_masked_prob' \
    or 'max_tokens_repr'.")

parser.add_argument("--min", default=0, type=int) 
parser.add_argument("--max", default=100, type=int) 
parser.add_argument("--plddt_ths", default=80, type=int) 
parser.add_argument("--ptm_ths", default=0.4, type=float) 
parser.add_argument("--normalize", default=False, type=eval)

parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute.')
parser.add_argument("--verbose", default=True, type=eval)
parser.add_argument("--plot", default=True, type=eval)
args = parser.parse_args()
print("\n", args)

filename = f"msa_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}_{args.loss_method}_attn={args.target_attention}"
# out_dir = 'out/' if socket.gethostname()=="dmgdottorati" else args.out_dir 
alphafold_dir = os.path.join(args.out_dir, "data/")
out_data_path = os.path.join(args.out_dir, "data/structures/", filename)
out_plots_path = os.path.join(args.out_dir, "plots/")
os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
os.makedirs(os.path.dirname(out_plots_path), exist_ok=True)

perturbations_keys = ['max_dist','max_cmap_dist','max_entropy','max_cos']

all_keys = ['original']+perturbations_keys
atk_df = pd.read_csv(os.path.join(args.data_dir, filename+"/", filename+"_atk.csv"), index_col=[0])

atk_df.rename(columns = {'orig_token':'original_token'}, inplace = True)

new_df = atk_df[['name']+[f'{key}_sequence' for key in all_keys]]
new_df = new_df.drop_duplicates()
new_df = new_df.reset_index()

out_df = pd.DataFrame(columns=['seq_idx','perturbation','target_token_idxs','pert_tokens',\
        'pLDDT','PTM','LDDT','TM-score','RMSD'])   

key_counts = {key:0. for key in perturbations_keys}

for seq_idx, row in new_df.iterrows():

    gc.collect()

    if seq_idx>=args.min and seq_idx<=args.max:

        print(f"\n=== Sequence {seq_idx} ===\n")

        target_token_idxs = list(atk_df[atk_df['name']==row['name']]['target_token_idx'])
        print(f"target_token_idxs = {target_token_idxs}\n")

        coordinates_dict = {}

        #######################
        # original prediction #
        #######################

        key='original'

        if args.load is False:
            predict_structure(f'{key}_{seq_idx}', row[f'{key}_sequence'], savedir=alphafold_dir, filename=filename)

        structure_id = key+"_"+row['name']
        # filepath = os.path.join(out_data_path, f"{key}_{seq_idx}_relaxed_rank_1_model_1.pdb")
        # coordinates_dict[f'{key}_coordinates'] = get_coordinates(structure_id, filepath)
        # original_coordinates = get_coordinates(structure_id, filepath)

        filepath = os.path.join(out_data_path, f'{key}_{seq_idx}_unrelaxed_rank_1_model_1_scores.json')
        f = open(filepath, "r")
        data = json.loads(f.read())
        original_plddt = np.mean(data['plddt'])
        original_ptm = data['ptm']

        original_tokens = list(atk_df[atk_df['name']==row['name']][f'original_token'])

        print(f"\noriginal_sequence\ttokens = {original_tokens}", end='\t')
        print(f"\tpTM = {original_ptm}\tpLDDT = {original_plddt:.2f}")

        #################
        # perturbations #
        #################

        plddt = args.plddt_ths
        same_pdb_length = True

        # if original_ptm>=args.ptm_ths and original_plddt>=args.plddt_ths:
        if original_plddt>=args.plddt_ths:

            # original_coordinates = coordinates_dict['original_coordinates']
            original_filepath = os.path.join(out_data_path, f"original_{seq_idx}_relaxed_rank_1_model_1.pdb")

            for key in perturbations_keys:

                pert_tokens = list(atk_df[atk_df['name']==row['name']][f'{key}_token'])
                print(f'\n{key}_sequence\ttokens = {pert_tokens}', end='\t')

                if args.load is False:
                    predict_structure(f'{key}_{seq_idx}', row[f'{key}_sequence'], savedir=alphafold_dir, filename=filename)

                # structure_id = key+"_"+row['name']
                # filepath = os.path.join(out_data_path, f"{key}_{seq_idx}_relaxed_rank_1_model_1.pdb")
                # coordinates_dict[f'{key}_coordinates'] = get_coordinates(structure_id, filepath)
                
                scores_filepath = os.path.join(out_data_path, f'{key}_{seq_idx}_unrelaxed_rank_1_model_1_scores.json')
                f = open(scores_filepath, "r")
                data = json.loads(f.read())
                plddt = np.mean(data['plddt'])
                ptm = data['ptm']

                # if plddt<args.plddt_ths:

                #     print("\n\n\tLow confidence in structure prediction, discarding this sequence.")
                #     out_df = out_df.drop(out_df[out_df.seq_idx==seq_idx].index)
                #     break

                # else:

                # pert_coordinates = coordinates_dict[f'{key}_coordinates']
                pert_filepath = os.path.join(out_data_path, f"{key}_{seq_idx}_relaxed_rank_1_model_1.pdb")

                # pert_coordinates = get_coordinates(key, pert_filepath)
                # same_pdb_length = len(pert_coordinates)==len(original_coordinates)

                original_coordinates, pert_coordinates = \
                    get_corresponding_residues_coordinates("original", original_filepath, key, pert_filepath)


                # if same_pdb_length is False:
                #     exit()
                    # print("\tPart of the 3d structure is unknown, discarding this sequence.")
                #     out_df = out_df.drop(out_df[out_df.seq_idx==seq_idx].index)
                #     break
                #     # original_coordinates, pert_coordinates = \
                #     #     get_available_residues_coordinates("original", original_filepath, key, pert_filepath)

                # else:

                # if same_pdb_length is True:

                ##########
                # scores #
                ##########

                print(f"\tpTM = {ptm}\tpLDDT = {plddt:.2f}", end='\t')

                rmsd = get_RMSD(original_coordinates, pert_coordinates)

                true_dmap = get_dmap(cb_coordinates=original_coordinates)
                pert_dmap = get_dmap(cb_coordinates=pert_coordinates)
                lddt = get_LDDT(true_dmap, pert_dmap)

                tm = get_TM_score(len(row['original_sequence']), original_coordinates, pert_coordinates)

                print(f"\t\tRMSD = {rmsd:.2f}\tLDDT = {lddt:.2f}\tTM-score = {tm:.2f}")

                row_dict = {'seq_idx':seq_idx, 'perturbation':key, 'target_token_idxs':target_token_idxs,
                    'pert_tokens':pert_tokens, 'pLDDT':plddt, 'pTM':ptm, 'LDDT':lddt, 'TM-score':tm, 
                    'RMSD':rmsd}                  
                out_df = out_df.append(row_dict, ignore_index=True)

                key_counts[key] += 1

                # if args.normalize:
                #     from sklearn.preprocessing import minmax_scale
                #     rows_idxs = out_df['seq_idx']==seq_idx
                #     if rows_idxs.sum()>0:
                #         out_df.loc[rows_idxs, ['RMSD']] = minmax_scale(out_df[rows_idxs]['RMSD'])

        else:

            print("\n\n\tLow confidence in structure prediction, discarding this sequence.")



print(f"\nSaving: {out_data_path}{filename}_structure_prediction.csv")
out_df.to_csv(os.path.join(out_data_path, filename+"_structure_prediction.csv"))
print(f"\ndf size = {len(out_df)}\t max_n_sequences = {len(list(out_df['seq_idx'].unique()))}")

print(key_counts.items())

########
# plot #
########

linestyles=[':','-.','--','-']

if args.plot:

    matplotlib.rc('font', **{'size': 13})
    sns.set_style("darkgrid")
    sns.set_palette("mako", len(perturbations_keys))

    keys = ['pTM','pLDDT']
    g = sns.pairplot(out_df, x_vars=keys, y_vars=keys, hue="perturbation", diag_kws={'fill': False}, corner=True, 
        hue_order=perturbations_keys)    
    diag = g.diag_axes
    for ax in diag:
        for line, ls in zip(ax.lines, linestyles):
            line.set_linestyle(ls)    

    plt.savefig(os.path.join(out_plots_path, filename+f"_structure_prediction_pairplot.png"))
    plt.close()    

    keys = ['LDDT','TM-score','RMSD']
    g = sns.pairplot(out_df, x_vars=keys, y_vars=keys, hue="perturbation", diag_kws={'fill': False}, corner=True, 
        hue_order=perturbations_keys)
    diag = g.diag_axes
    for ax in diag:
        for line, ls in zip(ax.lines, linestyles):
            line.set_linestyle(ls)    

    plt.savefig(os.path.join(out_plots_path, filename+f"_structure_prediction_pairplot2.png"))
    plt.close()