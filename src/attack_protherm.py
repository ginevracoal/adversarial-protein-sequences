#!/usr/bin/python 

import os
import esm
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.data import *
from utils.plot_protherm import *
from sequence_attack import SequenceAttack
from models.esm_embedding import EsmEmbedding
from models.msa_esm_embedding import MsaEsmEmbedding
from utils.protein_sequences import compute_cmaps_distance_df, get_max_hamming_msa, get_contact_map

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/scratch/external/gcarbone/', type=str, help="Datasets path.")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
    help="Output data path.")
parser.add_argument("--model", default='ESM', type=str, help="Choose 'ESM' or 'ESM_MSA'")
parser.add_argument("--max_tokens", default=None, type=eval, 
    help="Optionally cut sequences to maximum number of tokens. None does not cut sequences.")
parser.add_argument("--min_filter", default=50, type=int, help="Minimum number of sequences selected for the filtered MSA.")
parser.add_argument("--n_substitutions", default=10, type=int, help="Number of token substitutions in the original sequence.")

parser.add_argument("--token_selection", default='max_attention', type=str, 
    help="Method used to select most relevant token idxs. Choose 'max_attention', 'max_entropy' or 'min_entropy'.")
parser.add_argument("--target_attention", default='last_layer', type=str, 
    help="Attention matrices used to choose target token idxs. Set to 'last_layer' or 'all_layers'. \
    Used only when `token_selection`=`max_attention")

parser.add_argument("--loss_method", default='max_masked_prob', type=str, 
    help="Loss function used to compute gradients in the first embedding space. Choose 'max_masked_ce', max_masked_prob' \
    or 'max_tokens_repr'.")

parser.add_argument("--cmap_dist_lbound", default=0.2, type=int, 
    help='Lower bound for upper triangular matrix of long range contacts.')
parser.add_argument("--cmap_dist_ubound", default=0.8, type=int, 
    help='Upper bound for upper triangular matrix of long range contacts.')

parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute.')
parser.add_argument("--verbose", default=True, type=eval)
args = parser.parse_args()
print("\n", args)


out_filename = f"ProTherm_{args.model}_atk_max_toks={args.max_tokens}_subst={args.n_substitutions}_minFilter={args.min_filter}_{args.loss_method}_attn={args.target_attention}"
out_plots_path = os.path.join(args.out_dir, "plots/", out_filename+"/")
out_data_path =  os.path.join(args.out_dir, "data/msa/", out_filename+"/")
os.makedirs(os.path.dirname(out_plots_path), exist_ok=True)
os.makedirs(os.path.dirname(out_data_path), exist_ok=True)

if args.model=='ESM':
    perturbations_keys = ['protherm','max_dist','max_cos','max_cmap_dist'] 

elif args.model=='ESM_MSA':
    perturbations_keys = ['protherm']#,'max_dist','max_cos','max_cmap_dist','max_entropy']

else:
    raise NotImplementedError

if args.load:

    atk_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_atk.csv"), index_col=[0])
    cmap_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_cmaps.csv"))

else:

    ### load data
    protherm_df = pd.read_csv(f"{args.data_dir}ProTherm/processed_single_mutation.csv", sep=';')
    sequences_df = pd.read_csv(f"{args.data_dir}ProTherm/sequences_single_mutation.csv", sep=';')

    ### select rows with top DDG values

    stabilization_ths=2.
    protherm_df = protherm_df[(protherm_df['DDG']>stabilization_ths) | (protherm_df['DDG']<-stabilization_ths)]
    print("\nDDG>0\n", protherm_df[protherm_df['DDG']>0]['DDG'].describe())
    print("\nDDG<0\n", protherm_df[protherm_df['DDG']<0]['DDG'].describe())

    print("\nprotherm_df:\n\n", protherm_df.head())
    print("\nsequences_df:\n\n", sequences_df.head())
    print(f"\n{len(sequences_df)} ProTherm sequences")

    if args.model=='ESM':
        pretrained_model=esm.pretrained.esm1b_t33_650M_UR50S
        embedding_model=EsmEmbedding

    elif args.model=='ESM_MSA':
        pretrained_model=esm.pretrained.esm_msa1b_t12_100M_UR50S
        embedding_model=MsaEsmEmbedding

    else:
        raise NotImplementedError

    esm_model, alphabet = pretrained_model()
    batch_converter = alphabet.get_batch_converter()
    n_layers = esm_model.args.layers
    esm_model = esm_model.to(args.device)
    
    emb_model = embedding_model(original_model=esm_model, alphabet=alphabet).to(args.device)
    emb_model = emb_model.to(args.device)

    atk = SequenceAttack(original_model=esm_model, embedding_model=emb_model, alphabet=alphabet)

    ### Dataframes

    atk_df = pd.DataFrame()
    cmap_df = pd.DataFrame()
    embeddings_distances = []

    # for row_idx, row in sequences_df.iterrows():
    for row_idx, row in tqdm(sequences_df.iterrows(), total=len(sequences_df)):

        original_sequence=row.SEQUENCE[:args.max_tokens]

        print(f"\n=== Mutations for sequence {row.FASTA} ===\n")
        print(original_sequence, "\n")

        mutations_df = protherm_df[(protherm_df.PFAM==row.PFAM) & (protherm_df.UNIPROT==row.UNIPROT)]
        print(mutations_df.head())

        if not mutations_df.empty:

            name = row.FASTA
            seq_filename = row.FASTA.replace('/','_')
            
            ### compute first continuous embedding

            if args.model=='ESM_MSA':

                msa, max_tokens = load_msa(
                    filepath=f"{args.data_dir}msa/hhfiltered/hhfiltered_{row.PFAM}_filter={args.min_filter}", 
                    filename=f"{row.PFAM}_{seq_filename}_no_gaps_filter={args.min_filter}", 
                    max_model_tokens=esm_model.args.max_tokens, max_tokens=args.max_tokens)

                ### put current sequence on top of the msa
                msa = dict(msa)
                if name in msa.keys():
                    msa.pop(name)       
                msa = tuple(msa.items())
                msa = [(name, original_sequence)] + list(msa)

                batch_labels, batch_strs, batch_tokens = batch_converter(msa)
                batch_tokens = batch_tokens.to(args.device)

                with torch.no_grad():
                    repr_layer_idx = 0
                    results = esm_model(batch_tokens, repr_layers=[repr_layer_idx], return_contacts=True)
                    first_embedding = results["representations"][repr_layer_idx].to(args.device)

            else:

                msa=None
                
                batch_labels, batch_strs, batch_tokens = batch_converter([(name, original_sequence)])
                batch_tokens = batch_tokens.to(args.device)

                with torch.no_grad():
                    results = esm_model(batch_tokens, repr_layers=[0], return_contacts=True)
                    first_embedding = results["representations"][0].to(args.device)

            ### choose target positions

            target_token_idxs, target_tokens_attention = atk.choose_target_token_idxs(token_selection=args.token_selection, 
                n_token_substitutions=args.n_substitutions, msa=msa, batch_tokens=batch_tokens, 
                target_attention=args.target_attention, verbose=False)

            for _, mutation_row in mutations_df.iterrows():

                pdb_start = mutation_row.PDB_START
                target_pdb_idxs = [idx+pdb_start for idx in target_token_idxs]
                protherm_token_idx = mutation_row.POSITION
                token_idx_found = bool(protherm_token_idx in target_pdb_idxs)
                print(f"target_pdb_idxs = {target_pdb_idxs}")
                print(f"protherm_token_idx = {protherm_token_idx}")
                print(f"token_idx_found = {token_idx_found}")

                ### attack sequence at ProTherm mutant position

                for perturbation in perturbations_keys:

                    target_token=mutation_row.MUTANT if perturbation=='protherm' else None

                    target_token_idx=protherm_token_idx-pdb_start
                    if mutation_row.WILD_TYPE==original_sequence[target_token_idx]:

                        signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence,
                            batch_tokens=batch_tokens, target_token_idxs=[target_token_idx], first_embedding=first_embedding, 
                            loss_method=args.loss_method)

                        row_dict = atk.attack_single_position(name=name, original_sequence=original_sequence, 
                            original_batch_tokens=batch_tokens, msa=msa, position=protherm_token_idx, pdb_start=pdb_start,
                            target_token_idx=target_token_idx, first_embedding=first_embedding, signed_gradient=signed_gradient, 
                            perturbation=perturbation, verbose=args.verbose, target_token=target_token)

                        row_dict['DDG'] = mutation_row.DDG
                        row_dict['token_idx_found'] = token_idx_found
                        row_dict['chose_mutant_token'] = bool(row_dict['target_token']==mutation_row.MUTANT)

                        atk_df = atk_df.append(row_dict, ignore_index=True)

                        ### contact maps distances

                        perturbed_sequences_dict = {perturbation:row_dict['perturbed_sequence']}

                        df = compute_cmaps_distance_df(model=esm_model, alphabet=alphabet, original_sequence=original_sequence, 
                                sequence_name=name, perturbed_sequences_dict=perturbed_sequences_dict, verbose=args.verbose,
                                cmap_dist_lbound=args.cmap_dist_lbound, cmap_dist_ubound=args.cmap_dist_ubound)

                        cmap_df = pd.concat([cmap_df, df], ignore_index=True)

    atk_df.to_csv(os.path.join(out_data_path,  out_filename+"_atk.csv"))
    cmap_df.to_csv(os.path.join(out_data_path,  out_filename+"_cmaps.csv"))

### plots

print("\natk_df:\n", atk_df.keys())
print("\ncmap_df:\n", cmap_df.keys())

plot_confidence(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_embeddings_distances(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_blosum_distances(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cmap_distances(cmap_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
