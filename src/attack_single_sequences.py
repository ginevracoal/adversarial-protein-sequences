#!/usr/bin/python 

import os
import esm
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data import *
from utils.plot import *
from sequence_attack import SequenceAttack
from models.esm_embedding import EsmEmbedding
from utils.protein_sequences import compute_cmaps_distance_df, get_max_hamming_msa

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", default='/scratch/external/gcarbone/pfam/', type=str, help="Datasets path.")
parser.add_argument("--data_dir", default='/scratch/external/gcarbone/msa/hhfiltered/', type=str, help="Datasets path.")
parser.add_argument("--dataset", default='PF00533', type=str, help="Dataset name.")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
	help="Output data path.")

parser.add_argument("--max_tokens", default=None, type=eval, 
	help="Optionally cut sequences to maximum number of tokens. None does not cut sequences.")
parser.add_argument("--n_sequences", default=100, type=eval, 
	help="Number of sequences from the chosen dataset. None loads all sequences.")
parser.add_argument("--min_filter", default=100, type=eval, help="Minimum number of sequences selected for the filtered MSA.")

parser.add_argument("--n_substitutions", default=3, type=int, help="Number of token substitutions in the original sequence.")

parser.add_argument("--token_selection", default='max_attention', type=str, 
	help="Method used to select most relevant token idxs. Only 'max_attention' is available on single sequence models.")
parser.add_argument("--target_attention", default='last_layer', type=str, 
	help="Attention matrices used to choose target token idxs. Set to 'last_layer' or 'all_layers'.")

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

# out_filename = f"single_seq_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_{args.loss_method}"
out_filename = f"single_seq_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}_{args.loss_method}_attn={args.target_attention}"

out_plots_path = os.path.join(args.out_dir, "plots/", out_filename+"/")
out_data_path =  os.path.join(args.out_dir, "data/single_sequence/", out_filename+"/")
os.makedirs(os.path.dirname(out_plots_path), exist_ok=True)
os.makedirs(os.path.dirname(out_data_path), exist_ok=True)

perturbations_keys = ['masked_pred','max_dist','max_cos','max_cmap_dist']

if args.load:

	atk_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_atk.csv"), index_col=[0])
	cmap_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_cmaps.csv"))
	embeddings_distances = load_from_pickle(filepath=out_data_path, filename=out_filename)

else:

	### instantiate models and load data    

	esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
	batch_converter = alphabet.get_batch_converter()
	n_layers = esm_model.args.layers
	esm_model = esm_model.to(args.device)
	
	emb_model = EsmEmbedding(original_model=esm_model, alphabet=alphabet).to(args.device)
	emb_model = emb_model.to(args.device)

	atk = SequenceAttack(original_model=esm_model, embedding_model=emb_model, alphabet=alphabet)

	# data, max_tokens = load_pfam(filepath=args.data_dir, filename=args.dataset, 
	# 	max_model_tokens=esm_model.args.max_tokens, n_sequences=args.n_sequences, max_tokens=args.max_tokens)

	data, max_tokens = load_msa(
		filepath=f"{args.data_dir}hhfiltered_{args.dataset}_seqs={args.n_sequences}_filter={args.min_filter}", 
		filename=f"{args.dataset}_top_{args.n_sequences}_seqs", 
		max_model_tokens=esm_model.args.max_tokens, n_sequences=args.n_sequences, max_tokens=args.max_tokens)

	### fill dataframes

	atk_df = pd.DataFrame()
	cmap_df = pd.DataFrame()
	embeddings_distances = []


	for seq_idx, single_sequence_data in tqdm(enumerate(data), total=len(data)):

		name, original_sequence = single_sequence_data
		original_sequence = original_sequence.replace('-','')
		
		batch_labels, batch_strs, batch_tokens = batch_converter([(name, original_sequence)])
		batch_tokens = batch_tokens.to(args.device)

		with torch.no_grad():
			results = esm_model(batch_tokens, repr_layers=[0], return_contacts=True)
			first_embedding = results["representations"][0].to(args.device)

		### sequence attacks

		target_token_idxs, target_tokens_attention = atk.choose_target_token_idxs(batch_tokens=batch_tokens, 
			n_token_substitutions=args.n_substitutions, token_selection=args.token_selection,
			target_attention=args.target_attention, verbose=args.verbose)

		signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, 
			batch_tokens=batch_tokens,
			target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss_method=args.loss_method)

		df, emb_dist_single_seq = atk.incremental_attack(name=name, original_sequence=original_sequence, 
			original_batch_tokens=batch_tokens, target_token_idxs=target_token_idxs, 
			target_tokens_attention=target_tokens_attention, first_embedding=first_embedding, 
			signed_gradient=signed_gradient, perturbations_keys=perturbations_keys, verbose=args.verbose)

		### update sequence row in the df

		atk_df = pd.concat([atk_df, df], ignore_index=True)
		embeddings_distances.append(emb_dist_single_seq)

		### contact maps distances

		perturbed_sequences_dict = {key:df[f'{key}_sequence'].unique()[0] for key in perturbations_keys}

		df = compute_cmaps_distance_df(model=esm_model, alphabet=alphabet, original_sequence=original_sequence, 
				sequence_name=name, perturbed_sequences_dict=perturbed_sequences_dict,
				cmap_dist_lbound=args.cmap_dist_lbound, cmap_dist_ubound=args.cmap_dist_ubound)

		cmap_df = pd.concat([cmap_df, df], ignore_index=True)

	os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
	atk_df.to_csv(os.path.join(out_data_path,  out_filename+"_atk.csv"))
	cmap_df.to_csv(os.path.join(out_data_path,  out_filename+"_cmaps.csv"))

	embeddings_distances = torch.stack(embeddings_distances)
	save_to_pickle(data=embeddings_distances, filepath=out_data_path, filename=out_filename)


### plots

print("\natk_df:\n", atk_df.keys())
print("\ncmap_df:\n", cmap_df.keys())

plot_attention_scores(atk_df, filepath=out_plots_path, filename=out_filename)
# plot_tokens_hist(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_token_substitutions(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_confidence(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_embeddings_distances(atk_df, keys=perturbations_keys, embeddings_distances=embeddings_distances, filepath=out_plots_path, filename=out_filename)
plot_blosum_distances(atk_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cmap_distances(cmap_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)



