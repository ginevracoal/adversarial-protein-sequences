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
from utils.protein import compute_cmaps_distance, get_max_hamming_msa

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\tversion =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/scratch/external/gcarbone/pfam/', type=str, help="Datasets path")
parser.add_argument("--out_dir", default='../out', type=str, help="Output path")
parser.add_argument("--dataset", default='fastaPF00001', type=str, help="Dataset name")
parser.add_argument("--align", default=False, type=eval, help='If True pad and align sequences')
parser.add_argument("--loss_method", default='max_tokens_repr', type=str, help="Loss function")
parser.add_argument("--target_attention", default='last_layer', type=str, help="Attention matrices used to \
	choose target token idxs. Set to 'last_layer' or 'all_layers'.")
parser.add_argument("--max_tokens", default=100, type=eval, help="Cut sequences to max number of tokens")
parser.add_argument("--n_sequences", default=100, type=eval, help="Number of sequences from the chosen dataset. \
	None loads all sequences")
parser.add_argument("--n_substitutions", default=3, type=int, help="Number of token substitutions in the original sequence")
parser.add_argument("--cmap_dist_lbound", default=0.2, type=int, help='Lower bound for upper triangular matrix of long \
	range contacts')
parser.add_argument("--cmap_dist_ubound", default=0.8, type=int, help='Upper bound for upper triangular matrix of long \
	range contacts')
parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute')
parser.add_argument("--verbose", default=True, type=eval)
args = parser.parse_args()
print("\n", args)

out_data_path = os.path.join(args.out_dir, 'data/')
out_plots_path = os.path.join(args.out_dir, 'plots/single_sequence/')
out_filename = f"{args.dataset}_align={args.align}_seqs={args.n_sequences}_toks={args.max_tokens}_subst={args.n_substitutions}"

perturbations_keys = ['pred','max_cos','min_dist','max_dist'] 

if args.load:

	df = pd.read_csv(os.path.join(out_data_path, out_filename+".csv"), index_col=[0])
	cmap_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_cmap.csv"))
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

	data, max_tokens = load_sequences(filepath=args.data_dir, filename=args.dataset, 
		max_model_tokens=esm_model.args.max_tokens, n_sequences=args.n_sequences, max_tokens=args.max_tokens, 
		align=args.align)

	### fill dataframes

	df = pd.DataFrame()
	cmap_df = pd.DataFrame()
	embeddings_distances = []

	for seq_idx, single_sequence_data in tqdm(enumerate(data), total=len(data)):

		name, original_sequence = single_sequence_data
		batch_labels, batch_strs, batch_tokens = batch_converter([single_sequence_data])

		batch_tokens = batch_tokens.to(args.device)

		with torch.no_grad():
			results = esm_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

		first_embedding = results["representations"][0].to(args.device)

		### sequence attacks

		target_token_idxs, _ = atk.choose_target_token_idxs(batch_tokens=batch_tokens, 
			n_token_substitutions=args.n_substitutions, target_attention=args.target_attention, 
			verbose=args.verbose)

		signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, 
			target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss_method=args.loss_method)

		atk_df, emb_dist_single_seq = atk.attack_sequence(name=name, original_sequence=original_sequence, 
			target_token_idxs=target_token_idxs, first_embedding=first_embedding, signed_gradient=signed_gradient, 
			perturbations_keys=perturbations_keys, verbose=args.verbose)

		# update sequence row in the df

		df = pd.concat([df, atk_df], ignore_index=True)
		embeddings_distances.append(emb_dist_single_seq)

		### contact maps distances

		cmap_df = compute_cmaps_distance(model=esm_model, alphabet=alphabet, atk_df=atk_df, cmap_df=cmap_df, 
			original_sequence=original_sequence, 
			sequence_name=name, max_tokens=max_tokens, perturbations_keys=perturbations_keys,
			cmap_dist_lbound=args.cmap_dist_lbound, cmap_dist_ubound=args.cmap_dist_ubound)

	os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
	df.to_csv(os.path.join(out_data_path,  out_filename+".csv"))
	cmap_df.to_csv(os.path.join(out_data_path,  out_filename+"_cmap.csv"))

	embeddings_distances = torch.stack(embeddings_distances)
	save_to_pickle(data=embeddings_distances, filepath=out_data_path, filename=out_filename)



print("\n", df.keys())
print("\n", cmap_df)

plot_tokens_hist(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_token_substitutions(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cosine_similarity(df, filepath=out_plots_path, filename=out_filename)
plot_confidence(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_embeddings_distances(df, keys=perturbations_keys, embeddings_distances=embeddings_distances, filepath=out_plots_path, filename=out_filename)
plot_blosum_distances(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cmap_distances(cmap_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)