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
from utils.plot import *
from sequence_attack import SequenceAttack
from models.msa_esm_embedding import MsaEsmEmbedding
from utils.protein import compute_cmaps_distance, get_max_hamming_msa

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/scratch/external/gcarbone/hhfiltered/', type=str, 
	help="Datasets path. Choose `msa/` or `hhfiltered/`.")
parser.add_argument("--dataset", default='PF00533', type=str, help="Dataset name")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
	help="Output data path.")

parser.add_argument("--max_tokens", default=None, type=eval, 
	help="Optionally cut sequences to maximum number of tokens. None does not cut sequences.")
parser.add_argument("--n_sequences", default=100, type=eval, 
	help="Number of sequences from the chosen dataset. None loads all sequences.")
parser.add_argument("--min_filter", default=50, type=eval, help="Minimum number of sequences selected for the filtered MSA.")

parser.add_argument("--n_substitutions", default=3, type=int, help="Number of token substitutions in the original sequence.")

parser.add_argument("--token_selection", default='max_attention', type=str, 
	help="Method used to select most relevant token idxs. Choose 'max_attention' or 'min_entropy'.")
parser.add_argument("--target_attention", default='last_layer', type=str, 
	help="Attention matrices used to choose target token idxs. Set to 'last_layer' or 'all_layers'. \
	Used only when `token_selection`=`max_attention")

parser.add_argument("--loss_method", default='max_tokens_repr', type=str, 
	help="Loss function used to compute gradients in the first embedding space. Choose 'max_logits' or 'max_tokens_repr'.")

parser.add_argument("--cmap_dist_lbound", default=0.2, type=int, 
	help='Lower bound for upper triangular matrix of long range contacts.')
parser.add_argument("--cmap_dist_ubound", default=0.8, type=int, 
	help='Upper bound for upper triangular matrix of long range contacts.')

parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute.')
parser.add_argument("--verbose", default=True, type=eval)
args = parser.parse_args()
print("\n", args)


out_filename = f"msa_{args.dataset}_seqs={args.n_sequences}_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}"
out_path = os.path.join(args.out_dir, "msa/", out_filename+"/")
out_plots_path = os.path.join(out_path, "plots/")
out_data_path = os.path.join(out_path, "data/")

perturbations_keys = ['masked_pred','max_cos','min_dist','max_dist'] 

if args.load:

	df = pd.read_csv(os.path.join(out_data_path, out_filename+".csv"), index_col=[0])
	cmap_df = pd.read_csv(os.path.join(out_data_path, out_filename+"_cmap.csv"))
	embeddings_distances = load_from_pickle(filepath=out_data_path, filename=out_filename)

else:

	### instantiate models and load data

	esm_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
	batch_converter = alphabet.get_batch_converter()
	n_layers = esm_model.args.layers
	esm_model = esm_model.to(args.device)

	emb_model = MsaEsmEmbedding(original_model=esm_model, alphabet=alphabet).to(args.device)
	emb_model = emb_model.to(args.device)

	atk = SequenceAttack(original_model=esm_model, embedding_model=emb_model, alphabet=alphabet)

	data, max_tokens = load_msa(
		filepath=f"{args.data_dir}hhfiltered_{args.dataset}_seqs={args.n_sequences}_filter={args.min_filter}", 
		filename=f"{args.dataset}_top_{args.n_sequences}_seqs", 
		max_model_tokens=esm_model.args.max_tokens, n_sequences=args.n_sequences, max_tokens=args.max_tokens)

	### fill dataframes

	df = pd.DataFrame()
	cmap_df = pd.DataFrame()
	embeddings_distances = []

	for seq_idx, single_sequence_data in tqdm(enumerate(data), total=len(data)):

		name, original_sequence = single_sequence_data

		seq_filename = name.replace('/','_')
		original_sequence = original_sequence.replace('-','')

		msa, max_tokens = load_msa(
			filepath=f"{args.data_dir}hhfiltered_{args.dataset}_seqs={args.n_sequences}_filter={args.min_filter}", 
			filename=f"{args.dataset}_{seq_filename}_no_gaps_filter={args.min_filter}", 
			max_model_tokens=esm_model.args.max_tokens, max_tokens=args.max_tokens)

		### put current sequence on top of the msa

		msa = dict(msa)
		if name in msa.keys():
			msa.pop(name)		
		msa = tuple(msa.items())
		msa = [(name, original_sequence)] + list(msa)

		### compute first continuous embedding

		batch_labels, batch_strs, batch_tokens = batch_converter(msa)

		with torch.no_grad():
			batch_tokens = batch_tokens.to(args.device)
			results = esm_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

		first_embedding = results["representations"][0].to(args.device)

		### sequence attacks

		target_token_idxs = atk.choose_target_token_idxs(token_selection=args.token_selection, 
			n_token_substitutions=args.n_substitutions, msa=msa, batch_tokens=batch_tokens, 
			target_attention=args.target_attention, verbose=args.verbose)

		signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, 
			target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss_method=args.loss_method)

		atk_df, emb_dist_single_seq = atk.attack_sequence(name=name, original_sequence=original_sequence, 
			original_batch_tokens=batch_tokens, msa=msa, target_token_idxs=target_token_idxs, 
			first_embedding=first_embedding, signed_gradient=signed_gradient, 
			perturbations_keys=perturbations_keys, verbose=args.verbose)

		### update sequence row in the df

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
print("\n", cmap_df.keys())

print("\nmasked_pred_accuracy:\n", df["masked_pred_accuracy"].describe())

plot_tokens_hist(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_token_substitutions(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cosine_similarity(df, filepath=out_plots_path, filename=out_filename)
plot_confidence(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
# plot_embeddings_distances(df, keys=perturbations_keys, embeddings_distances=embeddings_distances, filepath=out_plots_path, filename=out_filename)
plot_embeddings_distances(df, keys=['max_cos','min_dist','max_dist'] , embeddings_distances=embeddings_distances, filepath=out_plots_path, filename=out_filename)
plot_blosum_distances(df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)
plot_cmap_distances(cmap_df, keys=perturbations_keys, filepath=out_plots_path, filename=out_filename)



