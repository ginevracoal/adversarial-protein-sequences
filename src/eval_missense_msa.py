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
from Bio.SubsMat import MatrixInfo
blosum = MatrixInfo.blosum62

from utils.data import *
from utils.plot import *
from sequence_attack import SequenceAttack
from models.msa_esm_embedding import MsaEsmEmbedding
from utils.protein_sequences import compute_cmaps_distance, get_max_hamming_msa, get_blosum_score

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='/scratch/external/gcarbone/msa/hhfiltered/', type=str, 
	help="Datasets path. Choose `msa/` or `msa/hhfiltered/`.")
parser.add_argument("--missense_dir", default='/scratch/external/gcarbone/missense/', type=str, help="Missense dataset path.")
parser.add_argument("--dataset", default='PF00533', type=str, help="Dataset name")
parser.add_argument("--out_dir", default='/fast/external/gcarbone/adversarial-protein-sequences_out/', type=str, 
	help="Output data path.")
parser.add_argument("--max_tokens", default=None, type=eval, 
	help="Optionally cut sequences to maximum number of tokens. None does not cut sequences.")
parser.add_argument("--n_sequences", default=30, type=eval, 
	help="Number of sequences from the chosen dataset. None loads all sequences.")
parser.add_argument("--min_filter", default=100, type=eval, help="Minimum number of sequences selected for the filtered MSA.")

parser.add_argument("--n_substitutions", default=3, type=int, help="Number of token substitutions in the original sequence.")

parser.add_argument("--token_selection", default='max_attention', type=str, 
	help="Method used to select most relevant token idxs. Choose 'max_attention', 'max_entropy' or 'min_entropy'.")
parser.add_argument("--target_attention", default='all_layers', type=str, 
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


# out_path = os.path.join(args.out_dir, f"msa/msa_{args.dataset}_seqs={args.n_sequences}_max_toks={args.max_tokens}_{args.token_selection}_subst={args.n_substitutions}_minFilter={args.min_filter}/")
# out_plots_path = os.path.join(out_path, "plots/")
# out_data_path = os.path.join(out_path, "data/")
# out_missense_filename = f"msa_{args.dataset}_minFilter={args.min_filter}"
# out_missense_path = os.path.join(args.out_dir, "missense/")
# os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
# os.makedirs(os.path.dirname(out_missense_path), exist_ok=True)

perturbations_keys = ['masked_pred','max_entropy','min_dist','max_dist'] 

if args.load:

	raise NotImplementedError

	# missense_evaluation_df = pd.read_csv(os.path.join(out_missense_path, out_missense_filename+"_missense_eval.csv"))
	# missense_cmap_df = pd.read_csv(os.path.join(out_missense_path, out_missense_filename+"_missense_cmaps.csv"))

else:

	esm_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
	batch_converter = alphabet.get_batch_converter()
	n_layers = esm_model.args.layers
	esm_model = esm_model.to(args.device)

	emb_model = MsaEsmEmbedding(original_model=esm_model, alphabet=alphabet).to(args.device)
	emb_model = emb_model.to(args.device)

	atk = SequenceAttack(original_model=esm_model, embedding_model=emb_model, alphabet=alphabet)

	missense_df = pd.read_csv(os.path.join(args.missense_dir, f"missense_mutations_{args.dataset}.csv"))

	missense_evaluation_df = pd.DataFrame()
	missense_cmap_df = pd.DataFrame()

	for row_idx, row in missense_df.iterrows():

		blosum_score = get_blosum_score(row['original_token'],row['mutated_token'])

		print(f"\n=== Mutation {row['mutation_name']} ===")
		print(f"\nmutation_idx = {row['mutation_idx']}\toriginal_token = {row['original_token']}\tmutated_token = {row['mutated_token']}\tblosum_score = {blosum_score}")

		name = row['pfam_name']
		original_sequence = row['original_sequence']
		mutated_idxs = [row['mutation_idx']] # single amino acid mutation (eventually update for other families)

		assert len(original_sequence)==len(row['mutated_sequence'])
		assert original_sequence[row['mutation_idx']]==row['original_token']
		assert row['mutated_sequence'][row['mutation_idx']]==row['mutated_token']

		seq_filename = name.replace('/','_')
		original_sequence = original_sequence.replace('-','')

		msa, max_tokens = load_msa(
			filepath=f"{args.missense_dir}hhfiltered/hhfiltered_{args.dataset}_filter={args.min_filter}", 
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

		### attack sequence

		target_token_idxs, target_tokens_attention = atk.choose_target_token_idxs(token_selection=args.token_selection, 
			n_token_substitutions=args.n_substitutions, msa=msa, batch_tokens=batch_tokens, 
			target_attention=args.target_attention, verbose=args.verbose)

		target_idx_accuracy = np.sum([idx in target_token_idxs for idx in mutated_idxs])/len(mutated_idxs)

		signed_gradient, loss = atk.compute_loss_gradient(original_sequence=original_sequence, batch_tokens=batch_tokens,
			target_token_idxs=target_token_idxs, first_embedding=first_embedding, loss_method=args.loss_method)

		atk_df, _ = atk.incremental_attack(name=name, original_sequence=original_sequence, 
			original_batch_tokens=batch_tokens, msa=msa, target_token_idxs=target_token_idxs, 
			target_tokens_attention=target_tokens_attention,
			first_embedding=first_embedding, signed_gradient=signed_gradient, 
			perturbations_keys=perturbations_keys, verbose=args.verbose)

		### evaluate against missense mutations

		missense_row = atk.evaluate_missense(missense_row=row,  msa=msa, original_embedding=first_embedding, 
			signed_gradient=signed_gradient, adversarial_df=atk_df, perturbations_keys=perturbations_keys, 
			verbose=args.verbose)

		# print('original_sequence', missense_row['original_sequence'])
		# print('mutated_sequence', missense_row['mutated_sequence'])
		# print('masked_pred_sequence', atk_df['masked_pred_sequence'])
		# print('max_cos_sequence', atk_df['max_cos_sequence'])
		# print('min_dist_sequence', atk_df['min_dist_sequence'])
		# print('max_dist_sequence', atk_df['max_dist_sequence'])

		missense_row['target_idx_accuracy'] = target_idx_accuracy.item()
		missense_evaluation_df = missense_evaluation_df.append(missense_row, ignore_index=True)

		### compute cmaps distances against missense mutations

		perturbed_sequences_dict = {key:atk_df[f'{key}_sequence'].unique()[0] for key in perturbations_keys}

		cmap_df = compute_cmaps_distance(model=esm_model, alphabet=alphabet, 
			original_sequence=row['mutated_sequence'], sequence_name=row['mutation_name'], 
			perturbed_sequences_dict=perturbed_sequences_dict,
			cmap_dist_lbound=args.cmap_dist_lbound, cmap_dist_ubound=args.cmap_dist_ubound)

		missense_cmap_df = pd.concat([missense_cmap_df, cmap_df], ignore_index=True)

	# missense_evaluation_df.to_csv(os.path.join(out_missense_path, out_missense_filename+"_missense_eval.csv"))
	# missense_cmap_df.to_csv(os.path.join(out_missense_path, out_missense_filename+"_missense_cmaps.csv"))

print("\nmissense_evaluation_df:\n", missense_evaluation_df.keys())
print("\nmissense_cmap_df:\n", missense_cmap_df.keys())




