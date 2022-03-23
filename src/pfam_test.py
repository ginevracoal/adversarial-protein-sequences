import os
import esm
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from paths import *
from embedding_model import EmbModel
from sequence_attack import SequenceAttack
from plot_utils import plot_cmap_distances, plot_cosine_distances, plot_tokens_hist
from data_utils import filter_pfam

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='fastaPF00004', type=str, help="Dataset name")
parser.add_argument("--n_sequences", default=None, type=int, help="Number of sequences from the chosen dataset")
parser.add_argument("--n_token_substitutions", default=2, type=int, help="Number of token substitutions in the original \
    sequence")
parser.add_argument("--cmap_dist_lbound", default=100, type=int, help='Lower bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--cmap_dist_ubound", default=20, type=int, help='Upper bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute')
parser.add_argument("--verbose", default=False, type=eval)
args = parser.parse_args()

filename = args.dataset
df_filename = filename+f"_{args.n_token_substitutions}tokens.csv" if args.n_sequences is None \
    else filename+f"_{args.n_token_substitutions}tokens_{args.n_sequences}seq.csv"
cmap_df_filename = filename+f"_{args.n_token_substitutions}tokens_cmap.csv" if args.n_sequences is None \
    else filename+f"_{args.n_token_substitutions}tokens_{args.n_sequences}seq_cmap.csv"

perturbations_keys = ['adv','safe','min_dist','max_dist']

if args.load:

    df = pd.read_csv(os.path.join(out_data_path, df_filename))
    cmap_df = pd.read_csv(os.path.join(out_data_path, cmap_df_filename))

else:

    esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    n_layers = esm1_model.args.layers

    esm1_model = esm1_model.to(args.device)

    data = filter_pfam(max_tokens=esm1_model.args.max_tokens, filepath=pfam_path, filename=filename)

    if args.n_sequences is not None:
        data = data[:args.n_sequences]

    df = pd.DataFrame()
    cmap_df = pd.DataFrame()

    for seq_idx, single_sequence_data in tqdm(enumerate(data), total=len(data)):

        name, original_sequence = single_sequence_data
        batch_labels, batch_strs, batch_tokens = batch_converter([single_sequence_data])

        batch_tokens = batch_tokens.to(args.device)

        with torch.no_grad():
            results = esm1_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

        first_embedding = results["representations"][0].to(args.device)

        model = EmbModel(esm1_model, alphabet).to(args.device)
        model.check_correctness(original_model=esm1_model, batch_tokens=batch_tokens)

        atk = SequenceAttack(original_model=esm1_model, embedding_model=model, alphabet=alphabet)

        target_token_idxs, repr_norms_matrix = atk.choose_target_token_idxs(batch_tokens=batch_tokens, 
            n_token_substitutions=args.n_token_substitutions, verbose=args.verbose)

        signed_gradient, loss = atk.perturb_embedding(first_embedding=first_embedding)

        atk_df = atk.attack_sequence(original_sequence=original_sequence, target_token_idxs=target_token_idxs, 
            first_embedding=first_embedding, signed_gradient=signed_gradient, verbose=args.verbose)

        df = pd.concat([df, atk_df], ignore_index=True)

        atk.original_model.to(args.device)
        original_contact_map = atk.compute_contact_maps(sequence=original_sequence)

        min_k_idx, max_k_idx = len(original_sequence)-args.cmap_dist_lbound, len(original_sequence)-args.cmap_dist_ubound
        for k_idx, k in enumerate(torch.arange(min_k_idx, max_k_idx, 1)):

            row_list = [['name', name],['k',k_idx]]
            for key in perturbations_keys:

                topk_original_contacts = torch.triu(original_contact_map, diagonal=k)
                new_contact_map = atk.compute_contact_maps(sequence=atk_df[f'{key}_sequence'].unique()[0])
                topk_new_contacts = torch.triu(new_contact_map, diagonal=k)

                cmap_distance = torch.norm((topk_original_contacts-topk_new_contacts).flatten()).item()
                row_list.append([f'{key}_cmap_dist',cmap_distance])

            cmap_df = cmap_df.append(dict(row_list), ignore_index=True)
                
    os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
    df.to_csv(os.path.join(out_data_path, df_filename))
    cmap_df.to_csv(os.path.join(out_data_path, cmap_df_filename))


print(df)
print(df.columns)
print(cmap_df)

plot_tokens_hist(df, keys=perturbations_keys, filepath=plots_path, filename=filename+"_tokens_hist")
plot_cosine_distances(df, x='adv_token', keys=['adv','safe'], filepath=plots_path, filename=filename+"_cosine_distances")
plot_cmap_distances(cmap_df, keys=perturbations_keys, filepath=plots_path, filename=filename+"_cmap_distances")