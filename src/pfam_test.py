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
from plot_utils import plot_cmap_distances, plot_embeddings_distances, plot_tokens_hist
from data_utils import filter_pfam

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='fastaPF00004', type=str, help="Dataset name")
parser.add_argument("--n_sequences", default=None, type=int, help="Number of sequences from the chosen dataset")
parser.add_argument("--cmap_dist_lbound", default=100, type=int, help='Lower bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--cmap_dist_ubound", default=20, type=int, help='Upper bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'")
parser.add_argument("--load", default=False, type=eval, help='If True load else compute')
args = parser.parse_args()

filename = args.dataset
df_filename = filename+".csv" if args.n_sequences is None else filename+f"_{args.n_sequences}seq.csv"
cmap_df_filename = filename+"_cmap.csv" if args.n_sequences is None else filename+f"_cmap_{args.n_sequences}seq.csv"

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

        target_token_idx, repr_norms_matrix = atk.choose_target_token_idx(batch_tokens=batch_tokens)
        print("\ntarget_token_idx =", target_token_idx)

        signed_gradient, loss = atk.perturb_embedding(first_embedding=first_embedding)

        atk_dict = atk.attack_sequence(original_sequence=original_sequence, 
            target_token_idx=target_token_idx, first_embedding=first_embedding, signed_gradient=signed_gradient)

        df = df.append({
            'name':name, 'sequence':original_sequence, 
            'target_token_idx':target_token_idx, 'loss':loss.item(),
            'adv_token':atk_dict['adv_token'], 
            'adv_sequence':atk_dict['adv_sequence'], 
            'adv_cosine_distance':atk_dict['adv_cosine_distance'], 
            'safe_token':atk_dict['safe_token'], 
            'safe_sequence':atk_dict['safe_sequence'], 
            'safe_cosine_distance':atk_dict['safe_cosine_distance'],
            'min_dist_token':atk_dict['min_dist_token'], 
            'min_dist_sequence':atk_dict['min_dist_sequence'],
            'min_euclidean_dist':atk_dict['min_euclidean_dist'], 
            'max_dist_token':atk_dict['max_dist_token'], 
            'max_dist_sequence':atk_dict['max_dist_sequence'],
            'max_euclidean_dist':atk_dict['max_euclidean_dist']
            }, ignore_index=True)

        atk.original_model.to(args.device)
        original_contact_map = atk.compute_contact_maps(sequence=original_sequence)

        min_k_idx, max_k_idx = len(original_sequence)-args.cmap_dist_lbound, len(original_sequence)-args.cmap_dist_ubound
        for k_idx, k in enumerate(torch.arange(min_k_idx, max_k_idx, 1)):

            for key in ['adv','safe','min_dist','max_dist']:

                topk_original_contacts = torch.triu(original_contact_map, diagonal=k)
                new_contact_map = atk.compute_contact_maps(sequence=atk_dict[f'{key}_sequence'])
                topk_new_contacts = torch.triu(new_contact_map, diagonal=k)

                cmap_distance = torch.norm((topk_original_contacts-topk_new_contacts).flatten()).item()
                cmap_df = cmap_df.append({'name':name, 'k':k_idx, f'{key}_cmap_dist':cmap_distance}, ignore_index=True)
                
                # print(f"l2 distance bw contact maps diag {k} = {cmap_distance}")

    os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
    df.to_csv(os.path.join(out_data_path, df_filename))
    cmap_df.to_csv(os.path.join(out_data_path, cmap_df_filename))

plot_tokens_hist(df, filepath=plots_path, filename=filename+"_tokens_hist")
plot_embeddings_distances(df, x='adv_token', y='adv_cosine_distance', filepath=plots_path, filename=filename+"_embeddings_distances")
plot_embeddings_distances(df, x='adv_token', y='safe_cosine_distance', filepath=plots_path, filename=filename+"_safe_embeddings_distances")
plot_cmap_distances(cmap_df, filepath=plots_path, filename=filename+"_cmap_distances")