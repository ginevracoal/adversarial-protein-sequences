import os
import esm
import torch
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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='fastaPF00004', type=str, help="Dataset name")
parser.add_argument("--n_sequences", default=None, type=int, help="Number of sequences from the chosen dataset")
parser.add_argument("--embedding_distance", default='cosine', type=str, help="Distance measure between representations \
    in the first embedding space")
parser.add_argument("--cmap_dist_lbound", default=100, type=int, help='Lower bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--cmap_dist_ubound", default=20, type=int, help='Upper bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--device", default='cpu', type=str, help='Device')
parser.add_argument("--load", default=False, type=eval, help='If True load else compute')
args = parser.parse_args()

filename = args.dataset
df_filename = filename+".csv" if args.n_sequences is None else filename+f"_{args.n_sequences}seq.csv"

if args.load:

    df = pd.read_csv(os.path.join(out_data_path, df_filename))
    # print(df.describe())

else:

    esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    n_layers = esm1_model.args.layers

    esm1_model = esm1_model.to(args.device)

    data = filter_pfam(max_tokens=esm1_model.args.max_tokens, filepath=pfam_path, filename=filename)

    if args.n_sequences is not None:
        data = data[:args.n_sequences]

    df = pd.DataFrame()

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

        new_token, adversarial_sequence, baseline_sequence, embeddings_distance = atk.attack_sequence(original_sequence=original_sequence, 
            target_token_idx=target_token_idx, first_embedding=first_embedding, signed_gradient=signed_gradient, 
            embedding_distance=args.embedding_distance)

        original_contacts, adversarial_contacts, baseline_contacts = atk.compute_contact_maps(original_sequence, adversarial_sequence, baseline_sequence)
        
        for k_idx, k in enumerate(torch.arange(len(original_sequence)-args.cmap_dist_lbound, 
            len(original_sequence)-args.cmap_dist_ubound, 1)):

            topk_original_contacts = torch.triu(original_contacts, diagonal=k)
            topk_adversarial_contacts = torch.triu(adversarial_contacts, diagonal=k)
            topk_baseline_contacts = torch.triu(baseline_contacts, diagonal=k)

            adv_cmap_distance = torch.norm((topk_original_contacts-topk_adversarial_contacts).flatten()).item()
            baseline_cmap_distance = torch.norm((topk_original_contacts-topk_baseline_contacts).flatten()).item()
            # print(f"l2 distance bw contact maps diag {k} = {cmap_distance}")

            df = df.append({'name':name, 'sequence':original_sequence, 'target_token_idx':target_token_idx, 
                'new_token':new_token, 'adversarial_sequence':adversarial_sequence, 
                'embeddings_distance':embeddings_distance, 'loss':loss.item(),
                'k':k_idx, 'adv_cmap_distance':adv_cmap_distance, 'baseline_cmap_distance':baseline_cmap_distance
                }, ignore_index=True)

    os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
    df.to_csv(os.path.join(out_data_path, df_filename))

plot_tokens_hist(df, filepath=plots_path, filename=filename+"_tokens_hist")
plot_embeddings_distances(df, filepath=plots_path, filename=filename+"_embeddings_distances")
plot_cmap_distances(df, filepath=plots_path, filename=filename+"_cmap_distances")