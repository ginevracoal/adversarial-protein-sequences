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
from plot_utils import plot_cmap_distances, plot_tokens_heatmap
from data_utils import filter_pfam, save_to_pickle, load_from_pickle


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='fastaPF00001', type=str, help="Dataset name")
parser.add_argument("--embedding_distance", default='cosine', type=str, help="Distance measure between representations \
    in the first embedding space")
parser.add_argument("--cmap_dist_lbound", default=100, type=int, help='Lower bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--cmap_dist_ubound", default=20, type=int, help='Upper bound for upper triangular matrix of long \
    range contacts')
parser.add_argument("--load", default=False, type=eval, help='If True load else compute')
parser.add_argument("--debug", default=False, type=eval, help='Debugging mode')
args = parser.parse_args()

filename = args.dataset

if args.load:

    df = pd.read_csv(os.path.join(out_data_path, args.dataset+".csv"))
    print(df.describe())

    cmap_distances = load_from_pickle(filepath=out_data_path, 
        filename=filename+f"_cmap_dist_{args.cmap_dist_lbound}_{args.cmap_dist_ubound}")

else:
    data = filter_pfam(filepath=pfam_path, filename=filename)

    if args.debug:
        data = data[:2]

    esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    n_layers = esm1_model.args.layers

    df = pd.DataFrame()

    cmap_distances = []

    for seq_idx, single_sequence_data in tqdm(enumerate(data), total=len(data)):

        name, original_sequence = single_sequence_data
        batch_labels, batch_strs, batch_tokens = batch_converter([single_sequence_data])

        with torch.no_grad():
            results = esm1_model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=True)

        first_embedding = results["representations"][0]

        # istantiate models

        model = EmbModel(esm1_model, alphabet)
        model.check_correctness(original_model=esm1_model, batch_tokens=batch_tokens)

        atk = SequenceAttack(original_model=esm1_model, embedding_model=model, alphabet=alphabet)

        target_token_idx, repr_norms_matrix = atk.choose_target_token_idx(batch_tokens=batch_tokens)
        print("\ntarget_token_idx =", target_token_idx)

        signed_gradient, loss = atk.perturb_embedding(first_embedding=first_embedding)

        new_token, adversarial_sequence, embeddings_distance = atk.attack_sequence(original_sequence=original_sequence, 
            target_token_idx=target_token_idx, first_embedding=first_embedding, signed_gradient=signed_gradient, 
            embedding_distance=args.embedding_distance)

        df = df.append({'name':name, 'sequence':original_sequence, 'target_token_idx':target_token_idx, 
            'new_token':new_token, 'adversarial_sequence':adversarial_sequence, 
            'embeddings_distance':embeddings_distance, 'loss':loss.item()}, ignore_index=True)

        # contact maps
        original_contacts, adversarial_contacts = atk.compute_contact_maps(original_sequence, adversarial_sequence)
        
        topk_cmap_distances=[]

        for k in torch.arange(len(original_sequence)-args.cmap_dist_lbound, 
            len(original_sequence)-args.cmap_dist_ubound, 1):

            topk_original_contacts = torch.triu(original_contacts, diagonal=k)
            topk_adversarial_contacts = torch.triu(adversarial_contacts, diagonal=k)

            cmap_distance = torch.norm((topk_original_contacts-topk_adversarial_contacts).flatten()).item()
            print(f"l2 distance bw contact maps diag {k} = {cmap_distance}")

            topk_cmap_distances.append(cmap_distance)

        cmap_distances.append(topk_cmap_distances)

    cmap_distances = np.array(cmap_distances)
    save_to_pickle(cmap_distances, filepath=out_data_path, 
        filename=filename+f"_cmap_dist_{args.cmap_dist_lbound}_{args.cmap_dist_ubound}")

    print(df)
    os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
    df.to_csv(os.path.join(out_data_path, filename+".csv"))

plot_tokens_heatmap(df, filepath=plots_path, filename=filename+"_tokens_heatmap")
plot_cmap_distances(cmap_distances, filepath=plots_path, filename=filename+"_cmap_distances")
