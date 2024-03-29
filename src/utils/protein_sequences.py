import torch
import numpy as np
import pandas as pd
from Bio.SubsMat import MatrixInfo

DEBUG=True


def get_max_hamming_msa(reference_sequence, msa, max_size):

    if len(msa) <= args.max_size:
      raise ValueError("Choose n_sequences > max_size")

    if len(reference_sequence)!=2:
        raise AttributeError("reference_sequence should be a couple (name, sequence)")

    def hamming_distance(s1, s2):
        if len(s1) != len(s2):
            raise ValueError("Lengths are not equal!")
        return sum(ch1 != ch2 for ch1,ch2 in zip(s1,s2))

    hamming_distances = []
    for _, sequence_data in enumerate(msa):
        hamming_distances.append(hamming_distance(s1=reference_sequence[1], s2=sequence_data[1]))

    n_sequences = min(len(msa), max_size)
    topk_idxs = torch.topk(torch.tensor(hamming_distances), k=n_sequences).indices.cpu().detach().numpy()
    max_hamming_msa = [reference_sequence] + [msa[idx] for idx in topk_idxs]

    assert len(max_hamming_msa)==n_sequences+1
    return max_hamming_msa

def get_blosum_score(token1, token2, penalty=2):
    blosum = MatrixInfo.blosum62
    try:
        return blosum[(token1,token2)]
    except ValueError:
        return blosum[(token2,token1)]
    except:
        return penalty*min(blosum.values()) # very unlikely substitution 

def compute_blosum_distance(seq1, seq2, target_token_idxs=None, verbose=False):

    if target_token_idxs is None:
        target_token_idxs = range(len(seq1))

    assert len(seq1)==len(seq2)

    if verbose:
        for i in target_token_idxs:
            print(f"token={i}\t{seq1[i]} -> {seq2[i]}\tblosum subst score = {get_blosum_score(seq1[i],seq1[i])-get_blosum_score(seq1[i],seq2[i])}")

    blosum_distance = sum([get_blosum_score(seq1[i],seq1[i])-get_blosum_score(seq1[i],seq2[i]) for i in target_token_idxs])
    return blosum_distance

def get_contact_map(model, alphabet, sequence):
    device = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([('seq',sequence)])
    contact_map = model.predict_contacts(batch_tokens.to(device))[0]
    return contact_map

def compute_cmaps_distance(model, alphabet, sequence_name, original_sequence, perturbed_sequence, k=6, p=1):
    # k has to be > 0
    # k=1 computes the entire triu cmap
    # parameter k restricts the comptuation of submatrices to residues that are at least k positions apart

    original_contact_map = get_contact_map(model=model, alphabet=alphabet, sequence=original_sequence)
    topk_original_contacts = torch.triu(original_contact_map, diagonal=k)
    new_contact_map = get_contact_map(model=model, alphabet=alphabet, sequence=perturbed_sequence)
    topk_new_contacts = torch.triu(new_contact_map, diagonal=k)
    cmap_distance = torch.norm((topk_original_contacts-topk_new_contacts).flatten(), p=p).item()

    return cmap_distance

def compute_cmaps_distance_df(model, alphabet, perturbed_sequences_dict, original_sequence, sequence_name, 
    cmap_dist_lbound=0.2, cmap_dist_ubound=0.8, p=2, verbose=False):

    cmap_df = pd.DataFrame()

    for k_idx, k in enumerate(np.arange(0, len(original_sequence), 1)):

        row_list = [['name', sequence_name],['k',k_idx]]
        for key, perturbed_sequence in perturbed_sequences_dict.items():

            cmap_distance = compute_cmaps_distance(model=model, alphabet=alphabet, sequence_name=sequence_name, 
                original_sequence=original_sequence, perturbed_sequence=perturbed_sequence, k=k, p=p)
            row_list.append([f'{key}_cmap_dist', cmap_distance])

        cmap_df = cmap_df.append(dict(row_list), ignore_index=True)

    print(f"\n\ncmap_distances at k=0:\n\n{cmap_df[cmap_df['k']==0]}\n")
    return cmap_df

def compute_cmaps_distance_df_protherm(model, alphabet, perturbation, perturbed_sequences_dict, original_sequence, sequence_name, 
    ddg, cmap_dist_lbound=0.2, cmap_dist_ubound=0.8, p=1, verbose=False):

    cmap_df = pd.DataFrame()

    for k_idx, k in enumerate(np.arange(0, len(original_sequence), 1)):

        row_list = [['name', sequence_name],['k',k_idx],['perturbation',perturbation],['ddg',ddg]]
        for key, perturbed_sequence in perturbed_sequences_dict.items():

            cmap_distance = compute_cmaps_distance(model=model, alphabet=alphabet, sequence_name=sequence_name, 
                original_sequence=original_sequence, perturbed_sequence=perturbed_sequence, k=k, p=p)
            row_list.append([f'cmaps_distance', cmap_distance])

        cmap_df = cmap_df.append(dict(row_list), ignore_index=True)

    print(f"\n\ncmap_distances at k=0:\n\n{cmap_df[cmap_df['k']==0]}\n")
    return cmap_df
