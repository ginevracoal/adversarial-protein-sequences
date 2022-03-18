import os
import esm
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from re import search, IGNORECASE

from embedding_model import EmbModel
from sequence_attack import SequenceAttack
from plot_utils import plot_tokens_attention, plot_representations_norms, plot_contact_maps
from data_utils import filter_pfam

plots_path='plots/'
filename='fastaPF00001'
embedding_distance = 'cosine'
out_data_path='data/out/'

# df = pd.read_csv(os.path.join(out_data_path, filename+".csv"))
# print(df)
# print(df['A0A1J1IDL3'])
# exit()

data = filter_pfam(filepath="data/pfam/", filename=filename)

esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
n_layers = esm1_model.args.layers


output_dict = {}

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

    signed_gradient = atk.perturb_embedding(first_embedding=first_embedding)

    adversarial_sequence, embeddings_distance = atk.attack_sequence(original_sequence=original_sequence, target_token_idx=target_token_idx, 
        first_embedding=first_embedding, signed_gradient=signed_gradient, embedding_distance=embedding_distance)

    # contact maps

    original_contacts, adversarial_contacts = atk.compute_contact_maps(original_sequence, adversarial_sequence)
    l2_distance = torch.norm((original_contacts-adversarial_contacts).flatten()).item()
    print("\nl2 distance bw contact maps =", l2_distance)

    output_dict.update({name: {'sequence':original_sequence, 'target_token_idx':target_token_idx,
                            'signed_gradient':signed_gradient, 'adversarial_sequence':adversarial_sequence,
                            'embeddings_distance':embeddings_distance,
                            'original_contacts':original_contacts, 'adversarial_contacts':adversarial_contacts,
                            'l2_dist_contact_maps':l2_distance}})

df = pd.DataFrame(output_dict)
print(df)

os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
df.to_csv(os.path.join(out_data_path, filename+".csv"))