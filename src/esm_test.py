import os
import esm
import torch
import matplotlib.pyplot as plt
from embedding_model import EmbModel

esm1_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()

data = [
    ("protein", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = esm1_model(batch_tokens, repr_layers=[0], return_contacts=True)
    
print("\nresults.keys() =", results.keys())

token_representations = results["representations"][0]
print("\ntoken_representations.shape =", token_representations.shape)

first_embedding = token_representations
model = EmbModel(esm1_model, alphabet)

# check output logits are equal
assert torch.all(torch.eq(model(first_embedding)['logits'], results['logits']))

