import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_tokens_heatmap(df, filepath=None, filename=None):

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=df, x="target_token_idx", hue="new_token", multiple="stack", discrete=True)

    print(df['target_token_idx'])
    print(df['new_token'])

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig


def plot_cmap_distances(cmap_distances, filepath=None, filename=None):

    fig, ax = plt.subplots(figsize=(10, 6))
    diagonal_idxs = np.repeat([range(1,cmap_distances.shape[1]+1)], repeats=cmap_distances.shape[0], axis=0)

    sns.lineplot(x=diagonal_idxs.flatten(), y=cmap_distances.flatten())
    ax.set(xlabel='k = len(sequence)-diag_idx', ylabel='l2 distance', 
        title='dist. bw original and adversarial contact maps')

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_tokens_attention(sequence, attentions, layer_idx, filepath=None, filename=None):

    layer_attention_scores = attentions[:,layer_idx-1,:,:,:].squeeze().detach().cpu().numpy()

    fig = plt.figure(figsize=(20, 20))

    for idx, scores in enumerate(layer_attention_scores):
        scores_np = np.array(scores)
        ax = fig.add_subplot(5, 4, idx+1)
        im = ax.imshow(scores, cmap='viridis')

        ax.set_xticks(range(len(sequence)))
        ax.set_yticks(range(len(sequence)))

        fontdict = {'fontsize': 10}
        ax.set_xticklabels(sequence, fontdict=fontdict)
        ax.set_yticklabels(sequence, fontdict=fontdict, rotation=90)
        ax.set_xlabel('{} {}'.format('Head', idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig


def plot_representations_norms(norms_mat, sequence, target_token_idx, filepath=None, filename=None):

    n_layers = norms_mat.shape[0]

    fig, ax = plt.subplots(figsize=(10,7))
    ax = sns.heatmap(norms_mat, linewidth=0.2)

    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(n_layers))

    fontdict = {'fontsize': 10}
    ax.set_xticklabels(sequence, fontdict=fontdict)
    ax.set_yticklabels(list(range(n_layers)), fontdict=fontdict, rotation=90)

    x, y, w, h = target_token_idx, 0, 1, n_layers
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2, clip_on=False))
    ax.tick_params(length=0)

    plt.xlabel('Tokens')
    plt.ylabel('Layers')

    plt.tight_layout()
    plt.show()  

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_contact_maps(original_contacts, adversarial_contacts, filepath=None, filename=None):

    fig, ax = plt.subplots(figsize=(10, 4), ncols=3)
    ax[0].imshow(original_contacts, cmap="Blues")
    ax[1].imshow(adversarial_contacts, cmap="Blues")
    ax[2].imshow(original_contacts-adversarial_contacts, cmap="Blues")

    ax[0].set_xlabel('original')
    ax[1].set_xlabel('adversarial')
    ax[2].set_xlabel('orig-adv')

    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig