import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_embeddings_distances(df, x, y, filepath=None, filename=None):
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(8, 5))

    tokens_list = sorted(df[x].unique())
    sns.stripplot(data=df, x=x, y=y, dodge=True)
    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
    
    plt.close()
    return fig

def plot_tokens_hist(df, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    df['perc_token_idx'] = df.apply(lambda row: row['target_token_idx']/len(row['sequence']), axis=1)
    sns.histplot(data=df, x="perc_token_idx", legend=None)#, multiple="stack", hue="adv_token")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
    
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    df = df.sort_values('adv_token')
    sns.stripplot(data=df, y="perc_token_idx", x="adv_token", dodge=True)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+"_split.png"))
        plt.close()

    return fig


def plot_cmap_distances(df, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(x=df['k'], y=df['adv_cmap_dist'], label='adversarial')
    sns.lineplot(x=df['k'], y=df['safe_cmap_dist'], label='safe')
    sns.lineplot(x=df['k'], y=df['min_dist_cmap_dist'], label='min_dist')
    sns.lineplot(x=df['k'], y=df['max_dist_cmap_dist'], label='max_dist')
    ax.set(xlabel='k = len(sequence)-diag_idx', ylabel='l2 distance', 
        title='dist. bw original and perturbed contact maps')

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