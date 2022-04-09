import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_cosine_similarity(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(5*len(keys), 4), ncols=len(keys), sharey=True)

    for i in range(len(keys)):
        axis = ax if len(keys)==1 else ax[i]
        df = df.sort_values(f'{keys[i]}_token') 
        sns.stripplot(data=df, x=f'{keys[i]}_token', y=f'{keys[i]}_similarity', dodge=True, ax=axis)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
    
    plt.close()
    return fig

def plot_tokens_hist(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    df['perc_token_idx'] = df.apply(lambda row: row['target_token_idx']/len(row['original_sequence']), axis=1)
    sns.histplot(data=df, x="perc_token_idx", legend=None)#, multiple="stack", hue="adv_token")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
    
    plt.close()

    assert len(keys)==4
    fig, ax = plt.subplots(figsize=(10, 7), nrows=2, ncols=2, sharey=True)
    jitter = 0.1

    df = df.sort_values(f'{keys[0]}_token') 
    sns.stripplot(data=df, y="perc_token_idx", x=f"{keys[0]}_token", dodge=True, ax=ax[0,0], jitter=jitter)
    df = df.sort_values(f'{keys[1]}_token') 
    sns.stripplot(data=df, y="perc_token_idx", x=f"{keys[1]}_token", dodge=True, ax=ax[1,0], jitter=jitter)
    df = df.sort_values(f'{keys[2]}_token') 
    sns.stripplot(data=df, y="perc_token_idx", x=f"{keys[2]}_token", dodge=True, ax=ax[0,1], jitter=jitter)
    df = df.sort_values(f'{keys[3]}_token') 
    sns.stripplot(data=df, y="perc_token_idx", x=f"{keys[3]}_token", dodge=True, ax=ax[1,1], jitter=jitter)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+f"_token_split.png"))
        plt.close()

    return fig


def plot_cmap_distances(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlabel='k = len(sequence)-diag_idx', ylabel='l2 distance', 
        title='dist. bw original and perturbed contact maps')

    for key in keys:
        sns.lineplot(x=df['k'], y=df[f'{key}_cmap_dist'], label=key)

    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_blosum_distances(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    df = df[['original_sequence']+[f"{key}_blosum" for key in keys]]
    df = df.melt(id_vars=['original_sequence'], var_name="key", value_name="blosum")

    ax = sns.histplot(x=df["blosum"], hue=df["key"], kde=False, multiple="stack")
    plt.tight_layout()
    plt.show()

    if filepath is not None and filename is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fig.savefig(os.path.join(filepath, filename+".png"))
        plt.close()

    return fig

def plot_confidence(df, keys, filepath=None, filename=None):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    plt.yscale('log')

    for key in keys:
        print(df[f"{key}_confidence"].describe())
        # ax = sns.histplot(x=df[f"{key}_confidence"], label=key)
    
    # df = df[['original_sequence']+[f"{key}_confidence" for key in keys]]

    # df = df.melt(id_vars=['original_sequence'], 
    #                 var_name="key", 
    #                 value_name="confidence")
    # print(df['confidence'].min(), df["confidence"].mean())
    # ax = sns.displot(x=df["confidence"], hue=df["key"], kde=False)#, kind="kde")

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


def plot_representations_norms(norms_mat, sequence, target_token_idxs, filepath=None, filename=None):

    n_layers = norms_mat.shape[0]

    fig, ax = plt.subplots(figsize=(10,7))
    ax = sns.heatmap(norms_mat, linewidth=0.2)

    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(n_layers))

    fontdict = {'fontsize': 10}
    ax.set_xticklabels(sequence, fontdict=fontdict)
    ax.set_yticklabels(list(range(n_layers)), fontdict=fontdict, rotation=90)

    for target_token_idx in target_token_idxs:
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