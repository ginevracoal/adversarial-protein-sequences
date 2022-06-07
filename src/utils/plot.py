import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FONT_SIZE=12


def plot_cosine_similarity(df, keys=['max_cos'], filepath=None, filename=None):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	fig, ax = plt.subplots(figsize=(5*len(keys), 4), ncols=len(keys), sharey=True)

	for i in range(len(keys)):
		axis = ax if len(keys)==1 else ax[i]
		df = df.sort_values(f'{keys[i]}_token') 
		sns.stripplot(data=df, x=f'{keys[i]}_token', y=f'{keys[i]}', dodge=True, ax=axis)
		ax.set_ylabel(r'Cosine similarity $(z-\tilde{z}, z-z_I(C_I))$')

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_cosine_similarity.png"))
		plt.close()

	return fig

def plot_tokens_hist(df, keys, split=True, filepath=None, filename=None):
	df['perc_token_idx'] = df.apply(lambda row: row['target_token_idx']/len(row['original_sequence']), axis=1)

	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	if split:

		### split by perturbations_keys
		assert len(keys)>1
		fig, ax = plt.subplots(figsize=(10, 7), nrows=2, ncols=2, sharey=True)
		jitter = 0.1

		for key, axis in ((keys[0],ax[0,0]), (keys[1],ax[1,0]), (keys[2],ax[0,1]), (keys[3],ax[1,1])):
			df = df.sort_values(f'{key}_token') 
			# sns.swarmplot(data=df, y="perc_token_idx", x=f"{key}_token", ax=axis)
			sns.stripplot(data=df, y="perc_token_idx", x=f"{key}_token", dodge=True, ax=axis, jitter=jitter)
			axis.set_ylabel('token idx percentile w.r.t. seq length')

	else:
		### histogram of token idx percentiles
		fig, ax = plt.subplots(figsize=(8, 5))
		sns.histplot(data=df, x="perc_token_idx", legend=None)
		plt.yscale('log')

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+f"_tokens_hist.png"))
		plt.close()

	return fig

def plot_token_substitutions(df, keys, filepath=None, filename=None):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	for key in keys:

		fig, ax = plt.subplots(figsize=(10, 7))

		subst_counts = df.groupby(['orig_token', f'{key}_token'], as_index=False).size()
		subst_counts = subst_counts.rename(columns={"size": "n_substitutions"})

		df_heatmap = subst_counts.pivot_table(values='n_substitutions', columns='orig_token', index=f'{key}_token')
		sns.heatmap(df_heatmap, annot=True, cmap="rocket_r")

		plt.tight_layout()
		plt.show()

		if filepath is not None and filename is not None:
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			fig.savefig(os.path.join(filepath, filename+f"_substitutions_{key}_token.png"))
			plt.close()

def plot_cmap_distances(df, keys, filepath=None, filename=None):
	matplotlib.rc('font', **{'size': FONT_SIZE})    
	sns.set_style("darkgrid")

	fig, ax = plt.subplots(figsize=(8, 5))
	ax.set(xlabel=r'Upper triangular matrix index $k$ = len(sequence)-diag_idx', 
		ylabel=r'$||$cmap$(x)-$cmap$(\tilde{x})||_2$', 
		title='l2 dist. bw original and perturbed contact maps')

	for key in keys:
		sns.lineplot(x=df['k'], y=df[f'{key}_cmap_dist'], label=key)

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_cmap_distances.png"))
		plt.close()

	return fig

def plot_confidence(df, keys, filepath=None, filename=None):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	### masked pred accuracy vs pseudo likelihood

	df["masked_pred_accuracy"] = df["masked_pred_accuracy"].apply(lambda x: format(float(x),".2f"))
	df = df.sort_values(by=['masked_pred_accuracy'])

	fig, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

	# sns.stripplot(data=df, x="masked_pred_accuracy", y="masked_pred_pseudo_likelihood", jitter=0.3, ax=ax[0])
	sns.violinplot(data=df, x="masked_pred_accuracy", y="masked_pred_pseudo_likelihood", ax=ax[0], palette="Blues")
	plt.setp(ax[0].collections, alpha=.8)

	sns.histplot(data=df, x="masked_pred_accuracy", ax=ax[1])

	plt.tight_layout()
	# plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_accuracy_vs_likelihood.png"))
		plt.close()

	### pseudo-likelihood

	fig, ax = plt.subplots(figsize=(8, 5))
	for idx, key in enumerate(keys):
		sns.distplot(x=df[f"{key}_pseudo_likelihood"], label=key, kde=True, hist=False)

	plt.xlabel(r'Pseudo likelihood of substitution: $\mathbb{E}_{i\in I}[p(\tilde{x}_i|x_{<i>})]]$')
	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_pseudo_likelihood.png"))
		plt.close()

	### evo-velocity

	fig, ax = plt.subplots(figsize=(8, 5))
	for idx, key in enumerate(keys):
		sns.distplot(x=df[f"{key}_evo_velocity"], label=key, kde=True, hist=False)
	plt.xlabel(r'Evo velocity $\mathbb{E}_{i\in I}[ \log p(x_i|x_{<i>})-\log p(\tilde{x}_i|x_{<i>})]$')
	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_evo_velocity.png"))
		plt.close()

def plot_embeddings_distances(df, keys, embeddings_distances, filepath, filename):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")
	fig, ax = plt.subplots(figsize=(8, 5))

	# print(df[f'max_cos_embedding_distance'].describe())
	# print(df[f'masked_pred_embedding_distance'].describe())
	
	### adversarial perturbations
	for idx, key in enumerate(keys):
		sns.distplot(x=df[f'{key}_embedding_distance'], label=f'{key} embeddings', kde=True, hist=False)

	### all possible token choices and residues substitutions
	sns.distplot(x=embeddings_distances.flatten(), label='all possible embeddings', kde=True, hist=True)

	plt.xlabel(r'Distribution of embeddings distances $||z-z_I(C_I)||_2$ (varying $z$ and $C_I$)')
	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_embeddings_distances.png"))
		plt.close()

def plot_blosum_distances(df, keys, filepath=None, filename=None, plot_method='histplot'):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	fig, ax = plt.subplots(figsize=(8, 5))
	plt.xlabel(r'Blosum distance $(x,\tilde{x})$')
	
	if plot_method=='histplot':
		df = df[['original_sequence']+[f"{key}_blosum_dist" for key in keys]]
		df = df.melt(id_vars=['original_sequence'], var_name="key", value_name="blosum")
		ax = sns.histplot(x=df["blosum"], hue=df["key"], kde=False, multiple="stack")

	elif plot_method=='distplot':
		for idx, key in enumerate(keys):
			sns.distplot(x=df[f"{key}_blosum_dist"], label=key, kde=True, hist=False)

	else:
		raise ValueError

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_blosum_distances.png"))
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
		fig.savefig(os.path.join(filepath, filename+"_tokens_attention.png"))
		plt.close()

	return fig


def plot_contact_maps(original_contacts, adversarial_contacts, filepath=None, filename=None):

	matplotlib.rc('font', **{'size': FONT_SIZE})

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
		fig.savefig(os.path.join(filepath, filename+"_contact_maps.png"))
		plt.close()

	return fig