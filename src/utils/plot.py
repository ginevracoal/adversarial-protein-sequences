import os
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DPI=150
TOP=0.92
FONT_SIZE=13
palette="rocket_r"
heatmaps_palette="rocket_r"
sns.set_style("darkgrid")
sns.set_palette(palette, 5)
linestyles=['-', '--', '-.', ':', '-', '--','-.']
matplotlib.rc('font', **{'size': FONT_SIZE})

def set_boxplot_linecolor(ax):
	for i, patch in enumerate(ax.artists):
		r, g, b, a = patch.get_facecolor()
		col = (r, g, b, a) 
		patch.set_facecolor((r, g, b, .7))
		patch.set_edgecolor(col)

		for j in range(i*6, i*6+6):
			line = ax.lines[j]
			line.set_color(col)
			line.set_mfc(col)
			line.set_mec(col)

def plot_attention_scores(df, missense_df=None, filepath=None, filename=None):
	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=DPI, sharey=True)
	df = df.sort_values('original_token') 
	sns.stripplot(data=df, x='original_token', y='target_token_attention', hue='original_token', label='',
		palette=palette)
	ax.set_xlabel('Target residue')
	ax.set_ylabel('Attention')
	ax.get_legend().remove()
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_residue_vs_attention.png"))
		plt.close()

	fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=DPI, sharey=True)
	df['target_token_idx'] = df['target_token_idx'].astype(int)
	sns.scatterplot(data=df, x='target_token_idx', y='target_token_attention', hue='target_token_idx', label='')
	ax.set_xlabel('Target token idx')
	ax.set_ylabel('Attention')
	ax.get_legend().remove()
	plt.tight_layout()
	plt.show()
	
	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_token_idx_vs_attention.png"))
		plt.close()

	return fig

def plot_tokens_hist(df, keys, split=True, filepath=None, filename=None):
	df['perc_token_idx'] = df.apply(lambda row: row['target_token_idx']/len(row['original_sequence']), axis=1)

	matplotlib.rc('font', **{'size': FONT_SIZE})
	sns.set_style("darkgrid")

	if split:

		### split by perturbations_keys
		assert len(keys)>1
		fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI, nrows=2, ncols=2, sharey=True)
		jitter = 0.1

		for key, axis in ((keys[0],ax[0,0]), (keys[1],ax[1,0]), (keys[2],ax[0,1]), (keys[3],ax[1,1])):
			df = df.sort_values(f'{key}_token') 
			# sns.swarmplot(data=df, y="perc_token_idx", x=f"{key}_token", ax=axis)
			sns.stripplot(data=df, y="perc_token_idx", x=f"{key}_token", dodge=True, ax=axis, jitter=jitter,
				palette=palette)
			axis.set_ylabel('token idx percentile') # percentile in case of different lenghts

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

		fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)

		subst_counts = df.groupby(['original_token', f'{key}_token'], as_index=False).size()
		subst_counts = subst_counts.rename(columns={"size": "n_substitutions"})

		df_heatmap = subst_counts.pivot_table(values='n_substitutions', columns='original_token', index=f'{key}_token')
		sns.heatmap(df_heatmap, annot=True, cmap="rocket_r")

		plt.tight_layout()
		plt.show()

		if filepath is not None and filename is not None:
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			fig.savefig(os.path.join(filepath, filename+f"_substitutions_{key}_token.png"))
			plt.close()

def plot_cmap_distances(df, keys, distances_df=None, missense_df=None, filepath=None, filename=None, L=1, R=20):

	df = df[(df['k']>=L) & (df['k']<=R)]
	distances_df = distances_df[(distances_df['k']>=L) & (distances_df['k']<=R)]

	fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
	ax.set(xlabel=r'Upper triangular matrix index $k$', #' = len(sequence)-diag_idx', 
		ylabel=r'dist(cmap($x$),cmap($\tilde{x}$))')

	for idx, key in enumerate(keys):
		sns.lineplot(x=df['k'], y=df[f'{key}_cmap_dist'], label=key, ls=linestyles[idx])

	if missense_df is not None:
		sns.lineplot(x=df['k'], y=missense_df['missense_cmap_dist'], label='missense', ls='--', color='black')

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_cmap_distances.png"))
		plt.close()

	return fig

def plot_confidence(df, keys, missense_df=None, filepath=None, filename=None, plot_method='boxplot'):

	### masked pred accuracy vs pseudo likelihood

	df["masked_pred_accuracy"] = df["masked_pred_accuracy"].apply(lambda x: format(float(x),".2f"))
	df = df.sort_values(by=['masked_pred_accuracy'])

	fig, ax = plt.subplots(2, 1, figsize=(6, 5), dpi=DPI, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

	sns.violinplot(data=df, x="masked_pred_accuracy", y="masked_pred_pseudo_likelihood", ax=ax[0], palette="Blues")
	plt.setp(ax[0].collections, alpha=.8)

	sns.histplot(data=df, x="masked_pred_accuracy", ax=ax[1])
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_accuracy_vs_likelihood.png"))
		plt.close()

	### perplexity

	if plot_method=='boxplot':

		columns = ["perturbation", "perplexity"]
		new_df = pd.DataFrame(columns=columns)
		for key in keys:
			entry = pd.DataFrame({"perplexity":df[f"{key}_perplexity"]}).assign(perturbation=key) 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		sns.boxplot(data=new_df, y='perplexity', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
		for idx, key in enumerate(keys):
			if key!='masked_pred':
				g = sns.distplot(x=df[f"{key}_perplexity"], label=key, kde=True, hist=False, 
					kde_kws={'linestyle':linestyles[idx]})

		plt.xlabel(r'Perplexity of predictions: $e^{H(p)}$')
		plt.legend()
	
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_perplexity.png"))
		plt.close()

	### bleu

	fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
	for idx, key in enumerate(keys):
		if key!='masked_pred':
			g = sns.distplot(x=df[f"{key}_bleu"], label=key, kde=True, hist=False, 
				kde_kws={'linestyle':linestyles[idx]})

	plt.xlabel(r'BLEU score')
	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_bleu.png"))
		plt.close()

	### pseudo-likelihood

	if plot_method=='boxplot':

		columns = ["perturbation", "pseudo_likelihood"]
		new_df = pd.DataFrame(columns=columns)
		for key in keys:
			entry = pd.DataFrame({"pseudo_likelihood":df[f"{key}_pseudo_likelihood"]}).assign(perturbation=key) 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)
		sns.boxplot(data=new_df, y='pseudo_likelihood', x="perturbation", palette=palette)
		ax.set_xticklabels(ax.get_xticklabels(),rotation=20)
		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
		for idx, key in enumerate(keys):
			hist=True if key=='masked_pred' else False
			sns.distplot(x=df[f"{key}_pseudo_likelihood"], label=key, kde=True, hist=hist, 
				kde_kws={'linestyle':linestyles[idx]})

		plt.xlabel(r'Pseudo likelihood of substitutions: $\mathbb{E}_{i\in I}[p(\tilde{x}_i|x^M_i)]$')
		plt.legend()
	
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_pseudo_likelihood.png"))
		plt.close()

	### perp vs pseudo likelihood

	temp_df = pd.DataFrame()
	for index, row in df.iterrows():
		new_row = {}
		for key in keys:
			if key!='masked_pred':			
				new_row['key'] = key
				new_row['perplexity'] = row[f"{key}_perplexity"]
				new_row['pseudo_likelihood'] = row[f"{key}_pseudo_likelihood"]
				temp_df = temp_df.append(new_row, ignore_index=True)

	fig = plt.figure(figsize=(6, 5), dpi=DPI)
	g = sns.jointplot(data=temp_df, x="pseudo_likelihood", y="perplexity", hue="key", kind='kde')
	plt.xlabel(r'Pseudo likelihood of substitutions: $\mathbb{E}_{i\in I}[p(\tilde{x}_i|x^M_i)]$')
	plt.ylabel(r'Perplexity of predictions: $e^{H(p)}$')

	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		plt.savefig(os.path.join(filepath, filename+"_perplexity_vs_likelihood.png"))
		plt.close()

	### evo-velocity

	fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
	for idx, key in enumerate(keys):
		hist=True if key=='masked_pred' else False
		sns.distplot(x=df[f"{key}_evo_velocity"], label=key, kde=True, hist=hist,
			kde_kws={'linestyle':linestyles[idx]})

	plt.xlabel(r'Evo velocity $\mathbb{E}_{i\in I}[ \log p(\tilde{x}_i|x^M_i)-\log p(x_i|x^M_i)]$')

	if missense_df is not None:
		ymax = ax.get_ylim()[1]

		for idx, value in enumerate(missense_df['missense_evo_velocity'].unique()):
			label='missense' if idx==0 else ''
			plt.plot([value,value], [0, ymax], ls='--', lw=1, color='black', label=label)

	plt.tight_layout()
	plt.legend()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_evo_velocity.png"))
		plt.close()

def plot_embeddings_distances(df, keys, filepath, filename, distances_df=None, plot_method='distplot'):
	sns.set_style("darkgrid")

	if 'masked_pred' in keys:
		keys.remove('masked_pred')

	if plot_method=='boxplot':
		matplotlib.rc('font', **{'size': 10})

		fig, ax = plt.subplots(figsize=(6, 4), dpi=DPI)

		columns = ["perturbation", "embedding_distance"]
		new_df = pd.DataFrame(columns=columns)
		for key in keys:
			entry = pd.DataFrame({"embedding_distance":df[f"{key}_embedding_distance"]}).assign(perturbation=key) 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		if distances_df is not None:
			entry = pd.DataFrame({"embedding_distance":distances_df['embedding_distance']}).assign(perturbation='other') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		print(new_df)
		sns.boxplot(data=new_df, y='embedding_distance', x="perturbation", palette=palette)
		plt.ylabel(r'Embeddings distance')#: dist($z,\tilde{z}$)')

		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)

		if distances_df is not None:
			sns.distplot(x=distances_df['embedding_distance'], label='other', kde=True, hist=False,
				kde_kws={'linestyle':linestyles[0]})

		for idx, key in enumerate(keys):
			hist=True if key=='masked_pred' else False		
			sns.distplot(x=df[f'{key}_embedding_distance'], label=f'{key}', kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx+1]})

		plt.legend()
		plt.xlabel(r'Embeddings distance')

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_embeddings_distances.png"))
		plt.close()

def plot_blosum_distances(df, keys, distances_df=None, missense_df=None, filepath=None, filename=None, plot_method='distplot'):

	if 'masked_pred' in keys:
		keys.remove('masked_pred')

	if plot_method=='boxplot':
		matplotlib.rc('font', **{'size': 10})

		fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)
		columns = ["perturbation", "blosum_distance"]
		new_df = pd.DataFrame(columns=columns)
		for key in keys:
			entry = pd.DataFrame({"blosum_distance":df[f"{key}_blosum_dist"]}).assign(perturbation=key) 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		if distances_df is not None:
			entry = pd.DataFrame({"blosum_distance":distances_df['blosum_distance']}).assign(perturbation='other') 
			new_df = pd.concat([new_df, entry], ignore_index=True)

		print(new_df)
		sns.boxplot(data=new_df, y='blosum_distance', x="perturbation", palette=palette)
		plt.ylabel(r'Blosum distance $(x,\tilde{x})$')

		set_boxplot_linecolor(ax)

	elif plot_method=='distplot':

		fig, ax = plt.subplots(figsize=(6, 3.5), dpi=DPI)

		if distances_df is not None:
			sns.distplot(x=distances_df['blosum_distance'], label='other', kde=True, hist=False,
				kde_kws={'linestyle':linestyles[0]})

		for idx, key in enumerate(keys):
			hist=True if key=='masked_pred' else False
			sns.distplot(x=df[f"{key}_blosum_dist"], label=key, kde=True, hist=hist,
				kde_kws={'linestyle':linestyles[idx+1]})

		plt.legend()
		plt.xlabel(r'Blosum distance')

	else:
		raise ValueError

	if missense_df is not None:

		ymax = ax.get_ylim()[1]

		for idx, blosum_dist in enumerate(missense_df['missense_blosum_distance'].unique()):
			label='missense' if idx==0 else ''
			plt.plot([blosum_dist,blosum_dist], [0, ymax], ls='--', lw=1, color='black', label=label)

	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_blosum_distances.png"))
		plt.close()

	return fig

def plot_attention_grid(sequence, heads_attention, layer_idx, target_token_idxs, filepath=None, filename=None):

	assert len(sequence)==heads_attention.shape[1]

	fig = plt.figure(figsize=(13, 9), dpi=DPI)

	for idx, scores in enumerate(heads_attention):

		if len(heads_attention)==20:
			ax = fig.add_subplot(4, 5, idx+1)
		elif len(heads_attention)==12:
			ax = fig.add_subplot(3, 4, idx+1)

		im = ax.imshow(np.array(scores), cmap=heatmaps_palette)

		ax.set_xticks([])
		ax.set_yticks([])

		fontdict = {'fontsize': 10}
		ax.set_xlabel('{} {}'.format('Head', idx+1))

		fig.colorbar(im, fraction=0.046, pad=0.04)

	plt.grid(False)
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_grid.png"))
		plt.close()

	fig, ax = plt.subplots(figsize=(10, 9), dpi=DPI)
	fontdict = {'fontsize': 9}
	avg_attentions = np.array(heads_attention.mean(0).squeeze())
	ax = sns.heatmap(avg_attentions, linewidth=0.01, cmap=heatmaps_palette)

	ax.set_xticks(range(len(sequence)))
	ax.set_yticks(range(len(sequence)))

	ax.set_xticklabels(sequence, fontdict=fontdict, rotation=0)
	ax.set_yticklabels(sequence, fontdict=fontdict, rotation=90)

	for col_idx in target_token_idxs:
		x, y, w, h = col_idx, 0, 1, len(sequence)
		ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='black', lw=2, clip_on=False))
		ax.tick_params(length=0)

	plt.grid(False)
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_avg_attention.png"))
		plt.close()

	return fig


def plot_cmaps(original_contacts, adversarial_contacts, key, filepath=None, filename=None):

	matplotlib.rc('font', **{'size': FONT_SIZE})

	fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI, ncols=3)

	im = ax[0].imshow(original_contacts, cmap=heatmaps_palette) 
	plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

	im = ax[1].imshow(adversarial_contacts, cmap=heatmaps_palette)
	plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

	im = ax[2].imshow(original_contacts-adversarial_contacts, cmap=heatmaps_palette)
	plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
	
	for idx in [0,1,2]:
		ax[idx].set_xticks([])
		ax[idx].set_yticks([])

	ax[0].set_xlabel('original')
	ax[1].set_xlabel(f'{key} perturbation')
	ax[2].set_xlabel('orig-pert')

	plt.grid(False)
	plt.tight_layout()
	plt.show()

	if filepath is not None and filename is not None:
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		fig.savefig(os.path.join(filepath, filename+"_contact_maps.png"))
		plt.close()

	return fig