### ranking features for full feature set clusters for all methods

import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from anndata import AnnData
import yaml

# for creating the spatial context

import sys
sys.path.insert(1,'../../scripts/hmivae/')
from ScModeDataloader import ScModeDataloader

## Using scanpy so load data and create view-specific adatas

cohort = 'Hoch-Melanoma' # switch between cohorts

method = 'category' # switch for which method you're working with

adata = sc.read_h5ad(f"{cohort}_adata_new.h5ad")

clusters = pd.read_csv("Hoch-Melanoma/best_run_Hoch-Melanoma_out/Hoch-Melanoma_flowsom_clusters.tsv", 
                       sep='\t', index_col=0) # which method clusters - 'category' is flowsom 

clusters.cell_id = clusters.cell_id.map(str)

adata.obs = pd.merge(adata.obs.reset_index(), clusters, on=['Sample_name', 'cell_id']) # don't need to do this for hmiVAE clusters - already in adata

adata.obs[method] = adata.obs[method].map(str)

sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=10)
print('done neighbours')

data = ScModeDataloader(adata)

spatial_context = data.C.numpy()

print(spatial_context.shape)

spatial_context_names = ['neighbour_'+ i for i in list(adata.var_names)+adata.uns['names_correlations'].tolist()+adata.uns['names_morphology'].tolist()]

### Create the adatas

exp_adata = AnnData(X=adata.X, obs=adata.obs)

exp_adata.var_names = adata.var_names

corr_adata = AnnData(X=adata.obsm['correlations'], obs=adata.obs)

corr_adata.var_names = adata.uns['names_correlations']

morph_adata = AnnData(X=adata.obsm['morphology'], obs=adata.obs)

morph_adata.X = MinMaxScaler().fit_transform(morph_adata.X) # scaling morphology because area is much larger than others

morph_adata.var_names = adata.uns['names_morphology']

sc_adata = AnnData(X=spatial_context, obs=adata.obs)

sc_adata.var_names = spatial_context_names

### Rank the features

sc.tl.rank_genes_groups(exp_adata, groupby=method, key_added=f'exp_ranks_{method}')
sc.tl.rank_genes_groups(corr_adata, groupby=method, key_added=f'corr_ranks_{method}')
sc.tl.rank_genes_groups(morph_adata, groupby=method, key_added=f'morph_ranks_{method}')
sc.tl.rank_genes_groups(sc_adata, groupby=method, key_added=f'sc_ranks_{method}')

### Now we pick top 3 features from each view to determine drivers

full_df_list = []

full_df_list.append(
        sc.get.rank_genes_groups_df(
            exp_adata,
            group=None, 
            key=f'exp_ranks_{method}')
    )

full_df_list.append(
        sc.get.rank_genes_groups_df(
            corr_adata, 
            group=None,
            key=f'corr_ranks_{method}')
    )

full_df_list.append(
        sc.get.rank_genes_groups_df(
            morph_adata, 
            group=None, 
            key=f'morph_ranks_{method}')
    )

full_df_list.append(
        sc.get.rank_genes_groups_df(
            sc_adata, 
            group=None, 
            key=f'sc_ranks_{method}')
    )

complete_full_r_df = pd.concat(full_df_list)

df1_list = []
for g in adata.obs[method].unique():
    df = sc.get.rank_genes_groups_df(
            exp_adata, 
            group=[g], 
            pval_cutoff=0.05, 
            key=f'exp_ranks_{method}').query("scores > 0").iloc[0:3,:] # top 3
    
    df['group'] = [g]*df.shape[0]
    df1_list.append(df) 

df1 = pd.concat(df1_list).reset_index(drop=True)

df2_list = []
for g in adata.obs[method].unique():
    df = sc.get.rank_genes_groups_df(
            corr_adata, 
            group=[g], 
            pval_cutoff=0.05, 
            key=f'corr_ranks_{method}').query("scores > 0").iloc[0:3,:] # top 3
    
    df['group'] = [g]*df.shape[0]
    df2_list.append(df) 

df2 = pd.concat(df2_list).reset_index(drop=True)

df3_list = []
for g in adata.obs[method].unique():
    df = sc.get.rank_genes_groups_df(
            morph_adata, 
            group=[g], 
            pval_cutoff=0.05, 
            key=f'morph_ranks_{method}').query("scores > 0").iloc[0:3,:] # top 3
    
    df['group'] = [g]*df.shape[0]
    df3_list.append(df) 

df3 = pd.concat(df3_list).reset_index(drop=True)

df4_list = []
for g in adata.obs[method].unique():
    df = sc.get.rank_genes_groups_df(
            sc_adata, 
            group=[g], 
            pval_cutoff=0.05, 
            key=f'sc_ranks_{method}').query("scores > 0").iloc[0:3,:] # top 3
    
    df['group'] = [g]*df.shape[0]
    df4_list.append(df) 

df4 = pd.concat(df4_list).reset_index(drop=True)

full_r_df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)

names = full_r_df.names.unique()

plot_df = complete_full_r_df[complete_full_r_df['names'].isin(names)]

def select_string(x):
    all_strings = []
    for i in x.split('_'):
        if i in ['neighbour', 'mean', 'nuc', 'stain']:
            continue
        else:
            all_strings.append(i)
            
    return '_'.join(all_strings)

plot_df = plot_df[['names', 'group', 'scores']].pivot(columns='names', index='group', values='scores').reset_index()

views = ['Views']

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

for i in plot_df.columns[1:].tolist():
    if i.startswith('neighbour'):
        if 'mean_nuc_stain' in i:
            views.append('Spatial Context - NC')
        elif len(intersection(i.split('_'), ['area', 'perimeter', 'concavity', 'eccentricity', 'asymmetry'])) > 0:
            views.append('Spatial Context - M')
        else:
            views.append('Spatial Context - E')
    elif 'mean_nuc_stain' in i:
        views.append('Nuclear Co-localization')
    elif i in ['area', 'perimeter', 'concavity', 'eccentricity', 'asymmetry']:
        views.append('Morphology')
    else:
        views.append('Expression')
        
print(len(views))

plot_df.loc[plot_df.shape[0]] = views

plot_df.columns = ['Group'] + [select_string(i) for i in plot_df.columns[1:]]

### save files
full_r_df.to_csv(f"{cohort}_{method}_int_clusters_rank_features.tsv", sep='\t')
plot_df.to_csv(f"{cohort}_{method}_int_clusters_rank_plot.tsv", sep='\t')
