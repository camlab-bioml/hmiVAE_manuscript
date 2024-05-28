#### creating clusters for FlowSOM and Louvain

from flowsom import flowsom as flowsom
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from sklearn.cluster import AgglomerativeClustering
from anndata import AnnData

import sys
sys.path.insert(1,'../../scripts/hmivae/')
from ScModeDataloader import ScModeDataloader

cohorts = ['Jackson-BC', 'Ali-BC', 'Hoch-Melanoma']

cofactors = {'Jackson-BC':5.0, 'Ali-BC':0.8, 'Hoch-Melanoma': 1.0}

### Running FlowSOM - full features set

# creating the files for flowsom
n_clusters = {}
for cohort in cohorts:
    adata = sc.read_h5ad(f"../pdac_analysis/best_run_{cohort}_out/{cohort}_adata_new.h5ad")
    h5ad = adata.copy()
    sc.pp.neighbors(h5ad, use_rep='spatial', n_neighbors=10)
    print('done neighbours')

    data = ScModeDataloader(h5ad)

    spatial_context = data.C.numpy()
    
    print(spatial_context.shape)

    spatial_context_names = ['neighbour_'+ i for i in list(h5ad.var_names)+h5ad.uns['names_correlations'].tolist()+h5ad.uns['names_morphology'].tolist()]
    
    print(len(h5ad.var_names), len(h5ad.uns['names_correlations']), len(h5ad.uns['names_morphology']), len(spatial_context_names))
    n_clusters[cohort] = len(h5ad.obs.leiden.unique())
    full_feature_array = np.concatenate([h5ad.X, 
                                         h5ad.obsm['correlations'], 
                                         h5ad.obsm['morphology'], 
                                         spatial_context], axis=1)
    full_feature_names = list(h5ad.var_names)+list(h5ad.uns['names_correlations'])+list(h5ad.uns['names_morphology'])+spatial_context_names
    print(full_feature_names)
    df = pd.DataFrame(full_feature_array, columns=full_feature_names)
    df['Sample_name'] = h5ad.obs.Sample_name.tolist()
    df['cell_id'] = h5ad.obs.reset_index()['cell_id'].tolist()
    
    print(cohort, df.shape)
    
    df.to_csv(f"{cohort}_csv_for_flowsom.csv") # save file

# FlowSOM centers all channels before it runs by subtracting the mean of each channel from every observation. 
# - https://support.cytobank.org/hc/en-us/articles/360015918512-How-to-Configure-and-Run-a-FlowSOM-Analysis

for cohort in cohorts:
    print(f"Running {cohort}")
    start = time.time()
    fsom = flowsom(f"{cohort}_csv_for_flowsom.csv", # read in saved file
                   if_fcs=False, 
                   if_drop=True, 
                   drop_col=['Unnamed: 0','Sample_name', 'cell_id'])
    fsom.som_mapping(50, 50, fsom.df.shape[1], sigma=1, lr=0.5, batch_size=1000)  # trains SOM with 1000 iterations
    fsom.meta_clustering(AgglomerativeClustering, 
                         min_n=n_clusters[cohort], 
                         max_n=n_clusters[cohort], 
                         verbose=False, 
                         iter_n=10)
    print("Starting cluster labeling")
    fsom.labeling(verbose=False)
    
    df = fsom.df
    orig_df = pd.read_csv(f"{cohort}_csv_for_flowsom.csv", index_col=0)
    df['Sample_name'] = orig_df.Sample_name.tolist()
    df['cell_id'] = orig_df['cell_id'].tolist()
    df = df[['Sample_name', 'cell_id', 'category']]
    df.to_csv(f"{cohort}_flowsom_clusters.tsv", sep='\t') # save cluster assignments
    stop = time.time()
    fsom.vis(t=1, # the number of total nodes = t * bestk
       edge_color='b', 
       node_size=300, 
       with_labels=True)
    
    
    print(f"{cohort} done in {(stop-start)/60} minutes!")

### Running Louvain - full feature set

resolutions = {'Jackson-BC': 0.4, 'Ali-BC': 0.9, 'Hoch-Melanoma': 0.15}
for cohort in cohorts:
    print(f"Running {cohort}")
    df = pd.read_csv(f"{cohort}_csv_for_flowsom.csv", index_col=0)
    
    #df.head()
    c_adata = AnnData(X=df.iloc[:, 0:-2],
                     dtype=np.float32)
    
    c_adata.obs = df[['Sample_name', 'cell_id']]
    
    sc.pp.neighbors(c_adata, use_rep=None, key_added='standard', n_neighbors=10) # have to rerun neighbours because the previous is based on spatial coordinates, runs PCA cuz n_vars > 50
    
    c_adata.X = StandardScaler().fit_transform(c_adata.X)
    print("Running leiden")
    sc.tl.leiden(c_adata, neighbors_key='standard', key_added='standard_leiden', resolution=resolutions[cohort])
    
    #communities, graph, Q = phenograph.cluster(pd.DataFrame(c_adata.obsm['X_pca']),k=k) # run PhenoGraph
    # store the results in adata:
    #c_adata.obs['PhenoGraph_clusters'] = pd.Categorical(communities)
    
    df_cluster = c_adata.obs.reset_index()[['cell_id', 'Sample_name', 'standard_leiden']]
    
    print(f"{cohort} leiden clusters:{len(df_cluster.standard_leiden.unique())}")
    print(f"{cohort} hmivae clusters: {n_clusters[cohort]}")
    
    #print(f"num of vae clusters: {len(c_adata.obs.leiden.unique())}; num of std workflow clusters: {len(df.standard_louvain.unique())}")
    
    df_cluster.to_csv(
        f"{cohort}_leidenres{resolutions[cohort]}_clustering.tsv", sep='\t') # save cluster assignments with louvain resolution used

#### Running FlowSOM and Louvain - Expression-only features set

for cohort in cohorts:
    h5 = sc.read_h5ad(f"{cohort}/best_run_{cohort}_out/{cohort}_adata_new.h5ad")
    n_proteins = h5.X.shape[1]
    new_h5 = h5.copy()
    drop_cols = (
        list(new_h5.uns['names_correlations']) +
                 
        list(new_h5.uns['names_morphology']) + 
                 
        ['neighbour_'+i for i in list(new_h5.var_names)+list(new_h5.uns['names_correlations'])+list(new_h5.uns['names_morphology'])]
                )
    
    sc.pp.neighbors(new_h5, use_rep=None, key_added='standard', n_neighbors=10) # have to rerun neighbours because the previous is based on spatial coordinates
    
    new_h5.X = StandardScaler().fit_transform(new_h5.X)
    print("Running louvain")
    sc.tl.louvain(new_h5, neighbors_key='standard', key_added='standard_louvain', resolution=0.16) # try different resolutions
    
    print("Prep for FlowSOM")
    n_clusters = len(h5.obs.expression_leiden.unique())
    
    fsom = flowsom(f"{cohort}_csv_for_flowsom.csv", # file we created earlier
                   if_fcs=False, 
                   if_drop=True, 
                   drop_col=['Unnamed: 0','Sample_name', 'cell_id'] + drop_cols) # only the expression columns
    fsom.som_mapping(50, 50, fsom.df.shape[1], sigma=1, lr=0.5, batch_size=1000)  # trains SOM with 1000 iterations
    fsom.meta_clustering(AgglomerativeClustering, 
                         min_n=n_clusters, 
                         max_n=n_clusters, 
                         verbose=False, 
                         iter_n=10)
    print("Starting cluster labeling")
    fsom.labeling(verbose=False)
    
    df = new_h5.obs.reset_index()[['Sample_name', 'cell_id', 'standard_louvain', 'expression_leiden']] 
    df['category'] = fsom.df['category'].tolist()
    df['cell_id'] = df['cell_id'].map(str)
    df.to_csv(f"{cohort}_exp_only_louvres0.14_clusters.tsv", # use resolution used
              sep='\t') # save cluster assignments
    
    print(f"louvain: {len(df.standard_louvain.unique())}")
    print(f"flowsom: {len(df.category.unique())}")
    print(f"hmivae: {len(df.expression_leiden.unique())}") # making sure they all have the same number of clusters




