### latent space projection test

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import itertools
from scipy.stats.mstats import winsorize
from anndata import AnnData
import sys
sys.path.insert(1,'../../scripts/hmivae/')
from ScModeDataloader import ScModeDataloader
from rich.progress import (
    track,
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

### hmiVAE

### switch the files to get the analysis done on the other way around

proj_adata = sc.read_h5ad("Ali_on_Jackson_w_bkg_adata_new.h5ad") # adata after running trained hmiVAE on different dataset

base_adata = sc.read_h5ad("../../analysis/cluster_analysis/Jackson-BC/best_run_Jackson-BC_out/Jackson-BC_adata_new.h5ad") # dataset used to train hmiVAE

true_adata = sc.read_h5ad("../../analysis/cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_adata_new.h5ad") # original adata

### train the KNN model on each view separately - hmiVAE

views = { # [embedding name, clustering name]
    'integrated':['VAE', 'leiden'], 
    'expression': ['expression_embedding','expression_leiden'], 
    'correlation': ['correlation_embedding','correlation_leiden'], 
    'morphology': ['morphology_embedding','morphology_leiden'], 
    'spatial_context': ['spatial_context_embedding','spatial_context_leiden']
}

hmivae_predicted_labels = {}

progress = Progress(
        TextColumn("[progress.description]Predicting labels for projected embedding from view."),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
with progress:
    for v in progress.track(views.keys()):
        
        print(f"Starting prediction for {v}")

        X =  base_adata.obsm[views[v][0]] #key in obsm with embedding

        y = base_adata.obs[views[v][1]].tolist() #key in obs with labels

        classifier = KNeighborsClassifier(n_neighbors=100, metric='cosine') #100 neighbours is what we used in train/test on same dataset, change metric for cosine or euclid

        classifier.fit(X, y)

        hmivae_predicted_labels[v] = classifier.predict(proj_adata.obsm[views[v][0]]) #key in obsm with embedding

        print(f"Done prediction for {v} view")


for i in hmivae_predicted_labels.keys():
    proj_adata.obs[f'predicted_{i}'] = hmivae_predicted_labels[i]


### Baseline

full_features_jackson = pd.read_csv("../../analysis/cluster_analysis/Jackson-BC_csv_for_flowsom.csv", index_col=0)

full_features_ali = pd.read_csv("../../analysis/cluster_analysis/Ali-BC_csv_for_flowsom.csv", index_col=0) # files with all the features

full_jack_adata = sc.read_h5ad("../../analysis/cluster_analysis/Jackson-BC/vae_prep/all_samples_merged_vae.h5ad")

exclude = ['DNA1', 'DNA2']

full_jack_adata = full_jack_adata[:, ~full_jack_adata.var_names.isin(exclude)].copy()

full_jack_adata.obsm['correlations'] = np.delete(full_jack_adata.obsm['correlations'], [13,14], 1)

full_jack_adata.uns['names_correlations'] = np.delete(full_jack_adata.uns['names_correlations'], [13,14])

full_jack_adata.X = np.arcsinh(full_jack_adata.X/5)

for i in range(full_jack_adata.X.shape[1]):
    full_jack_adata.X[:,i] = winsorize(full_jack_adata.X[:,i], limits=[0, 0.01])
    
for i in range(full_jack_adata.obsm['morphology'].shape[1]):
    full_jack_adata.obsm['morphology'][:,i] = winsorize(full_jack_adata.obsm['morphology'][:,i], limits=[0, 0.01])

data = ScModeDataloader(full_jack_adata)

spatial_context = data.C.numpy()

spatial_context_names = ['neighbour_'+ i for i in list(full_jack_adata.var_names)+full_jack_adata.uns['names_correlations'].tolist()+full_jack_adata.uns['names_morphology'].tolist()]

full_jack_adata.obsm['spatial_context'] = spatial_context

full_jack_adata.uns['names_spatial_context'] = spatial_context_names

full_jack_adata_df = pd.DataFrame(np.concatenate([full_jack_adata.X, 
                                                 full_jack_adata.obsm['correlations'],
                                                 full_jack_adata.obsm['morphology'],
                                                 full_jack_adata.obsm['spatial_context']], axis=1),
                                 columns=list(full_jack_adata.var_names)+full_jack_adata.uns['names_correlations'].tolist()+full_jack_adata.uns['names_morphology'].tolist()+full_jack_adata.uns['names_spatial_context'])

full_jack_adata_predict = AnnData(full_jack_adata_df)

full_jack_adata_predict.obs = full_jack_adata.obs

jack_adata = sc.read_h5ad("../../analysis/cluster_analysis/Jackson-BC/best_run_Jackson-BC_out/Jackson-BC_adata_new.h5ad")

ali_adata = sc.read_h5ad("../../analysis/cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_adata_new.h5ad")

baseline_jack_adata = AnnData(full_features_jackson.iloc[:,:-2]) # removing Sample_name and cell_id so they don't show in X

baseline_ali_adata = AnnData(full_features_ali.iloc[:,:-2])

baseline_jack_adata.obs = full_features_jackson[['Sample_name', 'cell_id']] # add them back but into the obs

baseline_ali_adata.obs = full_features_ali[['Sample_name', 'cell_id']]

baseline_ali_adata.obs.cell_id = baseline_ali_adata.obs.cell_id.map(str)

ali_adata.obs = ali_adata.obs.reset_index()

ali_true_labels.cell_id = ali_true_labels.cell_id.map(str)

### create the true baseline labels

sc.pp.neighbors(baseline_jack_adata, n_neighbors=100) # consistent with hmiVAE

sc.pp.neighbors(baseline_ali_adata, n_neighbors=100)

sc.tl.leiden(baseline_jack_adata, key_added='true_baseline_full_labels')

sc.tl.leiden(baseline_ali_adata, key_added='true_baseline_full_labels')

baseline_jack_adata.obs.index = baseline_jack_adata.obs.index.map(str)

baseline_ali_adata.obs.index = baseline_ali_adata.obs.index.map(str)

for view in ['expression', 'correlation', 'morphology', 'spatial_context']:
    print(view)
        
    if view == 'expression':
        
        ali_names = list(ali_adata.var_names)
        jack_names = list(jack_adata.var_names)
        a_adata = baseline_ali_adata.copy()[:, ali_names]
        j_adata = baseline_jack_adata.copy()[:, jack_names]
        
        print('expression', j_adata.X.shape)
        
        sc.pp.neighbors(j_adata, n_neighbors=100) # consistent with hmiVAE
        sc.pp.neighbors(a_adata, n_neighbors=100)
        
        sc.tl.leiden(j_adata, key_added=f'true_baseline_{view}_labels')
        sc.tl.leiden(a_adata, key_added=f'true_baseline_{view}_labels')
        
        baseline_ali_adata.obs[f'true_baseline_{view}_labels'] = a_adata.obs[f'true_baseline_{view}_labels'].tolist()
        baseline_jack_adata.obs[f'true_baseline_{view}_labels'] = j_adata.obs[f'true_baseline_{view}_labels'].tolist()
    elif view == 'correlation':
        
        ali_names = list(ali_adata.uns['names_correlations'])
        jack_names = list(jack_adata.uns['names_correlations'])
        a_adata = baseline_ali_adata.copy()[:, ali_names]
        j_adata = baseline_jack_adata.copy()[:, jack_names]
        
        print('correlation', j_adata.X.shape)
        
        sc.pp.neighbors(j_adata, n_neighbors=100) # consistent with hmiVAE
        sc.pp.neighbors(a_adata, n_neighbors=100)
        
        sc.tl.leiden(j_adata, key_added=f'true_baseline_{view}_labels')
        sc.tl.leiden(a_adata, key_added=f'true_baseline_{view}_labels')
        
        baseline_ali_adata.obs[f'true_baseline_{view}_labels'] = a_adata.obs[f'true_baseline_{view}_labels'].tolist()
        baseline_jack_adata.obs[f'true_baseline_{view}_labels'] = j_adata.obs[f'true_baseline_{view}_labels'].tolist()
        
    elif view == 'morphology':
        
        ali_names = list(ali_adata.uns['names_morphology'])
        jack_names = list(jack_adata.uns['names_morphology'])
        a_adata = baseline_ali_adata.copy()[:, ali_names]
        j_adata = baseline_jack_adata.copy()[:, jack_names]
        
        print('morphology', j_adata.X.shape)
        
        sc.pp.neighbors(j_adata, n_neighbors=100) # consistent with hmiVAE
        sc.pp.neighbors(a_adata, n_neighbors=100)
        
        sc.tl.leiden(j_adata, key_added=f'true_baseline_{view}_labels')
        sc.tl.leiden(a_adata, key_added=f'true_baseline_{view}_labels')
        
        baseline_ali_adata.obs[f'true_baseline_{view}_labels'] = a_adata.obs[f'true_baseline_{view}_labels'].tolist()
        baseline_jack_adata.obs[f'true_baseline_{view}_labels'] = j_adata.obs[f'true_baseline_{view}_labels'].tolist()
    else:
        
        ali_names = [i for i in baseline_ali_adata.var_names if 'neighbour' in i]
        jack_names = [i for i in baseline_jack_adata.var_names if 'neighbour' in i]
        
        a_adata = baseline_ali_adata.copy()[:, ali_names]
        j_adata = baseline_jack_adata.copy()[:, jack_names]
        
        print('spatial_context', j_adata.X.shape)
        
        sc.pp.neighbors(j_adata, n_neighbors=100, use_rep='X') # consistent with hmiVAE
        sc.pp.neighbors(a_adata, n_neighbors=100, use_rep='X')
        
        sc.tl.leiden(j_adata, key_added=f'true_baseline_{view}_labels')
        sc.tl.leiden(a_adata, key_added=f'true_baseline_{view}_labels')
        
        baseline_ali_adata.obs[f'true_baseline_{view}_labels'] = a_adata.obs[f'true_baseline_{view}_labels'].tolist()
        baseline_jack_adata.obs[f'true_baseline_{view}_labels'] = j_adata.obs[f'true_baseline_{view}_labels'].tolist()


full_jack_adata_predict.obs.index = full_jack_adata_predict.obs.index.map(str)

### train KNN on baseline

X =  baseline_ali_adata.X #all features

y = baseline_ali_adata.obs['true_baseline_full_labels'].tolist() #key in obs with labels

classifier = KNeighborsClassifier(n_neighbors=100) #100 neighbours is what we used in train/test on same dataset, change metric between cosine and euclid

classifier.fit(X, y)

full_classes = classifier.predict(full_jack_adata_predict[:,list(baseline_ali_adata.var_names)].X) #features in other dataset

full_jack_adata_predict.obs['predicted_baseline_full_labels'] = full_classes

base_views = {
    'expression': [list(ali_adata.var_names), 
                   full_jack_adata_predict[:, list(ali_adata.var_names)].X],
    'correlation': [list(ali_adata.uns['names_correlations']), 
                    full_jack_adata_predict[:, list(ali_adata.uns['names_correlations'])].X],
    'morphology': [list(ali_adata.uns['names_morphology']), 
                   full_jack_adata_predict[:, list(ali_adata.uns['names_morphology'])].X],
    'spatial_context': [[i for i in baseline_ali_adata.var_names if 'neighbour' in i], 
                       full_jack_adata_predict[:, [i for i in baseline_ali_adata.var_names if 'neighbour' in i]].X]
}

baseline_predicted_labels = {}

progress = Progress(
        TextColumn("[progress.description]Predicting labels for baseline view."),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
with progress:
    for v in progress.track(base_views.keys()):
        
        print(f"Starting prediction for {v}")

        X =  baseline_ali_adata.copy()[:, base_views[v][0]].X #only the view features

        y = baseline_ali_adata.obs[f'true_baseline_{v}_labels'].tolist() #key in obs with labels

        classifier = KNeighborsClassifier(n_neighbors=100) #100 neighbours is what we used in train/test on same dataset

        classifier.fit(X, y)

        baseline_predicted_labels[v] = classifier.predict(base_views[v][1]) #features in other dataset

        print(f"Done prediction for {v} view")


for i in baseline_predicted_labels.keys():
    full_jack_adata_predict.obs[f'predicted_baseline_{i}_labels'] = baseline_predicted_labels[i]


##### make sure to save your true and predicted labels for hmiVAE and baseline #######




