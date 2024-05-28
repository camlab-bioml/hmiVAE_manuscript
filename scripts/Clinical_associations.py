#### Clinical associations

import pandas as pd
import numpy as np 
import scipy
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm
import tifffile

from collections import Counter

from rich.progress import (
    track,
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)


cohort = 'Hoch-Melanoma' # switch for whichever dataset you're working on

cofactors = {'Jackson-BC':5.0, 'Ali-BC':0.8, 'Hoch-Melanoma':1.0}

cohort_rename = {'Jackson-BC': 'basel', 'Ali-BC': 'metabric', 'Hoch-Melanoma': 'melanoma'}

resolutions = {'Jackson-BC': 0.4, 'Ali-BC': 0.9, 'Hoch-Melanoma': 0.15} #resolutions used for louvain

### full features

adata = sc.read_h5ad(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_adata_new.h5ad") # hmiVAE

reg_louvain = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_louvainres{resolutions[cohort]}_clustering.tsv", sep='\t', index_col=0) # louvain

flowsom = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_flowsom_clusters.tsv",
                     sep='\t', index_col=0) # flowsom

### or expression features

exp_only_clusters = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_exp_only_all_methods_new_louvain_clusters.tsv",
                               sep='\t', index_col=0) # expression-only clusters for all methods

# exp_only_clusters = pd.read_csv("../cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_exp_only_resleiden1_louv0.3_all_methods_clusters.tsv",
#                                sep='\t', index_col=0) # different file name for Ali-BC


#### Patient data

patient_data = pd.read_csv(f"{cohort_rename[cohort]}/{cohort_rename[cohort]}/{cohort_rename[cohort]}_survival_patient_samples.tsv", sep='\t', index_col=0)

# select variables of interest for each dataset

# clinical_variables = [
#     'ERStatus', 
#     'grade', 
#     'PRStatus', 
#     'HER2Status', 
#     'Subtype', 
#     'clinical_type', 
#     'HR',
#     'PTNM_T', # stage
#     'response',
# ]   #'diseasestatus'] # Jackson-BC

#clinical_variables = ['ER_status', 'Grade', 'Stage', 'Histological_type'] # Ali-BC

clinical_variables = [
    'IHC_T_score', 
    'Cancer_Stage', 
    'Primary_melanoma_type', 
    'Mutation', 
    'Status_at_3m', 
    'Response_at_3m', 
    'Response_at_6m', 
    'Response_at_12m'
] # Hoch-Melanoma

## Visualize counts

plt.rcParams['figure.figsize'] = [10,10]

for n,i in enumerate(clinical_variables):
    ax = plt.subplot(5,2,n+1)
    df = pd.DataFrame(patient_data[i].value_counts()).transpose()
    
    df.plot.bar(ax=ax)
    
    ax.set_xticklabels([i], rotation=0)
    plt.legend(bbox_to_anchor=[1.0,1.1])


#### Combine clusters with patient data and find associations

method = 'hmiVAE' # which method
patient_col = 'Sample_name' # column with patient IDs
cluster_col = 'leiden' # column with clustering

df_to_use = adata.obs.reset_index() # which dataframe is being used

clusters_patient = pd.merge(df_to_use[['Sample_name',cluster_col,'cell_id']], patient_data.reset_index(), 
                            on='Sample_name')

#### Description 1: Proportion of cells as a measure of cluster prevalence

hi_or_low = clusters_patient[[patient_col, cluster_col]]

## Proportion of cells belonging to each cluster for each image / patient

hi_or_low = hi_or_low.groupby([patient_col, cluster_col]).size().unstack(fill_value=0)


hi_or_low = hi_or_low.div(hi_or_low.sum(axis=1), axis=0).fillna(0)

hi_low_cluster_variables = pd.merge(hi_or_low.reset_index(), clusters_patient[clinical_variables+[patient_col]], on=patient_col).drop_duplicates().reset_index(drop=True)

prop_cluster_cols = [i for i in hi_low_cluster_variables.columns if i in clusters_patient[cluster_col].unique().tolist()]
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []
    filtered_df = hi_low_cluster_variables[~hi_low_cluster_variables[cvar].isna()].copy() # drop nan values for each var
    
    for sub_cvar in filtered_df[cvar].unique():
        print(cvar, sub_cvar)
        selected_df = filtered_df.copy()
        selected_df[cvar] = list(map(int,selected_df[cvar] == sub_cvar))
        sub_cvar_df = pd.DataFrame({})
        y = selected_df[cvar].to_numpy() # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped   
        X = selected_df[prop_cluster_cols].to_numpy() # select columns corresponding to latent dims and convert to numpy
        tvalues = {}
        for cluster in range(X.shape[1]):
            print(f"cluster {cluster}")
            X1 = X[:, cluster]
            X1 = sm.add_constant(X1)
            try:
                log_reg = sm.Logit(y, X1).fit() # fit the Logistic Regression model
                
                tvalues[cluster] = log_reg.tvalues[1] # there will be 2 t values, first one belongs to the constant
            
            except Exception as e:
                exception_variables.append((cvar,sub_cvar, cluster, e))
                print(f"{cvar}:{sub_cvar} had an exception occur for cluster {cluster}: {e}")
                
        sub_cvar_df['cluster'] = list(tvalues.keys())

        sub_cvar_df['tvalues'] = list(tvalues.values())
        
        sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]
        
        cvar_dfs.append(sub_cvar_df)
        
    full_cvar_dfs = pd.concat(cvar_dfs)
    
    dfs.append(full_cvar_dfs)

full_cluster_clin_df = pd.concat(dfs).reset_index(drop=True)

full_cluster_clin_df = pd.pivot_table(full_cluster_clin_df, index='clinical_variable', values='tvalues', columns='cluster')

### save file

### Description 2: Cluster prevalence per mm2 of tissue as a measure of cluster prevalence

cohort_dirs = {'Jackson-BC': ['OMEnMasks/Basel_Zuri_masks', '_a0_full_maks.tiff'],
              'Ali-BC': ['METABRIC_IMC/to_public_repository/cell_masks', '_cellmask.tiff'],
              'Hoch-Melanoma': ['full_data/protein/cpout/', '_ac_ilastik_s2_Probabilities_equalized_cellmask.tiff']} # location of images on local disk

clusters = df_to_use[cluster_col].unique().tolist()

sample_dfs = []

progress = Progress(
        TextColumn(f"[progress.description]Finding cluster prevalances in {cohort}."),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
with progress:
    for sample in progress.track(df_to_use.Sample_name.unique()):
        s_df = pd.DataFrame({})
        s_cluster_prevs = {}
        mask = tifffile.imread(f"../../../data/{cohort_dirs[cohort][0]}/{sample}{cohort_dirs[cohort][1]}")
        sample_df = df_to_use.copy().query("Sample_name==@sample")
        for cluster in clusters:
            num_cells_in_sample = Counter(sample_df[cluster_col].tolist())
            num_cells_in_clusters = num_cells_in_sample[cluster]
            
            #print(num_cells_in_clusters)
            #print(mask.shape[0] , mask.shape[1])

            cluster_prevalance_per_mm2 = (num_cells_in_clusters / (mask.shape[0] * mask.shape[1])) * 1e6 # scale, 1 pixel == 1 micron

            s_cluster_prevs[cluster] = cluster_prevalance_per_mm2

        s_df['cluster'] = list(s_cluster_prevs.keys())
        s_df['prevalance_per_mm2_scaled_by_1e6'] = list(s_cluster_prevs.values())
        s_df['Sample_name'] = [sample]*s_df.shape[0]

        sample_dfs.append(s_df)

full_cohort_df = pd.concat(sample_dfs)

full_cohort_df['cluster'] = full_cohort_df['cluster'].map(int)

full_cohort_df = pd.pivot_table(full_cohort_df, values='prevalance_per_mm2_scaled_by_1e6', index='Sample_name', columns='cluster')

clusters = full_cohort_df.columns.tolist() # to make sure correct order later

cluster_per_tissue_patient = pd.merge(full_cohort_df, patient_data[clinical_variables+['Sample_name']], on='Sample_name')

cluster_cols = clusters
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []
    #filtered_df = hi_low_cluster_variables[~hi_low_cluster_variables[cvar].isna()].copy() # drop nan values for each var
    
    for sub_cvar in cluster_per_tissue_patient[cvar].dropna().unique().tolist():
        print(cvar, sub_cvar)
        sub_cvar_df = pd.DataFrame({})
        selected_df = cluster_per_tissue_patient.copy()[~cluster_per_tissue_patient[cvar].isna()] # drop nan values for each var
        selected_df[cvar] = list(map(int,selected_df[cvar] == sub_cvar))
        
        y = selected_df[cvar].to_numpy() # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped   
        X = selected_df[cluster_cols].to_numpy()# select columns corresponding to latent dims and convert to numpy
        tvalues = {}
        for cluster in range(X.shape[1]):
            print(f"cluster {cluster}")
            X1 = X[:, cluster]
            X1 = sm.add_constant(X1)
            try:
                log_reg = sm.Logit(y, X1).fit() # fit the Logistic Regression model
                
                tvalues[cluster] = log_reg.tvalues[1] # there will be 2 t values, first one belongs to the constant
            
            except Exception as e:
                exception_variables.append((cvar,sub_cvar, cluster, e))
                print(f"{cvar}:{sub_cvar} had an exception occur for cluster {cluster}: {e}")
                
        sub_cvar_df['cluster'] = list(tvalues.keys())

        sub_cvar_df['tvalues'] = list(tvalues.values())
        
        sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]
        
        cvar_dfs.append(sub_cvar_df)
        
    full_cvar_dfs = pd.concat(cvar_dfs)
    
    dfs.append(full_cvar_dfs)

tissue_clin_assoc = pd.concat(dfs).reset_index(drop=True)

tissue_clin_assoc = pd.pivot_table(tissue_clin_assoc, values='tvalues', index='clinical_variable', columns='cluster')

### save file

#### Clinical variable association with latent variables

## get median value for each latent dim across cells for each sample ID

df = pd.DataFrame(columns=['Sample_name']+[f'median_latent_dim_{n}' for n in range(adata.obsm['VAE'].shape[1])])

for n, sample in enumerate(adata.obs.Sample_name.unique()):
    sample_adata = adata.copy()[adata.obs.Sample_name.isin([sample]),:]
    
    df.loc[str(n)] = [sample]+ np.median(sample_adata.obsm['VAE'], axis=0).tolist()

## save file because this takes a while

## find associations

patient_latent = pd.read_csv(f"{cohort_rename[cohort]}/{cohort}_latent_dim_plus_patient_data.tsv", sep='\t', index_col=0) # read in saved file

patient_latent = pd.merge(patient_latent, patient_data, on='Sample_name')

latent_dim_cols = [i for i in patient_latent.columns if 'median' in i]
exception_variables = []

dfs = []

for cvar in clinical_variables:
    cvar_dfs = []
    
    for sub_cvar in patient_latent[cvar].unique():
        print(cvar, sub_cvar)
        sub_cvar_df = pd.DataFrame({})
        selected_df = patient_latent.copy()[~patient_latent[cvar].isna()] # drop nan values for each var
        selected_df[cvar] = list(map(int,selected_df[cvar] == sub_cvar))
        
        X = selected_df[latent_dim_cols].to_numpy() # select columns corresponding to latent dims and convert to numpy
        X = sm.add_constant(X) # add constant
        y = selected_df[cvar].to_numpy() # select the clinical variable column and convert to numpy -- no fillna(0) since all the nans should have been dropped
        try:
            log_reg = sm.Logit(y, X).fit() # fit the Logistic Regression model

            sub_cvar_df['latent_dim'] = [c.split('_')[-1] for c in latent_dim_cols]

            sub_cvar_df['tvalues'] = log_reg.tvalues[1:] # remove the constant

            sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]

            cvar_dfs.append(sub_cvar_df)
        except Exception as e:
            exception_variables.append((cvar,sub_cvar))
            print(f"{cvar}:{sub_cvar} had an exception occur: {e}")
        
    full_cvar_dfs = pd.concat(cvar_dfs)
    
    dfs.append(full_cvar_dfs)

## dig into features that showed perfect separation

features_to_remove = []
cvar_dfs2 = []
for cvar, sub_cvar in exception_variables:
    #print(cvar, sub_cvar)
    selected_df = patient_latent[~patient_latent[cvar].isna()].copy() # drop nan values for each var
    selected_df[cvar] = list(map(int,selected_df[cvar] == sub_cvar))
    y = selected_df[cvar].to_numpy()
    X = selected_df[latent_dim_cols].to_numpy() # select columns corresponding to latent dims and convert to numpy
    
    perf_sep_features = []
    for i in range(X.shape[1]):
        X_1 = X.copy()[:,0:i+1] 
        X_1 = sm.add_constant(X_1) # add constant
        try:
            log_reg = sm.Logit(y, X_1).fit() # fit the Logistic Regression model
            print(f"Completed: tvalues for {cvar}:{sub_cvar}, features till {i} -> {log_reg.tvalues}")
            #print(log_reg.summary())
        except Exception as e:
            print(f"{cvar}:{sub_cvar} for feature {i} has exception: {e}")
            perf_sep_features.append(i)
        
    if len(perf_sep_features) == 0:
        sub_cvar_df = pd.DataFrame({})
        sub_cvar_df['latent_dim'] = [c.split('_')[-1] for c in latent_dim_cols]
        
        assert len(log_reg.tvalues) == X.shape[1]+1 #for constant -- check this is the last one

        sub_cvar_df['tvalues'] = log_reg.tvalues[1:] # remove the constant -- this should be the last one

        sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]
        
        cvar_dfs2.append(sub_cvar_df)
        
    else:
    
        features_to_remove.append((cvar, sub_cvar, perf_sep_features))

sub_cvars = []

for cvar, sub_cvar, del_inds in features_to_remove:
    selected_df = patient_latent[~patient_latent[cvar].isna()].copy() # drop nan values for each var
    selected_df[cvar] = list(map(int,selected_df[cvar] == sub_cvar))
    y = selected_df[cvar].to_numpy()
    X = selected_df[latent_dim_cols].to_numpy() # select columns corresponding to latent dims and convert to numpy
    del_inds = del_inds
    X = np.delete(X,del_inds, axis=1)
    print(X.shape)
    X = sm.add_constant(X) # add constant
    try:
        log_reg = sm.Logit(y, X).fit() # fit the Logistic Regression model
        print(f"Completed: tvalues for {cvar}:{sub_cvar}, features till {i} -> {log_reg.tvalues}")
        
        sub_cvar_df = pd.DataFrame({})
        sub_cvar_df['latent_dim'] = [c.split('_')[-1] for c in latent_dim_cols]
        
        tvalues = log_reg.tvalues[1:].tolist() #+ [np.nan]
        
        for i in del_inds:
            if i > len(tvalues):
                tvalues = np.insert(tvalues, i-1, np.nan)
            else:
                tvalues = np.insert(tvalues, i , np.nan)
        
#         tvalues = np.insert(tvalues, del_inds.remove(19), np.nan)
        #assert len(log_reg.tvalues) == X.shape[1]+1 #for constant -- check this is the last one

        sub_cvar_df['tvalues'] = tvalues

        sub_cvar_df['clinical_variable'] = [f"{cvar}:{sub_cvar}"]*sub_cvar_df.shape[0]
        
        sub_cvars.append(sub_cvar_df)
    except Exception as e:
        print(f"{cvar}:{sub_cvar} for feature {i} has exception: {e}")

sub_cvar_df1 = pd.concat(sub_cvars)

full_clin_df = pd.concat(dfs).reset_index(drop=True)

final_full_clin_df = pd.concat([full_clin_df, sub_cvar_df1]).reset_index(drop=True)

final_full_clin_df['latent_dim'] = final_full_clin_df['latent_dim'].map(int)

final_full_clin_df = pd.pivot_table(final_full_clin_df, index='clinical_variable', values='tvalues', columns='latent_dim')

### save file ####





