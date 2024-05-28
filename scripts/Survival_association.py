### Survival associations

from lifelines.plotting import add_at_risk_counts
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from statsmodels.stats.multitest import multipletests
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sys

from collections import Counter
from rich.progress import (
    track,
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
import tifffile

cofactors = {'Jackson-BC':5.0, 'Ali-BC':0.8, 'Hoch-Melanoma':1.0}

cohort_rename = {'Jackson-BC': 'basel', 'Ali-BC': 'metabric', 'Hoch-Melanoma': 'melanoma'}

resolutions = {'Jackson-BC': 0.4, 'Ali-BC': 0.9, 'Hoch-Melanoma': 0.15} #resolutions used for louvain

cohort = 'Ali-BC'

cofactor = cofactors[cohort]

res = resolutions[cohort]

patient_data = pd.read_csv(f"{cohort_rename[cohort]}/{cohort_rename[cohort]}/{cohort_rename[cohort]}_survival_patient_samples.tsv", sep='\t', index_col=0)

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
#     'observed', 
#     'OSmonth']  # Jackson-BC

clinical_variables = ['ER_status', 'Grade', 'Stage', 'Histological_type', 'DeathBreast', 'T'] # Ali-BC

# clinical_variables = ['IHC_T_score', 'Cancer_Stage', 'Primary_melanoma_type', 'Mutation', 
#                      'censoring_death', 'Time_to_death_or_last_PET'] # Hoch-Melanoma

strat_col = 'Stage' # stratifying on stage

durations_col = clinical_variables[-1]
censoring_col = clinical_variables[-2]

# full features clusters
hmi_clusters = sc.read_h5ad(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_adata_new.h5ad")

std_clusters = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_louvainres{str(res)}_clustering.tsv", sep='\t')

flowsom = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_flowsom_clusters.tsv", sep='\t')

# expression clusters

exp_clusters = pd.read_csv("../cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_exp_only_resleiden1_louv0.3_all_methods_clusters.tsv",
                      sep='\t', index_col=0) # different file name for Ali-BC

# exp_clusters = pd.read_csv(f"../cluster_analysis/{cohort}/best_run_{cohort}_out/{cohort}_exp_only_all_methods_new_louvain_clusters.tsv",
#                           sep='\t', index_col=0)

method = 'FlowSOM_exp_only'

patient_col = 'Sample_name'

cluster_col = 'category'

df_to_use = exp_clusters

n_clusters = len(df_to_use[cluster_col].unique())

clusters = df_to_use[cluster_col].unique()

clusters_patient = pd.merge(df_to_use[['Sample_name',cluster_col,'cell_id']], patient_data.reset_index(), 
                            on='Sample_name')

### Patient Survival wrt cluster proportion

hi_or_low = clusters_patient[[patient_col, cluster_col]]

## Proportion of cells belonging to each cluster for each image / patient

hi_or_low = hi_or_low.groupby([patient_col, cluster_col]).size().unstack(fill_value=0)


hi_or_low = hi_or_low.div(hi_or_low.sum(axis=1), axis=0).fillna(0)

hi_or_low_scaled = pd.DataFrame(MinMaxScaler().fit_transform(hi_or_low), index=hi_or_low.index, columns=hi_or_low.columns)

hi_low_cluster_variables = pd.merge(
    hi_or_low_scaled.reset_index(), 
    clusters_patient[[durations_col, censoring_col, strat_col]+[patient_col]], 
    on=patient_col).drop_duplicates().reset_index(drop=True)

hi_low_cluster_variables[durations_col] = hi_low_cluster_variables[durations_col].fillna(0).map(int)

hi_low_cluster_variables[censoring_col] = hi_low_cluster_variables[censoring_col].fillna(0).map(int)

f = plt.figure(figsize=(20,50))

cph_list = {}
concordance_ind = {}
    
for i in clusters:
    try:
        ax = f.add_subplot(10,3, i+1)

        df = hi_low_cluster_variables.loc[:, [i,durations_col, censoring_col, strat_col]].dropna()

        cph = CoxPHFitter(strata=[strat_col]) #

        cph.fit(df, durations_col, event_col=censoring_col)
        
        c_i = concordance_index(df[durations_col], -cph.predict_partial_hazard(df), df[censoring_col])

        #cph.print_summary()
        cph_list[i] = cph
        concordance_ind[i] = c_i

        cph.plot(ax=ax)
        ax.set_title(f'Cluster {i}')
    except:
        print(f'Cluster {i} gives error:', sys.exc_info()[0])

cond_df = pd.Series(concordance_ind, name=f'{method}')

cond_df = pd.DataFrame(cond_df) 

### save file

dfs = []

for i in cph_list.keys():
    dfs.append(cph_list[i].summary)

coef_df = pd.concat(dfs)

coef_df['p_adj'] = multipletests(coef_df['p'].tolist())[1]

### save file

### Patient survival wrt cluster prevalence per mm2 of tissue

cohort_dirs = {'Jackson-BC': ['OMEnMasks/Basel_Zuri_masks', '_a0_full_maks.tiff'],
              'Ali-BC': ['METABRIC_IMC/to_public_repository/cell_masks', '_cellmask.tiff'],
              'Hoch-Melanoma': ['full_data/protein/cpout/', '_ac_ilastik_s2_Probabilities_equalized_cellmask.tiff']}

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

full_cohort_df_scaled = pd.DataFrame(MinMaxScaler().fit_transform(full_cohort_df), index=full_cohort_df.index, columns=full_cohort_df.columns)

tissue_cluster_variables = pd.merge(
    full_cohort_df_scaled, 
    clusters_patient[[durations_col, censoring_col, strat_col]+[patient_col]], 
    on=patient_col).drop_duplicates().reset_index(drop=True)

f = plt.figure(figsize=(20,50))

cph_list = {}
concordance_ind = {}
    
for i in clusters:
    try:
        ax = f.add_subplot(10,3, i+1)

        df = tissue_cluster_variables.loc[:, [i,durations_col, censoring_col, strat_col]].dropna()

        cph = CoxPHFitter(strata=[strat_col])#

        cph.fit(df, durations_col, event_col=censoring_col)

        c_i = concordance_index(df[durations_col], -cph.predict_partial_hazard(df), df[censoring_col])

        #cph.print_summary()
        cph_list[i] = cph
        concordance_ind[i] = c_i


        cph.plot(ax=ax)
        ax.set_title(f'Cluster {i}')
    except:
        print(f'Cluster {i} gives error:', sys.exc_info()[0])

dfs = []

for i in cph_list.keys():
    dfs.append(cph_list[i].summary)

cond_df = pd.Series(concordance_ind, name=f'{method}')

cond_df = pd.DataFrame(cond_df)

 ### save file

 coef_df = pd.concat(dfs)

 coef_df['p_adj'] = multipletests(coef_df['p'].tolist())[1]

 ### save file







