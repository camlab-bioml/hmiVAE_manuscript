### comparing and making plots for latent space projection

from balanced_clustering import balanced_adjusted_rand_index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import scanpy as sc


sns.set_theme(context='paper', font='sans-serif', font_scale=1.5, style='ticks', rc={'font.size':15.0, "axes.spines.right": False, "axes.spines.top": False})

### get all the cluster asssignments from the saved files - switch the file names to get the other dataset and metric

adata = sc.read_h5ad('../../analysis/cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_adata_new.h5ad') # get the original clusters from hmiVAE

adata.obs.reset_index(inplace=True)

true_labels_base = pd.read_csv("Baseline_Ali_true_labels.tsv", sep='\t', index_col=0) # ground truth labels for baseline


predicted_labels_base = pd.read_csv("Baseline_AoJ_cosine.tsv", sep='\t') # predicted labels for projected dataset - baseline

full_labels_hmi = pd.read_csv("predicted_labels_AonJ_w_bkg_cosine.tsv", sep='\t', index_col=0) # predicted labels for projected dataset - hmiVAE
full_labels_hmi.reset_index(inplace=True)

full_labels_hmi.cell_id = full_labels_hmi.cell_id.map(str)

adata.obs.cell_id = adata.obs.cell_id.map(str)

full_labels_hmi = pd.merge(adata.obs.reset_index(), full_labels_hmi.reset_index(), on=['Sample_name', 'cell_id'])

true_labels_base.cell_id = true_labels_base.cell_id.map(str)
predicted_labels_base.cell_id = predicted_labels_base.cell_id.map(str)

full_labels_df = pd.merge(true_labels_base, predicted_labels_base, on=['Sample_name', 'cell_id'])

#### Calculate ARI and balanced ARI - remove from comment the relevant ones

# views = {
#     'integrated':['leiden', 'predicted_integrated'], 
# #     'expression': ['expression_leiden', 'predicted_expression'], 
# #     'correlation': ['correlation_leiden', 'predicted_correlation'], 
# #     'morphology': ['morphology_leiden', 'predicted_morphology'], 
#     'spatial_context': ['spatial_context_leiden', 'predicted_spatial_context']
# } # for hmiVAE

# views = [
#     'full', 
#     'expression', 
#     'correlation', 
#     'morphology', 
#     'spatial_context'
# ] # for baseline


ari = {}
b_ari = {}

for n, v in enumerate(views):
    
    # for baseline
    true_labels = full_labels_df[f'true_baseline_{v}_labels'].map(int).tolist()
    predicted_labels = full_labels_df[f'predicted_baseline_{v}_labels'].map(int).tolist()
    
    # for hmiVAE
#     true_labels = full_labels_hmi[views[v][0]].map(int).tolist()
#     predicted_labels = full_labels_hmi[views[v][1]].map(int).tolist()
    
    ari[v] = adjusted_rand_score(true_labels, predicted_labels)
    b_ari[v] = balanced_adjusted_rand_index(np.array(true_labels), np.array(predicted_labels))


# for baseline

ari_v_bari_base = pd.DataFrame({'ARI':list(ari.values()), 
                           'Balanced ARI': list(b_ari.values()),
                          'View': list(ari.keys())})

ari_v_bari_base = pd.melt(ari_v_bari_base, id_vars=['View'], value_vars=['ARI', 'Balanced ARI'])

ari_v_bari_base['Comparison'] = ['Baseline']*ari_v_bari_base.shape[0]

ari_v_bari_base['View'] = ari_v_bari_base['View'].str.replace('_', ' ')

ari_v_bari_base['View'] = ari_v_bari_base['View'].str.capitalize()

# for hmiVAE

ari_v_bari_hmi = pd.DataFrame({'ARI':list(ari.values()), 
                           'Balanced ARI': list(b_ari.values()),
                          'View': list(ari.keys())})

ari_v_bari_hmi = pd.melt(ari_v_bari_hmi, id_vars=['View'], value_vars=['ARI', 'Balanced ARI'])

ari_v_bari_hmi['Comparison'] = ['hmiVAE']*ari_v_bari_hmi.shape[0]

ari_v_bari_hmi['View'] = ari_v_bari_hmi['View'].str.replace('_', ' ')

ari_v_bari_hmi['View'] = ari_v_bari_hmi['View'].str.capitalize()

full_ari_v_bari = pd.concat([ari_v_bari_base, ari_v_bari_hmi])

full_ari_v_bari['View'] = full_ari_v_bari['View'].apply(lambda x: 'Full' if x=='Integrated' else x)

full_ari_v_bari['View'] = full_ari_v_bari['View']

### Save file

#### Plot using saved files

comparison = 'JoA'

full_ari_v_bari_cosine = pd.read_csv(f"{comparison}_ari_v_bari_baseline_v_hmivae_cosine.tsv", sep='\t', index_col=0)
full_ari_v_bari_cosine['Comparison'] = full_ari_v_bari_cosine['Comparison'].apply(lambda x: x + ' - Cosine')

full_ari_v_bari_euclid = pd.read_csv(f"{comparison}_ari_v_bari_baseline_v_hmivae_euclid.tsv", sep='\t', index_col=0)
full_ari_v_bari_euclid['Comparison'] = full_ari_v_bari_euclid['Comparison'].apply(lambda x: x + ' - Euclidean')

full_ari_v_bari = pd.concat([full_ari_v_bari_cosine, full_ari_v_bari_euclid]).reset_index(drop=True)
full_ari_v_bari.rename(columns={'Comparison': 'Method - Metric'}, inplace=True)

full_ari_v_bari['View'] = full_ari_v_bari['View'].map({'Full': 'Full', 'Expression': 'Expression', 'Correlation': 'Nuclear co-localization',
                                              'Morphology': 'Morphology', 'Spatial context': 'Spatial context'})

custom_dict = {'Full': 0, 'Expression': 1, 'Nuclear co-localization': 2, 'Morphology': 3, 'Spatial context': 4}

full_ari_v_bari['rank'] = full_ari_v_bari['View'].map(custom_dict)

full_ari_v_bari.sort_values(by=['Method - Metric', 'rank'], inplace=True)

f = plt.figure(figsize=(17,10))
comp = {'AoJ': 'Ali-BC projected on Jackson-BC', 'JoA': 'Jackson-BC projected on Ali-BC'}

for n, i in enumerate(full_ari_v_bari.variable.unique()):
    ax = f.add_subplot(1,2,n+1)
    df = full_ari_v_bari[full_ari_v_bari.variable == i]
    sns.barplot(data=df, x='View', y='value', hue='Method - Metric', ax=ax, palette='Set2')
    ax.set_title(f'{comp[comparison]}')
    ax.set_ylim([0, 0.9])
    ax.set_ylabel(i)
    ax.set_xlabel(' ')
    ax.set_xticklabels(labels= df['View'].unique().tolist(), rotation=45, ha='right')   
plt.tight_layout()
#plt.savefig(f'baseline_v_hmi_{comparison}_all.pdf') # save the figure

