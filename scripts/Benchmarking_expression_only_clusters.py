### comparing the expression only clusters to publication and ground truth (for Jackson-BC dataset only) 

import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


cohort = 'Jackson-BC'

clusters = pd.read_csv(f"{cohort}_exp_only_all_methods_new_louvain_clusters.tsv", sep='\t', index_col=0)

#clusters = pd.read_csv("Ali-BC_exp_only_resleiden1_louv0.3_all_methods_clusters.tsv", sep='\t', index_col=0) # different name for Ali-BC

clusters.head()

adata = sc.read_h5ad(f"{cohort}_adata_new.h5ad")

clusters.cell_id = clusters.cell_id.map(str)

adata.obs = pd.merge(adata.obs.reset_index()[['Sample_name', 'cell_id', 'leiden']], clusters, on=['Sample_name', 'cell_id'])

# do only for Jackson-BC dataset

if cohort == "Jackson-BC":
    new_cell_id_dfs = []

    for s in clusters.Sample_name.unique():
        
        s_df = clusters.query("Sample_name == @s").reset_index(drop=True)
        
        s_df['cell_id'] = s_df.reset_index()['index'].apply(lambda x: x+1)
        
        new_cell_id_dfs.append(s_df)

clusters = pd.concat(new_cell_id_dfs).reset_index()

# our annotations for the different methods

if cohort=='Ali-BC':
    louvain_cell_types = {
        0: 'immune',
        1: 'luminal epithelial 1',
        2: 'proliferating epithelial',
        3: 'luminal epithelial 2',
        4: 'hypoxic cells',
        5: 'GATA3+ epithelial 1',
        6: 'apoptotic epithelial',
        7: 'HER2/GATA3+ epithelial',
        8: 'PR+ epithelial 1',
        9: 'luminal epithelial 3',
        10: 'GATA3+ epithelial 2',
        11: 'basal epithelial 1',
        12: 'PR+ epithelial 2',
        13: 'luminal epithelial 4',
        14: 'HR+ epithelial',
        15: 'epithelial', #wasn't sure
        16: 'basal epithelial 2',
        17: 'HER2+ epithelial',
        18: 'luminal epithelial 5',
    }
    
    flowsom_cell_types = {
        0: 'epithelial 1', # none
        1: 'stromal',
        3: 'epithelial 2', # none
        4: 'luminal epithelial 1',
        9: 'epithelial 3', #none
        10: 'stromal 1', #none
        13: 'epithelial 4',
        16: 'luminal epithelial 2',
        17: 'stromal 2',
        18: 'none', # just one cell
    }
    
    hmivae_cell_types = {
        0: 'luminal epithelial 1',
        1: 'immune',
        2: 'fibroblasts',
        3: 'stromal 1',
        4: 'epithelial', # nothing
        5: 'luminal epithelial 2',
        6: 'luminal epithelial 3', 
        7: 'stromal 2',
        8: 'apoptotic epithelial 1',
        9: 'apoptotic epithelial 2',
        10: 't cells',
        11: 'PR+ epithelial',
        12: 'stromal 3',
        13: 'GATA3+ epithelial',
        14: 'hypoxic cells',
        15: 'stromal 4',
        16: 'proliferating epithelial',
        17: 'basal epithelial',
        18: 'b cells',
        19: 'EGFR+ cells'
    }
    
if cohort == 'Jackson-BC':
    louvain_cell_types = { #changed here but not in the file
        0: 'stromal 1',
        1: 'luminal epithelial 1',
        2: 'epithelial 1',
        3: 'hypoxic cells',
        4: 'mix 1',
        5: 'luminal epithelial 2',
        6: 'HR+/HER2+/GATA3+ luminal epithelial',
        7: 'GATA3+ luminal epithelial 1',
        8: 'basal epithelial 1',
        9: 'luminal epithelial 3',
        10: 'luminal epithelial 4',
        11: 'PR+ epithelial',
        12: 'GATA3+ luminal epithelial 2',
        13: 'mix 2',
        14: 'luminal epithelial 5',
        15: 'luminal epithelial 6',
        16: 'epithelial 2',
        17: 'apoptotic cells',
        18: 'proliferating cells',
        19: 'mix 4',
        20: 'HER2+ epithelial',
        21: 'luminal epithelial 7',
        22: 'fibroblasts',
        23: 'b cells',
        24: 'mix 3'
    }
    
    flowsom_cell_types = {
        0: 'epithelial 1',
        1: 'epithelial 2', #none
        2: 'HER2/GATA3+ luminal epithelial',
        3: 'stromal 1',
        5: 'epithelial 3', #none
        6: 'fibroblasts',
        7: 'epithelial 4',
        8: 'immune 1',
        11: 'stromal 2', #none
        13: 'epithelial 5', #none
        15: 'epithelial 6', #none
        18: 'epithelial 7', #none
        21: 'stromal 3', #none
        22: 'epithelial 8', #none
        25: 'epithelial 9', #none
        
    }
    
    hmivae_cell_types = {
        0: 'fibroblasts',
        1: 'immune',
        2: 'stromal',
        3: 'luminal epithelial 1',
        4: 'luminal epithelial 2',
        5: 'macrophages',
        6: 'hypoxic epithelial',
        7: 'basal epithelial',
        8: 'luminal epithelial 3',
        9: 'HER2/GATA3+ luminal epithelial',
        10: 'proliferating epithelial',
        11: 'endothelial',
        12: 'proliferating epithelial',
        13: 'luminal epithelial 4',
        14: 'luminal epithelial 5',
        15: 't cells',
        16: 'b cells',
        17: 'luminal epithelial 6',
        18: 'EGFR+ epithelial',
        19: 'PR+ luminal epithelial 1',
        20: 'luminal epithelial 7',
        21: 'PR+ luminal epithelial 2',
        22: 'hypoxic/proliferating epithelial', # c-Myc hi
        23: 'PR+/Slug+ luminal epithelial',
        24: 'HER2+ luminal epithelial',
        25: 'Slug+/apoptotic macrophages',
        
    }


clusters['hmivae_cell_types'] = clusters['expression_leiden'].map(hmivae_cell_types)

clusters['louvain_cell_types'] = clusters['standard_louvain'].map(louvain_cell_types)

clusters['flowsom_cell_types'] = clusters['category'].map(flowsom_cell_types)

### Comparison with ground truth (Jackson-BC dataset only)

gt_annot1 = pd.read_csv("assignments-basel_subset_for_annotation_annotator_1.tsv", sep='\t')
gt_annot2 = pd.read_csv("assignments-basel_subset_for_annotation_annotator_2.tsv", sep='\t')

names = []


for i in clusters.index:
    
    sample_lst = clusters.Sample_name[i].split('_')
    
    cellid = clusters.cell_id[i]
    
    start_name = f'{sample_lst[0]}_{sample_lst[1]}_{sample_lst[-3]}'
    
    if start_name == 'BaselTMA_SP41_218':
        sample_name = f'{sample_lst[0]}_{sample_lst[1]}_{sample_lst[-3]}_{sample_lst[-2]}_{sample_lst[-1]}_{cellid}'
        
    else:
    
        sample_name = f'{sample_lst[0]}_{sample_lst[1]}_{sample_lst[-3]}_{sample_lst[-2]}_{cellid}'
    
    names.append(sample_name)

clusters['cell_id'] = names # changing the names to merge

sub_df1 = pd.merge(gt_annot1, clusters, on=['cell_id'])

sub_df2 = pd.merge(gt_annot2, clusters, on=['cell_id'])

## convert annotations to lower case to match

sub_df1['cell_type'] = sub_df1['cell_type'].str.lower()

sub_df2['cell_type'] = sub_df2['cell_type'].str.lower()

## change the names of the annotations for comparison

annot_names = {
    'HER2/GATA3+ luminal epithelial': 'epithelial (luminal)',
    'luminal epithelial 1': 'epithelial (luminal)',
    'luminal epithelial 2': 'epithelial (luminal)',
    'luminal epithelial 3': 'epithelial (luminal)',
    'luminal epithelial 4': 'epithelial (luminal)',
    'luminal epithelial 5': 'epithelial (luminal)',
    'luminal epithelial 6': 'epithelial (luminal)',
    'macrophages': 'macrophage',
    'basal epithelial': 'epithelial (basal)',
    'basal epithelial 1': 'epithelial (basal)',
    'basal epithelial 2': 'epithelial (basal)',
    'b cells': 'b cells',
    't cells': 't cells',
    'endothelial': 'endothelial',
    'stromal': 'stromal',
    'immune': 't cells', # labels all have to match
    'immune 1': 't cells',
    'fibroblasts': 'stromal',
    'hypoxic epithelial': 'unclear',
    'proliferating epithelial': 'unclear',
    'epithelial 1': 'unclear',
    'epithelial 3': 'unclear',
    'stromal 1': 'stromal',
    'hypoxic cells': 'unclear'
}

sub_df1['hmivae_cell_types'] = sub_df1['hmivae_cell_types'].map(annot_names)
sub_df1['louvain_cell_types'] = sub_df1['louvain_cell_types'].map(annot_names)
sub_df1['flowsom_cell_types'] = sub_df1['flowsom_cell_types'].map(annot_names)

sub_df2['hmivae_cell_types'] = sub_df2['hmivae_cell_types'].map(annot_names)
sub_df2['louvain_cell_types'] = sub_df2['louvain_cell_types'].map(annot_names)
sub_df2['flowsom_cell_types'] = sub_df2['flowsom_cell_types'].map(annot_names)

## change for annotator 1 or 2

dfs = []

for m, i in list(zip(['hmiVAE', 'Louvain', 'FlowSOM'],['hmivae_cell_types', 'louvain_cell_types', 'flowsom_cell_types'])):
    report_str = classification_report(sub_df2['cell_type'].tolist(), sub_df2[i])

    report_lines = report_str.strip().split('\n')

    data = [line.split() for line in report_lines[2:-4]]


    for n, p in enumerate(data):

        for j in ['cells', '(basal)', '(luminal)', '(other)']:

            if j in p:
                data[n][0] = data[n][0] + ' ' + data[n][1]
                data[n].remove(j)

    columns = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']

    df = pd.DataFrame(data, columns=columns)

    conf_matrix = confusion_matrix(sub_df2['cell_type'], sub_df2[i])
    names = np.sort(sub_df2['cell_type'].unique()).tolist()
    num_classes = conf_matrix.shape[0]

    specificity_lst = []

    for k in range(num_classes):
        tp = conf_matrix[k, k]
        fn = sum(conf_matrix[k, :]) - tp
        fp = sum(conf_matrix[:, k]) - tp
        tn = sum(sum(conf_matrix)) - (tp + fn + fp)
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        specificity_lst.append(specificity)
        
        #print(f"Class {names[i]}: Sensitivity = {sensitivity}, Specificity = {specificity}")


    df['Specificity'] = specificity_lst

    df['Method'] = [m]*df.shape[0]


    dfs.append(df)

metric_df = pd.concat(dfs)

metric_df['Precision'] = metric_df['Precision'].map(float)
metric_df['Recall'] = metric_df['Recall'].map(float)
metric_df['F1-score'] = metric_df['F1-score'].map(float)
metric_df['Support'] = metric_df['Support'].map(float)

metric_df = metric_df[['Class', 'Precision', 'Recall', 'Specificity', 'F1-score', 'Method']].melt(id_vars=['Class', 'Method'])

g = sns.catplot(
    data=metric_df, x="Class", y="value", hue='Method', col="variable",
    kind="bar", palette='Set2', sharey=False, #col_wrap=1
)
g.set_axis_labels("", "Score")
g.set_xticklabels(list(np.sort(sub_df2['cell_type'].unique())), rotation=45)
g.set_titles("{col_name}")
g.set(ylim=(0, 1))
plt.tight_layout();
#plt.savefig("Jackson-BC_exp_only_clusters_quant_annot2.pdf"); # plot and save figure

### Comparisons with publications

# hmi_celltypes = pd.read_csv("../analysis/cluster_analysis/Ali-BC/best_run_Ali-BC_out/Ali-BC_exp_only_resleiden1_louv0.3_all_methods_clusters.tsv",
#                            sep='\t', index_col=0)

hmi_celltypes = pd.read_csv("../analysis/cluster_analysis/Jackson-BC/best_run_Jackson-BC_out/Jackson-BC_exp_only_all_methods_new_louvain_clusters.tsv",
                           sep='\t', index_col=0)

## don't need to do this for Ali-BC

new_cell_id_dfs = []

for s in hmi_celltypes.Sample_name.unique():
    
    s_df = hmi_celltypes.query("Sample_name == @s").reset_index(drop=True)
    
    s_df['cell_id'] = s_df.reset_index()['index'].apply(lambda x: x+1)
    
    new_cell_id_dfs.append(s_df)

hmi_celltypes = pd.concat(new_cell_id_dfs).reset_index()

hmi_celltypes['hmivae_cell_types'] = hmi_celltypes['expression_leiden'].map(hmivae_cell_types)

hmi_celltypes['louvain_cell_types'] = hmi_celltypes['standard_louvain'].map(louvain_cell_types)

hmi_celltypes['flowsom_cell_types'] = hmi_celltypes['category'].map(flowsom_cell_types)

## publication annotations

### For Jackson-BC dataset

# pub_annotations = pd.read_csv("../../../../../../home/campbell/share/datasets/jackson-imc/Cluster_labels/Metacluster_annotations.csv",
#                              sep=';')
# pub_clusters = pd.read_csv("../../../../../../home/campbell/share/datasets/jackson-imc/Cluster_labels/Basel_metaclusters.csv")
# pub_annot = dict(zip(pub_annotations['Metacluster '].tolist(), pub_annotations['Cell type'].tolist()))
# pub_clusters['publication_cell_types'] = pub_clusters['cluster'].map(pub_annot)
# names = []
# for i in hmi_celltypes.index:
    
#     sample_lst = hmi_celltypes.Sample_name[i].split('_')
    
#     cellid = hmi_celltypes.cell_id[i]
    
#     sample_name = f'{sample_lst[0]}_{sample_lst[1]}_{sample_lst[-3]}_{sample_lst[-2]}_{cellid}'
    
#     names.append(sample_name)
# hmi_celltypes['id'] = names


### For Ali-BC dataset

#pub_annotations = pd.read_csv("../../../../../..//home/campbell/share/datasets/ali-2020/data-raw/20191219-ftp/METABRIC_IMC/to_public_repository/single_cell_data.csv")
#cores = pd.read_csv('metabricid_to_imccore.tsv', sep='\t', index_col=0)
#cores.columns = ['metabricid', 'Sample_name']
#cores_dict = dict(zip(cores['metabricid'].tolist(), cores['Sample_name'].tolist()))
#pub_annotations['Sample_name'] = pub_annotations['metabricId'].map(cores_dict)
#pub_annotations.rename(columns={'ObjectNumber': 'cell_id'}, inplace=True)

compare_df = pd.merge(pub_clusters, hmi_celltypes, on=['id']) # merge the publication and our annotations

with open(f'{cohort}_ari_w_publication.txt', 'w+') as f:

    for i in ['hmivae_cell_types', 'louvain_cell_types', 'flowsom_cell_types']:

        ari = adjusted_rand_score(compare_df['description'].tolist(), compare_df[i])

        print(f'{i} and publication ari: {ari}')
        
        f.write(f'{i} and publication ari: {ari} \n')
    f.write(f'Number of cells compared: {compare_df.shape[0]}')

ari = {}

for cohort in ['Jackson-BC', 'Ali-BC']:

    with open(f"{cohort}_ari_w_publication.txt", 'r') as f:
        ari[cohort] = {}
        lines = f.readlines()

        for i in lines[0:3]:

            words = i.split(' ')

            ari[cohort][words[0]] = float(words[-2])

ari_df = pd.DataFrame(ari).reset_index()

ari_df = ari_df.melt(id_vars='index')

met = {'hmivae_cell_types': 'hmiVAE',
      'louvain_cell_types': 'Louvain',
      'flowsom_cell_types': 'FlowSOM'}

ari_df['Method'] = ari_df['index'].map(met)

sns.barplot(data=ari_df, x='variable', y='value', hue='Method')
plt.ylabel('Adjusted Rand Index (ARI)')
plt.xlabel('')
#plt.savefig("Comparisons_to_pubs.pdf"); # save figure
