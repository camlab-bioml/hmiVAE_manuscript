### comparing clinical associations across methods

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from statannotations.Annotator import Annotator # install statannotations


cofactors = {'Jackson-BC':5.0, 'Ali-BC':0.8, 'Hoch-Melanoma':1.0}

cohort_rename = {'Jackson-BC': 'basel', 'Ali-BC': 'metabric', 'Hoch-Melanoma': 'melanoma'}

resolutions = {'Jackson-BC': 0.4, 'Ali-BC': 0.9, 'Hoch-Melanoma': 0.15} #resolutions used for louvain

exp_resolutions = {'Jackson-BC': 0.14, 'Ali-BC': 0.3, 'Hoch-Melanoma': 0.16} # resolutions for louvain

cohort = 'Hoch-Melanoma'

cofactor = cofactors[cohort]

res = resolutions[cohort]

#### read in all the files

### switch the files names for the cluster proportion description

louvain_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_Louvain_cluster_tissue_per_mm2_association.tsv", 
                              sep='\t', index_col=0)

flowsom_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_FlowSOM_cluster_tissue_per_mm2_association.tsv",
                              sep='\t', index_col=0)

hmivae_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_hmiVAE_cluster_tissue_per_mm2_association.tsv",
                            sep='\t', index_col=0)

exp_louvain_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_Louvain_exp_only_cluster_tissue_per_mm2_association.tsv",
                                 sep='\t', index_col=0)

exp_flowsom_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_FlowSOM_exp_only_cluster_tissue_per_mm2_association.tsv",
                                 sep='\t', index_col=0)

exp_hmivae_patient = pd.read_csv(f"../survival_analysis/{cohort_rename[cohort]}/{cohort}_hmiVAE_exp_only_cluster_tissue_per_mm2_association.tsv",
                                 sep='\t', index_col=0)


names = ['Louvain_exp_only', 'Louvain_full', 'FlowSOM_exp_only', 'FlowSOM_full', 'hmiVAE_exp_only', 'hmiVAE_full']

dfs = []

for n, df in enumerate([exp_louvain_patient, louvain_patient, exp_flowsom_patient, flowsom_patient, exp_hmivae_patient, hmivae_patient]):
    name = names[n]
    
    df = df.abs()
    
    df = pd.DataFrame(df.max(1))
    
    df.columns = [name]
    
    dfs.append(df)


max_var_df = pd.concat(dfs, axis=1).sort_index()

### save the file 

#### Now use the saved files

cohort = 'Hoch-Melanoma' # change for each dataset you're working with

df_props = pd.read_csv(f"{cohort}_abs_max_vars_cluster_props_new.tsv", sep='\t', index_col=0)
#df_props = pd.read_csv("Ali-BC_abs_max_vars.tsv", sep='\t', index_col=0) # different file name for Ali-BC
df_tissue = pd.read_csv(f'{cohort}_abs_max_vars_tissue_prev_new.tsv', sep='\t', index_col = 0)
#df_tissue = pd.read_csv('Ali-BC_abs_max_vars_tissue_per_mm2.tsv', sep='\t', index_col = 0) # different file name for Ali-BC

col_combs = list(combinations(df_tissue.columns, 2))

new_combs = []

for i, j in col_combs:
    
    if ('hmiVAE' in i) or ('hmiVAE' in j): # get just the comparisons to hmiVAE
        
        new_combs.append((i,j))


df_tissue = pd.melt(df_tissue.reset_index(), id_vars=['clinical_variable'], value_vars=list(df_tissue.columns))

df_tissue['Method'] = ['Cluster prevalence per mm2 of tissue']*df_tissue.shape[0]

df_props = pd.melt(df_props.reset_index(), id_vars=['clinical_variable'], value_vars=list(df_props.columns))

df_props['Method'] = ['Cluster proportions']*df_props.shape[0]

full_assoc_df = pd.concat([df_props, df_tissue])


def get_log_ax(orient="v"):
    #from: https://github.com/trevismd/statannotations-tutorials/blob/main/Tutorial_1/utils.py
    if orient == "v":
        figsize = (12, 6)
        set_scale = "set_yscale"
    else:
        figsize = (10, 8)
        set_scale = "set_xscale"
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_alpha(1)
    getattr(ax, set_scale)("log")
    return ax

def add_legend(ax):
    #from: https://github.com/trevismd/statannotations-tutorials/blob/main/Tutorial_1/utils.py
    if orient == "v":
    ax.legend(loc=(1.01, 0.5))

plt.rcParams['figure.figsize'] = [10,6]

color_palette = {
    'hmiVAE_full': '#66c2a5',
    'hmiVAE_exp_only': '#fc8d62',
    'FlowSOM_full': '#8da0cb',
    'FlowSOM_exp_only': '#e78ac3',
    'Louvain_full': '#a6d854',
    'Louvain_exp_only': '#ffd92f'
}

for n, i in enumerate(full_assoc_df.Method.unique()):
    
    ax = plt.subplot(1,2,n+1)

    pairs = new_combs
    
    df = full_assoc_df.query("Method == @i")

    my_order = df.groupby(by=["variable"]).median().sort_values(by='value', ascending=False).index.tolist()
    col_order = df.groupby(by=["variable"]).median().sort_index(ascending=False).index.tolist()
    #print(my_order)

    hue_plot_params = {
        'data': df,
        'x': 'variable',
        'y': 'value',
        "palette": color_palette,
        "order": my_order,
    }


    with sns.plotting_context("paper", font_scale = 1.4):
        # Create new plot
        #ax = get_log_ax()

        # Plot with seaborn
        sns.boxplot(ax=ax, **hue_plot_params)

        # Add annotations
        annotator = Annotator(ax, pairs, verbose=1, **hue_plot_params)
        annotator.configure(test="t-test_ind").apply_and_annotate()

        # Label and show
        #add_legend(ax)
        ax.set_xlabel('')
        ax.set_ylabel('abs(t-value of association)')
        ax.set_title(i)
        ax.set_xticklabels(rotation=45, labels=my_order)

plt.tight_layout()

#### save figure


### Create similar figure for survival analysis

ci_df = pd.read_csv("all_meathods_concordance_index.tsv", sep='\t', index_col=0)

methods_rename = {'Louvain': 'Louvain_full', 'hmiVAE': 'hmiVAE_full', 'FlowSOM': 'FlowSOM_full', 'Louvain_exp_only':'Louvain_exp_only',
                 'hmiVAE_exp_only':'hmiVAE_exp_only', 'FlowSOM_exp_only': 'FlowSOM_exp_only'}

ci_df['Method'] = ci_df['Method'].map(methods_rename)

plt.rcParams['figure.figsize'] = [10,6]

color_palette = {
    'hmiVAE_full': '#66c2a5',
    'hmiVAE_exp_only': '#fc8d62',
    'FlowSOM_full': '#8da0cb',
    'FlowSOM_exp_only': '#e78ac3',
    'Louvain_full': '#a6d854',
    'Louvain_exp_only': '#ffd92f'
}

for c in ['Ali-BC', 'Jackson-BC', 'Hoch-Melanoma']:
    
    concord_df = ci_df.query("cohort==@c")

    for n, i in enumerate(full_assoc_df.Method.unique()):

        ax = plt.subplot(1,2,n+1)

        pairs = new_combs

        df = concord_df.query("Measure == @i")

        my_order = df.groupby(by=["Method"]).median().sort_values(by='C_index', ascending=False).index.tolist()
        col_order = df.groupby(by=["Method"]).median().sort_index(ascending=False).index.tolist()

        hue_plot_params = {
            'data': df,
            'x': 'Method',
            'y': 'C_index',
            "palette": color_palette,
            "order": my_order,
        }

        with sns.plotting_context("paper", font_scale = 1.4):
            # Create new plot
            #ax = get_log_ax()

            # Plot with seaborn
            sns.boxplot(ax=ax, **hue_plot_params)
            # Add annotations
            annotator = Annotator(ax, pairs, verbose=0, **hue_plot_params)
            annotator.configure(test="t-test_ind").apply_and_annotate()

            # Label and show
            #add_legend(ax)
            ax.set_xlabel('')
            ax.set_ylabel('Concordance Index')
            ax.set_title(i)
            ax.set_xticklabels(rotation=45, labels=my_order)
        #label_plot_for_subcats(ax)
        #plt.show()


    plt.tight_layout()
    #plt.savefig(f'../figures/{c}_concordance_boxplot.pdf') # save figure
    plt.close();



