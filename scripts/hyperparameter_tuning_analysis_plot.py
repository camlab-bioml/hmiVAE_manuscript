### for hyperparameter tuning and their effects on reconstruction

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats.mstats import winsorize
import seaborn as sns

#### Figuring out which runs were best for each cohort

cohort = 'Jackson-BC'

cohort_dir = f'../../analysis/cluster_analysis/{cohort}/hyperparams_tuning'

dirs = [i for i in os.listdir(cohort_dir) if f'{cohort}_vae_out_nh' in i] # get the dirs with the saved checkpoints

recon_liks = {}

for d in dirs:
    files = os.listdir(f"{cohort_dir}/{d}")
    
    for f in files:
        if "epoch" in f:
            epoch = f.split("_")[0].split("=")[-1]
            if int(epoch) > 0:
                lik = f.split("=")[-1].split(".ckpt")[0]

                recon_liks[d] = float(lik)

best_run = max(recon_liks, key=recon_liks.get)

### Get relationship between hyperparameters and reconstruction

n_hiddens = []
hidden_dim_size = []
latent_space_size = []
random_seed = []
batch_size = []
recon_liks = []
epoch_best = []
beta_scheme = []

dfs = []

for c in ['Jackson-BC', 'Ali-BC', 'Hoch-Melanoma']:
    cohort_dir = f'../../analysis/cluster_analysis/{c}/hyperparams_tuning'

    dirs = [i for i in os.listdir(cohort_dir) if f'{c}_vae_out_nh' in i]
    
    for d in dirs:
        files = os.listdir(f"{cohort_dir}/{d}")
        names = d.split('_')[3:]
        nh = int(names[0][2:])
        hd = int(names[1][2:])
        ls = int(names[2][2:])
        beta = names[3][4:]
        rs = int(names[4][2:])
        bs = int(names[5][2:])
        
        n_hiddens.append(nh)
        hidden_dim_size.append(hd)
        latent_space_size.append(ls)
        random_seed.append(rs)
        batch_size.append(bs)
        beta_scheme.append(beta)
        
        for f in files:
            if "epoch" in f:
                epoch = f.split("_")[0].split("=")[-1]
                lik = f.split("=")[-1].split(".ckpt")[0]
                recon_liks.append(float(lik))
                epoch_best.append(epoch)
   df = pd.DataFrame({
        'beta_scheme': beta_scheme,
        'n_hidden': n_hiddens,
        'hidden_layer_size': hidden_dim_size,
        'latent_space_size': latent_space_size,
        'random_seed': random_seed,
        'batch_size': batch_size,
        'best_epoch': epoch_best,
        'reconstruction_likelihood': recon_liks,
    })
    
    df['cohort'] = [c]*df.shape[0]
    
    dfs.append(df.set_index('cohort'))

hyperparameter_df = pd.concat(dfs).drop_duplicates().reset_index()

for c in ['Jackson-BC', 'Ali-BC', 'Hoch-Melanoma']:
    df = hyperparameter_df.loc[(hyperparameter_df.cohort == c) & (hyperparameter_df.best_epoch.map(int) > 0),:]
    print('cohort')
    print(df[df.reconstruction_likelihood == df.reconstruction_likelihood.max()])

#### Create figures

hyperparameter_df.columns = [i.replace('_', ' ').capitalize() for i in hyperparameter_df.columns]

hyperparameter_df.columns = ['Cohort', '\u03B2'+' scheme'] + list(hyperparameter_df.columns[2:])

sns.set_theme(context='paper', font='sans-serif', font_scale=1.5, style='ticks', rc={'font.size':15.0, "axes.spines.right": False, "axes.spines.top": False})

plt.rcParams['figure.figsize'] = (20,10)



counter = 0

for col in list(hyperparameter_df.columns[1:5])+[hyperparameter_df.columns[6]]:
    ax = plt.subplot(2,3, counter + 1)
    
    sns.boxplot(data=hyperparameter_df, x='Cohort', y='Reconstruction likelihood', hue=col, showfliers=False, ax=ax)
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0)
    #plt.legend(bbox_to_anchor=[1.001,1], title=col)
    
    counter += 1
    
    
plt.tight_layout()
#plt.savefig("hyperparameters_all_cohorts.pdf") # save figure
