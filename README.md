# hmiVAE_manuscript  

Scripts for the hmiVAE manuscript. This repository has the following structure:
```
|
|__ pipeline/ # contains scripts for hmiVAE runs 
    |__ run/
        |__ latent_space_proj/
            |__ make_proj_umaps.py # creates the adatas for latent space projection analysis
        |__ Snakefile # calls the make_umaps.py script over best hyperparameters for all datasets
        |__ make_umaps.py # creates the adatas with clustering + umaps for the best performing combination of hyperparameters 
    |__ Snakefile # master Snakefile that runs prep_input.smk and hyperparametering_tuning.smk
    |__ full_config.yaml # config file for Snakefile
    |__ hyperparameter_tuning.py # runs hmiVAE for a given combination of hyperparameters
    |__ hyperparameter_tuning.smk # calls the hyperparameter_tuning.py script over all combinations of hyperparameters for all datasets 
    |__ prep_input.smk # calls the vae_data_prep_* scripts and runs them over all samples and datasets
    |__ vae_data_prep_concat.py # concats all samples together for our input 
    |__ vae_data_prep_sample.py # creates the views for each sample
|
|_  r_plotting/ # contains scripts for creating plots in R
    |__ all_features_ranked_plot.R
    |__ cell_type_plotting.R
    |__ coxph_plots.R
|
|__ scripts/ # contains scripts for all analysis and plots created in Python
    |__ Benchmark_expression_only_clusters.py
    |__ Benchmark_w_FlowSOM_n_Louvain.py
    |__ Clinical_associations.py
    |__ Clinical_associations_compare.py
    |__ Ranking_features_per_view.py
    |__ Survival_association.py
    |__ hyperparameter_tuning_analysis_plot.py
    |__ latent_space_projection_analysis.py
    |__ latent_space_projection_plotting.py
|
|__ README.md
```
    
