#install.packages('forestploter')
#install.packages('patchwork')

library(grid)
library(forestploter)
library(magrittr)
library(dplyr)

cohort <- "Jackson-BC"
method <- "Louvain_exp_only_new"

dt <- read.csv(sprintf('Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_%s_stage_strat_cluster_props_coxph_coeffs.tsv', cohort, cohort, method), sep='\t')

colnames(dt)

dt <- rename(dt, Cluster=covariate, 'p_adj*'=p_adj)

dt$` ` <- paste(rep(" ", 20), collapse = " ")

dt$`HR (95% CI)` <- ifelse(is.na(dt$se.coef), "",
                           sprintf("%.2f (%.2f to %.2f)",
                                   dt$coef, dt$coef.lower.95, dt$coef.upper.95))

dt <- dt %>% mutate_if(~is.numeric(.), ~round(., digits = 2))



p <- forest(dt[,c(1, 13:15)],
            est = dt$coef,
            lower = dt$coef.lower.95, 
            upper = dt$coef.upper.95,
            sizes = dt$se.coef,
            ci_column = 3,
            ref_line = 0,
            arrow_lab = c("Lower risk", "Higher risk"),
            xlim = c(-4, 4),
            ticks_at = c(-4, -3, -2, -1, 0, 1, 2, 3, 4),
            title = sprintf('%s (stratified on Stage)', method)
            #footnote = "This is the demo data. Please feel free to change\nanything you want."
            )


ggplot2::ggsave(filename = sprintf("Desktop/imc_analysis/final_hmivae/analysis/plotting_r/figure3_clin/%s_%s_%s_stage_strat_cluster_props.pdf", cohort, cohort, method),
                plot = p,
                width = 7.5, height = 7.5, units = "in") # save figure
