library(data.table)


cohort <- 'Jackson-BC'
method <- 'category'



ma1 = read.csv(sprintf('Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_%s_int_clusters_rank_plot.tsv', cohort, cohort, method),
               sep='\t', row.names = 1, check.names = FALSE)

# cell.props = read.csv(sprintf('Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_int_clusters_cell_type_props.tsv', cohort, cohort),
#                       sep='\t', row.names = 1)
# 
# head(cell.props)

head(ma1)
tail(ma1)
news <- colnames(ma1)[2:ncol(ma1)]
news

colnames(ma1[,2:ncol(ma1)])

# views = matrix(ma1[nrow(ma1),2:ncol(ma1)])
# views[0:5]
# # Convert the transposed matrix back to a dataframe
transposed_df <- data.frame(t(ma1), check.names=FALSE)
#rownames(transposed_df) <- make.unique(rownames(transposed_df))
#rownames(transposed_df) <- colnames(ma1)
colnames(transposed_df)
rownames(transposed_df)

views = matrix(transposed_df[2:nrow(transposed_df),ncol(transposed_df)])

views

transposed_df <- transposed_df[2:nrow(transposed_df), 1:ncol(transposed_df)-1]

colnames(transposed_df)
head(transposed_df)

rownames(transposed_df)

#colnames(transposed_df) <- as.character(colnames(transposed_df))


# 
# # Set column names based on row names of the original dataframe
# colnames(transposed_df) <- ma1[1:nrow(ma1)-1,1]
# 
# # Print the final transposed dataframe
# print("Transposed Dataframe:")
# print(transposed_df)
# 
# 
# 
tt_df = as.data.frame(t(transposed_df))

tt_df <- sapply(tt_df, as.numeric )

rownames(tt_df) <- colnames(transposed_df)
colnames(tt_df) <- news
head(tt_df)
colnames(tt_df)
as.matrix(tt_df)

#help("Heatmap")


pdf(sprintf("Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_%s_int_clusters_markers.pdf", cohort, cohort, method),
    width = 25, height = 20)
f1 <- colorRamp2(seq(-50, 50, length = 3),  c(scales::muted("blue"), "#EEEEEE", scales::muted("red")), space = "RGB")
haa1 = column_ha <- HeatmapAnnotation("View" = views,
                                      col = list(
                                        "View" = c("Expression" = "darkorange", "Nuclear Co-localization" = "darkgreen", "Morphology" = 'brown', "Spatial Context - E" = 'purple',
                                                   "Spatial Context - NC" = 'magenta2', "Spatial Context - M" = 'orchid4')
                                      ),
                                      border = TRUE,
                                      annotation_legend_param = list(
                                        title_gp = gpar(fontsize = 12, fontface = "bold"), 
                                        labels_gp = gpar(fontsize = 12)
                                        )
                                      )
# haa2 = row_ha <- rowAnnotation(
#   "Major Cell Type" = anno_barplot(cell.props, gp = gpar(fill = 2:15), 
#                      bar_width = 1, height = unit(6, "cm")),
#   show_annotation_name = TRUE
# )

# col = list(
#   "Major Cell Type" = c("Immune" = 'chocolate', 'Epithelial' = 'mediumturquoise', 'Stromal' = 'hotpink', 'unknown' = 'gray', 'Endothelial' = 'seagreen1', 'Tumor' = 'slateblue2')
# ),
htt1 <- Heatmap(as.matrix(tt_df), 
                name = 't-value',
                column_title = " ",
                col = f1, #rev(rainbow(3)),
                bottom_annotation = haa1,
                #column_split = colnames(ma1),
                column_labels = colnames(tt_df),
                cluster_rows = TRUE, 
                cluster_columns = TRUE,
                column_dend_side = 'top',
                column_names_gp = gpar(fontsize = 15),
                show_heatmap_legend = TRUE, 
                border = TRUE,
                width = ncol(as.matrix(tt_df))*unit(5, "mm"), 
                height = nrow(as.matrix(tt_df))*unit(5, "mm"),
                cell_fun = s <- c('width' = 1, 'height' = 1),
                heatmap_legend_param = list(
                  title_position='topleft',
                  title_gp = gpar(fontsize = 12, fontface = "bold")
                )
)
#htt1

draw(htt1)
dev.off()