library(ComplexHeatmap)
#install.packages('colorRamp2')


library(colorRamp2)

cohort <- "Hoch-Melanoma"
method <- "louvain"

ma1 = read.csv(sprintf('Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_%s_cell_types_new_cluster_means.tsv', cohort, cohort, method),
               sep='\t')

head(ma1)

#ncol(ma1)


counts = log(as.integer(ma1[,ncol(ma1)-1])+1)

last_idx = ncol(ma1)-2

lineage = matrix(ma1[,ncol(ma1)])

#lineage

ma1 = ma1[, 1:last_idx]

#ma1[1, 1:last_idx]
#head(ma1)

# Convert the transposed matrix back to a dataframe
transposed_df <- as.data.frame(t(ma1[,2:ncol(ma1)]))

# Set column names based on row names of the original dataframe
colnames(transposed_df) <- ma1[,1]

# Print the final transposed dataframe
#print("Transposed Dataframe:")
#print(transposed_df)

#colnames(transposed_df)

#rownames(transposed_df)

#head(transposed_df)



#min(ma1[, 2:ncol(ma1)])

pdf(sprintf("Desktop/imc_analysis/final_hmivae/analysis/plotting_r/%s/%s_%s_cell_types_markers_new.pdf", cohort, cohort, method),
    width = 15, height = 20)
f1 <- colorRamp2(seq(0, 1, length = 3), c(scales::muted("blue"), "#EEEEEE", scales::muted("red")), space = "RGB")
haa1 = column_ha <- HeatmapAnnotation(
  "log(Cell Counts)" = anno_barplot(counts, axis = TRUE),
  "Cell lineage" = lineage,
  col = list(
    "log(Cell Counts)" = 'gray',
    "Cell lineage" = c("Tumor" = 'pink', 'Immune' = 'darkgreen', "unknown" = 'gray', 'Endothelial' = 'darkorange', 'Stromal' = 'brown')
    #"Cell lineage" = c("Epithelial" = "purple", "Immune" = "darkgreen", "Stromal" = 'brown', "Endothelial" = 'darkorange', "unknown" = 'gray')
  ),
  border = TRUE
  )



htt1 <- Heatmap(as.matrix(transposed_df), 
                name = 'Scaled Expression',
                column_title = " ",
                col = f1, #rev(rainbow(3)),
                top_annotation = haa1,
                #bottom_annotation = haa2,
                #column_split = colnames(transposed_df),
                column_labels = colnames(transposed_df),
                cluster_rows = TRUE, 
                cluster_columns = TRUE,
                column_names_gp = gpar(fontsize = 15),
                show_heatmap_legend = TRUE, 
                border = TRUE,
                width = ncol(as.matrix(transposed_df))*unit(5, "mm"), 
                height = nrow(as.matrix(transposed_df))*unit(5, "mm"),
                cell_fun = s <- c('width' = 1, 'height' = 1),
                heatmap_legend_param = list(
                  title_position='topleft'
                )
                )

draw(htt1)


dev.off()




