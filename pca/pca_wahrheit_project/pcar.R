# =========================================================================
# Principal Component Analysis (PCA) of NCI60 Cancer Data
# Author: Behrooz Filzadeh

# Description: This script performs PCA on the NCI60 gene expression dataset,
#              visualizes the results including scree plots, score plots,
#              and biplots, ensuring plots are displayed sequentially AND
#              saved to a dedicated folder.
# =========================================================================

# --- 0. Configuration ---
SAVE_PLOTS <- TRUE          # Set to TRUE to save plots, FALSE to just display
PLOT_DIR <- "pca_plot"     # Name of the directory to save plots
PLOT_WIDTH <- 8            # Default plot width in inches
PLOT_HEIGHT <- 6           # Default plot height in inches
PLOT_RES <- 300            # Plot resolution in DPI

# --- 1. Setup: Load Libraries and Data ---

# Ensure required libraries are installed and loaded
# install.packages("ISLR2") # Run once if not installed
# install.packages("RColorBrewer") # Run once if not installed
library(ISLR2)
library(RColorBrewer)

# Load the NCI60 dataset
# The dataset contains:
# - data: Gene expression matrix (64 samples x 6830 genes)
# - labs: Cancer type labels for each sample
data(NCI60)

# Extract data and labels
nci_data <- NCI60$data
nci_labs <- NCI60$labs

# Basic Data Checks (Optional but Recommended)
cat("--- Data Overview ---\n")
cat("Dimensions of gene expression data:", dim(nci_data), "\n")
cat("Number of samples:", nrow(nci_data), "\n")
cat("Number of genes:", ncol(nci_data), "\n")
cat("Length of labels:", length(nci_labs), "\n")
cat("Unique cancer types found:", length(unique(nci_labs)), "\n")
print(table(nci_labs)) # Show counts per cancer type
cat("---------------------\n\n")

# Stop if data dimensions/labels mismatch (basic sanity check)
stopifnot(nrow(nci_data) == length(nci_labs))

# --- 2. Perform PCA ---

cat("--- Performing PCA ---\n")
# Perform PCA, scaling variables to have unit variance and centering them.
# This is standard practice when variables are measured on different scales.
pca_results <- prcomp(nci_data, scale = TRUE, center = TRUE)
cat("PCA calculation complete.\n")
# Display PCA summary (variance explained by components)
print(summary(pca_results))
cat("---------------------\n\n")

# --- 3. Helper Function for Colors ---

# Function to generate distinct colors for cancer types
# Uses RColorBrewer if available for better palettes, otherwise uses base R colors.
get_cancer_colors <- function(types) {
  type_factors <- factor(types)
  unique_types <- levels(type_factors)
  n_types <- length(unique_types)
  
  # Check if RColorBrewer is loaded (it should be from Section 1)
  if (requireNamespace("RColorBrewer", quietly = TRUE) && n_types <= 12) {
    # Choose a qualitative palette suitable for categorical data
    palette_name <- ifelse(n_types <= 9, "Set1", ifelse(n_types <= 12, "Paired", "Set3")) # Adjusted palette choices
    # Ensure we request at least 3 colors for brewer.pal
    # Handle cases where n_types is less than the minimum palette size
    min_colors_for_palette <- RColorBrewer::brewer.pal.info[palette_name, "mincolors"]
    colors_needed <- RColorBrewer::brewer.pal(max(min_colors_for_palette, n_types), palette_name)[1:n_types]
  } else {
    # Fallback to base R rainbow colors if RColorBrewer isn't installed
    # or if there are too many types for standard palettes.
    colors_needed <- rainbow(n_types)
    if (n_types > 12) {
      warning(paste("Using rainbow colors for", n_types, "cancer types. Consider grouping types if colors are hard to distinguish."))
    }
  }
  
  color_map <- colors_needed
  names(color_map) <- unique_types
  # Return a vector of colors corresponding to the input 'types' vector
  return(color_map[as.character(type_factors)])
}


# Generate colors for the plots
plot_colors <- get_cancer_colors(nci_labs)
unique_cancer_types <- levels(factor(nci_labs))
# Regenerate legend colors using the function to ensure consistency
legend_colors_map <- get_cancer_colors(unique_cancer_types)
names(legend_colors_map) <- unique_cancer_types

# --- 4. Plotting Functions ---
# NOTE: These functions now focus *only* on drawing the plot to the *current* device.
# Saving is handled externally in section 5.

# Function to plot Scree Plot (Variance Explained)
plot_scree <- function(pca_res, num_pcs_to_show = 20) {
  cat("--- Plot 1: Scree Plot (Variance Explained) ---\n")
  sdev <- pca_res$sdev
  variance_explained <- sdev^2 / sum(sdev^2)
  cumulative_variance <- cumsum(variance_explained)
  n_pcs_total <- length(variance_explained)
  n_pcs_plot <- min(num_pcs_to_show, n_pcs_total)
  
  par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1), oma = c(0, 0, 0, 0)) # Reset to default single plot
  plot(1:n_pcs_plot, variance_explained[1:n_pcs_plot],
       type = "b", # Points connected by lines
       pch = 19, col = "blue",
       xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       ylim = c(0, max(variance_explained[1:n_pcs_plot], 0.1) * 1.1), # Ensure ylim starts at 0
       main = "Scree Plot: Variance Explained by Each PC")
  lines(1:n_pcs_plot, cumulative_variance[1:n_pcs_plot],
        type = "b", pch = 17, col = "red")
  abline(h=0.9, col="darkgreen", lty=2) # 90% threshold
  abline(h=0.8, col="orange", lty=2)    # 80% threshold
  legend("right",
         legend = c("Individual Variance", "Cumulative Variance", "90% Threshold", "80% Threshold"),
         col = c("blue", "red", "darkgreen", "orange"),
         pch = c(19, 17, NA, NA),
         lty = c(1, 1, 2, 2),
         cex = 0.8, bty = "n")
  cat("Scree plot drawn to current device.\n\n")
}

# Function to plot PCA scores (Samples)
plot_pca_scores <- function(pca_res, pc_x_idx, pc_y_idx, colors, sample_labels, title_suffix) {
  cat(paste0("--- Plot 2: PCA Scores (PC", pc_x_idx, " vs PC", pc_y_idx, ") ---\n"))
  scores <- pca_res$x
  variance_explained <- (pca_res$sdev^2 / sum(pca_res$sdev^2)) * 100
  
  xlab_text <- sprintf("PC%d (%.1f%% Variance)", pc_x_idx, variance_explained[pc_x_idx])
  ylab_text <- sprintf("PC%d (%.1f%% Variance)", pc_y_idx, variance_explained[pc_y_idx])
  main_title <- paste("PCA Scores:", title_suffix)
  
  par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 8.1), xpd = FALSE) # Adjust right margin for legend
  plot(scores[, pc_x_idx], scores[, pc_y_idx],
       col = colors,
       pch = 19,
       xlab = xlab_text,
       ylab = ylab_text,
       main = main_title)
  abline(h=0, v=0, lty=2, col="grey")
  
  # Add legend outside plot area
  unique_types <- levels(factor(sample_labels))
  # Use the pre-generated legend_colors_map for consistent colors
  current_legend_colors <- legend_colors_map[unique_types]
  legend("topright", inset = c(-0.25, 0), # Position legend outside the right edge
         legend = unique_types,
         fill = current_legend_colors, # Use map lookup
         title = "Cancer Type",
         cex = 0.7, bty = "n", xpd = TRUE) # xpd=TRUE allows drawing outside plot region
  cat(paste0("Score plot for PC", pc_x_idx, " vs PC", pc_y_idx, " drawn to current device.\n\n"))
}

# Function to plot a "Meaningful" Biplot (Top N variables)
plot_meaningful_biplot <- function(pca_res, sample_labels, n_vars_to_show = 15, pc_dims = c(1, 2)) {
  cat(paste0("--- Plot 3: Meaningful Biplot (Top ", n_vars_to_show, " Variables on PC", pc_dims[1], "/PC", pc_dims[2], ") ---\n"))
  scores <- pca_res$x
  loadings <- pca_res$rotation
  variance_explained <- (pca_res$sdev^2 / sum(pca_res$sdev^2)) * 100
  pc_x_idx <- pc_dims[1]
  pc_y_idx <- pc_dims[2]
  
  # Calculate magnitude of loadings for the chosen PCs
  loading_magnitudes <- sqrt(loadings[, pc_x_idx]^2 + loadings[, pc_y_idx]^2)
  # Get indices of top N variables
  top_var_indices <- order(loading_magnitudes, decreasing = TRUE)[1:min(n_vars_to_show, nrow(loadings))]
  top_var_names <- rownames(loadings)[top_var_indices]
  # Extract scores and loadings for the plot
  pc_scores_subset <- scores[, pc_dims]
  top_loadings_subset <- loadings[top_var_indices, pc_dims]
  
  # Prepare labels and colors
  xlab_text <- sprintf("PC%d (%.1f%% Variance)", pc_x_idx, variance_explained[pc_x_idx])
  ylab_text <- sprintf("PC%d (%.1f%% Variance)", pc_y_idx, variance_explained[pc_y_idx])
  main_title <- paste("Biplot (Top", length(top_var_indices), "Genes on PC", pc_x_idx, "&", pc_y_idx, ")")
  plot_cols <- get_cancer_colors(sample_labels) # Get colors for individual points
  unique_types <- levels(factor(sample_labels))
  # Use the pre-generated legend_colors_map for consistent colors
  current_legend_colors <- legend_colors_map[unique_types]
  
  # Use biplot function with custom settings for clarity
  par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 8.1), xpd = FALSE) # Margin for legend
  
  # biplot() can be tricky; manual plotting gives more control but biplot() is convenient
  # Using biplot() but suppressing default sample labels/points (`xlabs=`)
  # and then overlaying colored points. `scale=0` emphasizes variable relationships.
  biplot(pc_scores_subset, top_loadings_subset,
         scale = 0, # Emphasizes variable correlations (angles between arrows)
         xlabs = rep(".", nrow(pc_scores_subset)), # Use dots as placeholders, easier to overplot
         ylabs = top_var_names,              # Show top variable names
         cex = c(0.6, 0.7),                  # Size for placeholder dots and variable labels
         col = c("grey", "darkred"),         # Color for placeholders and variable arrows/labels
         main = main_title,
         xlab = xlab_text,
         ylab = ylab_text,
         arrow.len = 0.08)
  
  # Overplot the samples with correct colors
  points(pc_scores_subset[, 1], pc_scores_subset[, 2],
         pch = 19,
         col = plot_cols) # Use the color vector matching sample order
  abline(h=0, v=0, lty=2, col="grey")
  
  # Add legend outside plot area
  legend("topright", inset = c(-0.25, 0), # Position legend outside the right edge
         legend = unique_types,
         fill = current_legend_colors, # Use map lookup
         title = "Cancer Type",
         cex = 0.7, bty = "n", xpd = TRUE)
  cat("Meaningful biplot drawn to current device.\n\n")
}


# Function to plot Biplot Scaling Comparison (manual implementation)
plot_biplot_scaling_comparison <- function(pca_res, sample_labels, top_var_indices, pc_dims = c(1, 2)) {
  cat("--- Plot 4: Biplot Scaling Comparison (Manual) ---\n")
  scores <- pca_res$x
  loadings <- pca_res$rotation
  sdev <- pca_res$sdev
  variance_explained <- (sdev^2 / sum(sdev^2)) * 100
  pc_x_idx <- pc_dims[1]
  pc_y_idx <- pc_dims[2]
  
  # Extract relevant data
  pc_scores_subset <- scores[, pc_dims]
  top_loadings_subset <- loadings[top_var_indices, pc_dims]
  top_var_names <- rownames(loadings)[top_var_indices]
  plot_cols <- get_cancer_colors(sample_labels) # Get colors for individual points
  
  # Prepare labels
  xlab_text <- sprintf("PC%d (%.1f%%)", pc_x_idx, variance_explained[pc_x_idx])
  ylab_text <- sprintf("PC%d (%.1f%%)", pc_y_idx, variance_explained[pc_y_idx])
  
  # Set up multi-panel plot
  # We also increase outer margins (oma) to make space for a main title
  par(mfrow = c(1, 3), mar = c(4.5, 4, 3, 1), oma = c(0, 0, 3, 0)) # Inner margins, Outer margins (top=3 for title)
  
  # Determine a common scaling factor for arrows across plots for better comparison
  # Heuristic: scale arrows so their average max extent is somewhat comparable to scores max extent
  arrow_scale_factor <- mean(apply(abs(pc_scores_subset), 2, max)) / mean(apply(abs(top_loadings_subset), 2, max)) * 0.8
  
  # Loop through scaling types (manual interpretation based on common biplot goals)
  for (sc in 0:1) { # Only showing scale=0 and scale=1 as they are most common interpretations
    scores_scaled <- pc_scores_subset
    loadings_scaled <- top_loadings_subset
    main_title_panel <- ""
    
    if (sc == 0) {
      # Scale = 0: Emphasis on variables (angles between arrows approx. correlations)
      main_title_panel <- "Scale 0: Variable Focus"
      # No scaling applied here for simplicity in manual plot
    } else if (sc == 1) {
      # Scale = 1: Emphasis on samples (distances between points approx. Mahalanobis distance)
      main_title_panel <- "Scale 1: Sample Focus (Approx.)"
      # We visually emphasize samples by making arrows relatively smaller
      loadings_scaled <- loadings_scaled / sqrt(nrow(scores)) # Example scaling down
    }
    
    # Determine plot limits based on current scaled data
    # Multiply loadings by arrow_scale_factor for plotting, include 0
    # Add buffer for labels
    xlims <- range(scores_scaled[, 1], loadings_scaled[, 1] * arrow_scale_factor * 1.3, 0) * 1.1
    ylims <- range(scores_scaled[, 2], loadings_scaled[, 2] * arrow_scale_factor * 1.3, 0) * 1.1
    
    # Plot samples
    plot(scores_scaled,
         xlab = xlab_text,
         ylab = if(sc==0) ylab_text else "", # Y-label only on the first plot
         main = main_title_panel,
         col = plot_cols, # Use color vector matching sample order
         pch = 19,
         cex = 0.8,
         xlim = xlims,
         ylim = ylims)
    abline(h = 0, v = 0, lty = 2, col = "grey")
    
    # Plot variable arrows
    arrows(0, 0,
           x1 = loadings_scaled[, 1] * arrow_scale_factor,
           y1 = loadings_scaled[, 2] * arrow_scale_factor,
           col = "darkred", length = 0.08)
    
    # Plot variable labels
    text(loadings_scaled[, 1] * arrow_scale_factor * 1.15, # Slightly offset from arrow tip
         loadings_scaled[, 2] * arrow_scale_factor * 1.15,
         labels = top_var_names,
         col = "darkred", cex = 0.6)
  }
  
  # Add a placeholder plot for the third panel (optional, or plot scale=2 if defined)
  # Or add legend here if space allows
  plot.new()
  plot.window(xlim=c(0,1), ylim=c(0,1)) # Setup coordinate system
  text(0.5, 0.9, "Scale 2 (Not Shown Here)", cex=1.2)
  # Add shared legend
  unique_types <- levels(factor(sample_labels))
  current_legend_colors <- legend_colors_map[unique_types]
  legend("center", legend = unique_types, fill = current_legend_colors,
         title = "Cancer Type", cex = 0.7, bty = "n", ncol = 2)
  
  
  # Add an overall title to the multi-panel figure
  mtext("Biplot Scaling Comparison (Manual Interpretation)", outer = TRUE, cex = 1.2, line = 1)
  
  # Reset plotting parameters after the multi-panel plot
  # This will be done externally after saving the plot
  
  cat("Biplot scaling comparison plots drawn to current device.\n\n")
}


# --- 5. Generate and Save Plots Sequentially ---

if (SAVE_PLOTS) {
  # Create the directory if it doesn't exist
  if (!dir.exists(PLOT_DIR)) {
    dir.create(PLOT_DIR, showWarnings = FALSE)
    cat(paste("Created directory:", PLOT_DIR, "\n"))
  } else {
    cat(paste("Directory already exists:", PLOT_DIR, "\n"))
  }
}

# --- Plot 1: Scree Plot ---
cat("Generating Plot 1: Scree Plot...\n")
if (SAVE_PLOTS) {
  png(filename = file.path(PLOT_DIR, "01_scree_plot.png"), width = PLOT_WIDTH, height = PLOT_HEIGHT, units = "in", res = PLOT_RES)
  plot_scree(pca_results)
  dev.off()
  cat("Saved: 01_scree_plot.png\n")
} else {
  plot_scree(pca_results) # Just display if not saving
}
readline(prompt="Press [Enter] to continue to the next plot...") # Pause for user

# --- Plot 2a: Scores Plot (PC1 vs PC2) ---
cat("Generating Plot 2a: PCA Scores (PC1 vs PC2)...\n")
if (SAVE_PLOTS) {
  # Needs slightly wider plot due to external legend
  png(filename = file.path(PLOT_DIR, "02a_pca_scores_pc1_vs_pc2.png"), width = PLOT_WIDTH + 1, height = PLOT_HEIGHT, units = "in", res = PLOT_RES)
  plot_pca_scores(pca_results, pc_x_idx = 1, pc_y_idx = 2,
                  colors = plot_colors, sample_labels = nci_labs,
                  title_suffix = "PC1 vs PC2")
  dev.off()
  cat("Saved: 02a_pca_scores_pc1_vs_pc2.png\n")
} else {
  plot_pca_scores(pca_results, pc_x_idx = 1, pc_y_idx = 2,
                  colors = plot_colors, sample_labels = nci_labs,
                  title_suffix = "PC1 vs PC2")
}
readline(prompt="Press [Enter] to continue to the next plot...") # Pause for user

# --- Plot 2b: Scores Plot (PC1 vs PC3) ---
cat("Generating Plot 2b: PCA Scores (PC1 vs PC3)...\n")
if (SAVE_PLOTS) {
  # Needs slightly wider plot due to external legend
  png(filename = file.path(PLOT_DIR, "02b_pca_scores_pc1_vs_pc3.png"), width = PLOT_WIDTH + 1, height = PLOT_HEIGHT, units = "in", res = PLOT_RES)
  plot_pca_scores(pca_results, pc_x_idx = 1, pc_y_idx = 3,
                  colors = plot_colors, sample_labels = nci_labs,
                  title_suffix = "PC1 vs PC3")
  dev.off()
  cat("Saved: 02b_pca_scores_pc1_vs_pc3.png\n")
} else {
  plot_pca_scores(pca_results, pc_x_idx = 1, pc_y_idx = 3,
                  colors = plot_colors, sample_labels = nci_labs,
                  title_suffix = "PC1 vs PC3")
}
readline(prompt="Press [Enter] to continue to the next plot...") # Pause for user

# --- Plot 3: Meaningful Biplot (Top Variables on PC1/PC2) ---
cat("Generating Plot 3: Meaningful Biplot...\n")
if (SAVE_PLOTS) {
  # Needs slightly wider plot due to external legend
  png(filename = file.path(PLOT_DIR, "03_meaningful_biplot_pc1_vs_pc2.png"), width = PLOT_WIDTH + 1, height = PLOT_HEIGHT, units = "in", res = PLOT_RES)
  plot_meaningful_biplot(pca_results, sample_labels = nci_labs, n_vars_to_show = 15, pc_dims = c(1, 2))
  dev.off()
  cat("Saved: 03_meaningful_biplot_pc1_vs_pc2.png\n")
} else {
  plot_meaningful_biplot(pca_results, sample_labels = nci_labs, n_vars_to_show = 15, pc_dims = c(1, 2))
}
readline(prompt="Press [Enter] to continue to the next plot...") # Pause for user

# --- Plot 4: Biplot Scaling Comparison (Manual Approach) ---
cat("Generating Plot 4: Biplot Scaling Comparison...\n")
# We need the indices of the top variables first for the comparison plot
n_vars_comp <- 15
loading_mags <- sqrt(pca_results$rotation[, 1]^2 + pca_results$rotation[, 2]^2)
top_indices_comp <- order(loading_mags, decreasing = TRUE)[1:min(n_vars_comp, nrow(pca_results$rotation))]

if (SAVE_PLOTS) {
  # This plot is wide due to mfrow = c(1, 3)
  png(filename = file.path(PLOT_DIR, "04_biplot_scaling_comparison.png"), width = 15, height = 6, units = "in", res = PLOT_RES)
  plot_biplot_scaling_comparison(pca_results, sample_labels = nci_labs, top_var_indices = top_indices_comp, pc_dims = c(1, 2))
  dev.off() # Close the PNG device
  cat("Saved: 04_biplot_scaling_comparison.png\n")
} else {
  plot_biplot_scaling_comparison(pca_results, sample_labels = nci_labs, top_var_indices = top_indices_comp, pc_dims = c(1, 2))
}
# No readline needed after the last plot

# --- 6. End of Script ---
cat("--- Analysis and Plotting Complete ---\n")
if (SAVE_PLOTS) {
  cat("Plots saved in directory:", PLOT_DIR, "\n")
}

# Optional: Reset graphics parameters explicitly after all plotting
par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1), oma = c(0, 0, 0, 0), xpd = FALSE)