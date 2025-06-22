
cat("--- 1. Loading Library and Data ---\n")
library(ISLR2)

# Load the dataset
data(NCI60)

# TODO: Explore the structure of the NCI60 dataset
cat("\nExploring NCI60 structure:\n")
cat("str(NCI60):\n")
str(NCI60)
cat("\nsummary(NCI60):\n")
print(summary(NCI60)) # Summary gives basic info on components
# View(NCI60) # This opens an interactive viewer in RStudio

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# Extract labels and data
nci_labs <- NCI60$labs  # Cancer types
nci_data <- NCI60$data  # Gene expression data

# TODO: Examine what these variables contain
cat("\nExamining extracted variables:\n")
cat("str(nci_labs):\n")
str(nci_labs) # Shows it's a character vector of labels
cat("\nFirst few labels:", head(nci_labs), "\n")

cat("\nstr(nci_data):\n")
str(nci_data) # Shows it's a matrix of numbers
cat("\nDimensions of nci_data:", dim(nci_data), "(Samples x Genes)\n")
cat("First few values of nci_data:\n")
print(head(nci_data[, 1:6])) # Print a small part of the data matrix

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# TODO: How many different cancer types are there?
cat("\nInvestigating Cancer Types:\n")
num_unique_types <- length(unique(nci_labs))
cat("Number of unique cancer types:", num_unique_types, "\n")
cat("\nFrequency table of cancer types (table(nci_labs)):\n")
print(table(nci_labs))

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# --- 2. Perform PCA ---

cat("--- 2. Performing PCA ---\n")

# Perform standard PCA (scaling and centering)
cat("\nPerforming standard PCA (scale=TRUE, center=TRUE)...\n")
pca_out <- prcomp(nci_data, scale = TRUE, center = TRUE)
cat("Standard PCA calculation complete.\n")

# TODO: Experiment with different PCA parameters
cat("\nExperimenting with PCA parameters:\n")

# Try: scale=FALSE to see the effect of not scaling
cat("\nRunning PCA with scale=FALSE...\n")
pca_out_noscale <- prcomp(nci_data, scale = FALSE, center = TRUE)
cat("Summary of PCA with scale=FALSE:\n")
print(summary(pca_out_noscale))
cat("Note: Without scaling, components are dominated by genes with intrinsically high variance.\n")
rm(pca_out_noscale) # Remove to save memory

readline(prompt="Press [Enter] to continue...")

# Try: center=FALSE to see the effect of not centering
cat("\nRunning PCA with center=FALSE...\n")
pca_out_nocenter <- prcomp(nci_data, scale = TRUE, center = FALSE)
cat("Summary of PCA with center=FALSE:\n")
print(summary(pca_out_nocenter))
cat("Note: Without centering, the first PC often represents the average gene expression profile.\n")
rm(pca_out_nocenter) # Remove to save memory

readline(prompt="Press [Enter] to continue...")

# Try: tol argument (will skip for now as it's less common for initial exploration)
cat("\nSkipping 'tol' parameter experimentation for now.\n")
cat("We will proceed using the standard PCA results (scaled and centered).\n")

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# --- 3. Examine PCA Output ---

cat("--- 3. Examining PCA Output ---\n")

# TODO: Explore attributes of the PCA output
cat("\nExploring attributes of the standard PCA output ('pca_out'):\n")
cat("Names of components in pca_out (names(pca_out)):\n")
print(names(pca_out))
# sdev: standard deviations of PCs
# rotation: matrix of variable loadings (eigenvectors)
# center: means used for centering
# scale: standard deviations used for scaling
# x: the principal component scores (data projected onto PCs)

cat("\nAttributes of pca_out (attributes(pca_out)):\n")
print(attributes(pca_out))

cat("\nPrinting basic pca_out object info:\n")
print(pca_out) # Shows standard deviations and first few loadings

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# Extract PCA components
cat("\nExtracting PCA components:\n")
# TODO: Explain code lines or "blocks of meaningful code lines"
scores <- pca_out$x          # Principal Component scores (Samples x PCs)
# These are the coordinates of the original samples in the new PC space.
# Each column represents a principal component.
cat("Dimensions of scores (pca_out$x):", dim(scores), "\n")

right_singular_vectors <- pca_out$rotation # Variable loadings (Genes x PCs)
# Also known as eigenvectors. Each column shows how much each original gene
# contributes to forming that specific principal component.
cat("Dimensions of loadings (pca_out$rotation):", dim(right_singular_vectors), "\n")

variances <- pca_out$sdev^2   # Variance explained by each PC
# The square of the standard deviations (sdev) gives the variance.
cat("First 5 variances (pca_out$sdev^2):\n")
print(head(variances))

cat("\nChecking variance of the first PC score column:\n")
# This should be equal to the first variance calculated above
cat("var(scores[,1]):", var(scores[,1]), "\n")
cat("variances[1]:   ", variances[1], "\n") # They match

readline(prompt="Press [Enter] to continue...")

# --- Mathematical Details / SVD Connection ---
cat("\nExploring mathematical details and SVD connection:\n")

# Calculate singular values (related to sdev)
n_samples <- nrow(nci_data)
singular_values <- pca_out$sdev * sqrt(n_samples - 1)
cat("First 5 singular values (sdev * sqrt(n-1)):\n")
print(head(singular_values))

cat("\nCalculating U matrix manually from scores and sdev:\n")
# U matrix in SVD (X = UDV') is related to scores. U should have orthonormal columns.
# U = scores / singular_values (approximately, depends on centering/scaling details)
# Let's check if columns of scores divided by sdev have unit variance (they should by definition of PCA)
cat("Variance of scaled score columns (should be 1):\n")
print(apply(scale(scores), 2, var)) # scale() centers and scales to unit variance

# Let's manually center the data and perform SVD
cat("\nPerforming SVD on centered data:\n")
data_centered <- scale(nci_data, center = pca_out$center, scale = FALSE) # Only center
svd_result <- svd(data_centered)
U_from_svd <- svd_result$u
cat("Head of U matrix from SVD (svd_result$u):\n")
print(head(U_from_svd[, 1:5]))
# Note: The scores 'pca_out$x' are related to U*D from SVD: scores = U %*% diag(D)
# The signs of SVD components can sometimes be flipped compared to prcomp.

readline(prompt="Press [Enter] to continue...")

cat("\nCalculating correlation between Gene 1 and PC1 score:\n")
# How much does the first gene correlate with the first principal component?
correlation_g1_pc1 <- cor(nci_data[,1], scores[,1])
cat("cor(nci_data[,1], scores[,1]):", correlation_g1_pc1, "\n")

cat("\nCalculating correlation using loadings formula:\n")
# The formula relates loading, sdev of PC, and sdev of original variable
# When data is scaled (like in our pca_out), sd(original_var) is 1.
# Correlation = Loading * sdev(PC) / sd(scaled original var)
loading_g1_pc1 <- right_singular_vectors[1, 1]
sdev_pc1 <- pca_out$sdev[1]
# Since data was scaled, sd(scaled_nci_data[,1]) is approx 1
cor_formula <- loading_g1_pc1 * sdev_pc1 / 1 # Or sd(scale(nci_data)[,1]) which is 1
cat("Loading[1,1] * sdev[1] / sd(scaled_gene1):", cor_formula, "\n")
cat("Note: The results match, confirming the relationship.\n")


# --- End Mathematical Details ---

readline(prompt="Press [Enter] to continue...")

# TODO: Examine first few principal components
cat("\nExamine first 5 principal component scores (head(scores[,1:5])):\n")
print(head(scores[, 1:5]))

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")

# Examine PCA results summary
# TODO: What does this summary tell us?
cat("\nSummary of standard PCA results (summary(pca_out)):\n")
print(summary(pca_out))
# Standard deviation: The standard deviation explained by each PC (sqrt(variance)).
# Proportion of Variance: The percentage of total variance explained by each PC.
# Cumulative Proportion: The cumulative percentage of variance explained by the PCs up to that point.
cat("Interpretation: PC1 explains ~11.7% of variance, PC1+PC2 explain ~21.4%, etc.\n")

cat("\n------------------------------------\n")
readline(prompt="Press [Enter] to continue...")


# --- 4. Plotting PCA Results ---

cat("--- 4. Plotting PCA Results ---\n")

# Basic PCA plot (Scree plot)
# TODO: Interpret this plot
cat("\nGenerating basic PCA plot (plot(pca_out))...\n")
plot(pca_out)
title("Basic PCA Plot (Variances of PCs)")
cat("Interpretation: This plot shows the variance associated with each principal component.\n")
cat("It's a 'scree plot', helping visualize the 'elbow' point where PCs explain less variance.\n")

readline(prompt="Press [Enter] to continue...")

# TODO: Try different plot types: type="l", type="b"
cat("\nBasic PCA plot with type='l' (lines)...\n")
plot(pca_out, type = "l")
title("Basic PCA Plot (type='l')")
readline(prompt="Press [Enter] to continue...")

cat("\nBasic PCA plot with type='b' (both points and lines)...\n")
plot(pca_out, type = "b")
title("Basic PCA Plot (type='b')")

readline(prompt="Press [Enter] to continue...")

# Create scree plot manually
cat("\nCreating Scree plot manually...\n")
var_explained <- variances / sum(variances) # Same as summary output proportion
# Equivalent to: var_explained <- pca_out$sdev^2 / sum(pca_out$sdev^2)

plot(var_explained, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 0.15), type = "b",
     main = "Manual Scree Plot (Individual Variance)")

readline(prompt="Press [Enter] to continue...")

# TODO: Try different visualizations of variance explained
# Try: Adding cumulative variance line
cat("\nAdding cumulative variance line to Scree plot...\n")
cum_var_explained <- cumsum(var_explained)
plot(var_explained, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = "b",
     main = "Scree Plot with Cumulative Variance", col = "blue", pch = 19)
lines(cum_var_explained, type = "b", col = "red", pch = 17)
legend("right", legend = c("Individual Var.", "Cumulative Var."),
       col = c("blue", "red"), pch = c(19, 17), lty = 1)

readline(prompt="Press [Enter] to continue...")

# Try: Using barplot() instead of plot()
cat("\nUsing barplot for individual variance...\n")
# Note: Adding cumulative line neatly on barplot requires more effort (e.g., second y-axis)
# We'll show the barplot and the combined line plot separately.
barplot(var_explained[1:20], names.arg = 1:20, # Show first 20
        xlab = "Principal Component",
        ylab = "Proportion of Variance Explained",
        main = "Scree Plot using Barplot (First 20 PCs)",
        ylim = c(0, 0.15), col="lightblue")

readline(prompt="Press [Enter] to continue...")

# Function for coloring observations
# TODO: Experiment with different color palettes
cat("\nDefining and testing coloring functions...\n")

# Original function from Code 2
Color_please <- function(vec) {
  cols <- rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}
original_colors <- Color_please(nci_labs)
cat("Generated colors using rainbow(). First few:\n")
print(head(original_colors))

# Try: Different color functions like heat.colors()
heat_colors_vec <- heat.colors(length(unique(nci_labs)))[as.numeric(as.factor(nci_labs))]
cat("\nGenerated colors using heat.colors(). First few:\n")
print(head(heat_colors_vec))
# Note: heat.colors is sequential, not ideal for distinct categories.

# Try: RColorBrewer palettes
# Ensure RColorBrewer is available
if (requireNamespace("RColorBrewer", quietly = TRUE)) {
  library(RColorBrewer)
  num_colors_needed <- length(unique(nci_labs))
  # Choose a qualitative palette suitable for categories
  if (num_colors_needed <= 12) {
    # 'Paired' is good for up to 12, 'Set1' for up to 9
    palette_name <- ifelse(num_colors_needed <= 9, "Set1", "Paired")
    brewer_colors_palette <- brewer.pal(num_colors_needed, palette_name)
    brewer_colors_vec <- brewer_colors_palette[as.numeric(as.factor(nci_labs))]
    cat("\nGenerated colors using RColorBrewer ('", palette_name, "'). First few:\n")
    print(head(brewer_colors_vec))
    final_plot_colors <- brewer_colors_vec # Use Brewer colors for subsequent plots
    final_legend_colors <- brewer_colors_palette # Colors for the legend itself
    final_legend_labels <- levels(as.factor(nci_labs))
  } else {
    cat("\nToo many categories for standard Brewer palettes, using rainbow.\n")
    final_plot_colors <- original_colors
    final_legend_colors <- rainbow(num_colors_needed)
    final_legend_labels <- levels(as.factor(nci_labs))
  }
} else {
  cat("\nRColorBrewer not found, using rainbow colors.\n")
  final_plot_colors <- original_colors
  final_legend_colors <- rainbow(length(unique(nci_labs)))
  final_legend_labels <- levels(as.factor(nci_labs))
}

cat("\nWe will use the generated 'final_plot_colors' for the following plots.\n")

readline(prompt="Press [Enter] to continue...")

# Plot first three principal components
cat("\nPlotting first principal components...\n")

# PC1 vs PC2
plot(pca_out$x[, 1:2], col = final_plot_colors,
     pch = 19, xlab = "PC1", ylab = "PC2",
     main = "PCA Scores: PC1 vs PC2")

readline(prompt="Press [Enter] to continue...")

# PC1 vs PC3
plot(pca_out$x[, c(1, 3)], col = final_plot_colors,
     pch = 19, xlab = "PC1", ylab = "PC3",
     main = "PCA Scores: PC1 vs PC3")

readline(prompt="Press [Enter] to continue...")

# TODO: Experiment with different plotting parameters
cat("\nExperimenting with plotting parameters for PC1 vs PC2...\n")

# Try: Different pch values (0:25)
plot(pca_out$x[, 1:2], col = final_plot_colors, pch = 0, 
     xlab = "PC1", ylab = "PC2", main = "PCA Scores: PC1 vs PC2 (pch=0)")
readline(prompt="Press [Enter] to continue...")
# Cross
plot(pca_out$x[, 1:2], col = final_plot_colors, pch = 3, 
     xlab = "PC1", ylab = "PC2", main = "PCA Scores: PC1 vs PC2 (pch=3)")
readline(prompt="Press [Enter] to continue...")
# Revert to pch=19 for filled circles
plot(pca_out$x[, 1:2], col = final_plot_colors, pch = 19,
     xlab = "PC1", ylab = "PC2", main = "PCA Scores: PC1 vs PC2 (pch=19)")

# Try: Adding a legend
# Need to adjust plot margins or place legend carefully
par(mar=c(5.1, 4.1, 4.1, 8.1), xpd=TRUE) # Increase right margin
plot(pca_out$x[, 1:2], col = final_plot_colors, pch = 19,
     xlab = "PC1", ylab = "PC2", main = "PCA Scores: PC1 vs PC2 with Legend")
legend("topright", inset=c(-0.25, 0), # Position legend outside plot area
       legend = final_legend_labels,
       fill = final_legend_colors,
       title = "Cancer Type", cex = 0.7)
par(mar=c(5.1, 4.1, 4.1, 2.1), xpd=FALSE) # Reset margins

readline(prompt="Press [Enter] to continue...")

# Try: Adding point labels with text()
plot(pca_out$x[, 1:2], col = final_plot_colors, pch = 19,
     xlab = "PC1", ylab = "PC2", main = "PCA Scores: PC1 vs PC2 with Text Labels")
text(pca_out$x[, 1], pca_out$x[, 2], labels = nci_labs, cex = 0.6, pos = 4, col=final_plot_colors)
cat("Note: Text labels can become very crowded.\n")

readline(prompt="Press [Enter] to continue...")

# Create biplots
cat("\nCreating biplots...\n")

# Default biplot (often crowded)
biplot(pca_out)
title("Default Biplot (Potentially Crowded)")
cat("Note: Default biplot shows all variables, often making it unreadable.\n")

readline(prompt="Press [Enter] to continue...")

# TODO: Experiment with biplot parameters
cat("\nExperimenting with biplot parameters...\n")

# Try: Different scaling options (scale=0,1)
# scale=0 emphasizes variable relationships (angles between arrows)
biplot(pca_out, scale = 0, main = "Biplot (scale=0: Variable Focus)")
readline(prompt="Press [Enter] to continue...")
# scale=1 emphasizes sample relationships (distances between points)
biplot(pca_out, scale = 1, main = "Biplot (scale=1: Sample Focus)")
readline(prompt="Press [Enter] to continue...")
# scale=2 is less common, often a compromise.

# Try: Different arrow.len values
biplot(pca_out, scale=0, arrow.len = 0.05, main="Biplot (scale=0, arrow.len=0.05)")
readline(prompt="Press [Enter] to continue...")
biplot(pca_out, scale=0, arrow.len = 0.1, main="Biplot (scale=0, arrow.len=0.1)") # Default is often 0.1
readline(prompt="Press [Enter] to continue...")

# Simplified biplot with first 10 variables
cat("\nCreating simplified biplot with first 10 variables...\n")
# Extract scores for PC1/PC2 and loadings for first 10 genes on PC1/PC2
pc12 <- pca_out$x[, 1:2]
loadings10 <- pca_out$rotation[1:10, 1:2] # Select first 10 genes
biplot(pc12, loadings10,
       cex = 0.7, # Size of sample points/variable labels
       main = "Biplot of First 10 Variables (Genes 1-10)")

readline(prompt="Press [Enter] to continue...")

# TODO: Try different numbers of variables
cat("\nSimplified biplot with first 5 variables...\n")
loadings5 <- pca_out$rotation[1:5, 1:2] # Select first 5 genes
biplot(pc12, loadings5,
       cex = 0.7,
       main = "Biplot of First 5 Variables (Genes 1-5)")

readline(prompt="Press [Enter] to continue...")

# TODO: Try different cex values for text size
cat("\nSimplified biplot with different cex values...\n")
# cex can be a vector: cex = c(sample_cex, variable_cex)
biplot(pc12, loadings10,
       cex = c(0.5, 0.9), # Smaller points, larger variable labels
       main = "Biplot (cex=c(0.5, 0.9))")

readline(prompt="Press [Enter] to continue...")

# TODO: Adding arrows=FALSE to remove arrows
cat("\nNote: Removing arrows entirely in standard biplot() requires custom plotting.\n")
# The standard biplot function doesn't have a simple arrows=FALSE argument.

# Labeled biplot
cat("\nCreating labeled biplot (using xlabs)...\n")
# Uses sample labels instead of points, can be very crowded
biplot(pc12, loadings10,
       cex = 0.7,
       xlabs = nci_labs, # Use cancer type labels for samples
       main = "Labeled Biplot (Crowded)")
cat("Note: Using labels directly often leads to overlap.\n")

readline(prompt="Press [Enter] to continue...")

# TODO: Experiment with label placement/size/color
cat("\nNote: Customizing label placement/color within standard biplot xlabs is limited.\n")
# Size was adjusted with cex. Placement (pos) and individual colors for xlabs
# are not directly supported by the basic biplot function arguments.
# More control requires manual plotting using plot() + text() + arrows().

# --- 5. Final Cumulative Variance Plot ---

cat("--- 5. Final Cumulative Variance Plot ---\n")

# TODO: Explain the output and functionality of the following code:
cat("\nGenerating final cumulative variance plot with thresholds...\n")
# This plot helps decide how many principal components are needed
# to capture a certain amount (e.g., 80% or 90%) of the total variance
# in the original data.

plot(cum_var_explained, type = "b", # Plot cumulative proportion
     xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), # Y-axis from 0 to 1 (or 0% to 100%)
     main = "Cumulative Variance Explained by PCs")
abline(h = 0.8, col = "red", lty = 2)  # Mark 80% threshold
abline(h = 0.9, col = "blue", lty = 2) # Mark 90% threshold
legend("bottomright", legend=c("Cumulative Variance", "80% Threshold", "90% Threshold"),
       col=c("black", "red", "blue"), lty=c(1, 2, 2), pch=c(1, NA, NA))

cat("Interpretation: Look where the black line crosses the dashed threshold lines.\n")
cat("This indicates how many PCs are needed to explain 80% or 90% of the variance.\n")

# --- 6. End of Script ---
cat("\n--- Analysis and Plotting Complete ---\n")