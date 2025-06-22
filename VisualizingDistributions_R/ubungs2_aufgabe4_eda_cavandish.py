import sys
print(sys.executable)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os  # Import the os module for directory operations


# --- Configuration ---
PLOT_DIR = "eda_plot_python_cavandish"  # Define the directory name to save plots

os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved in directory: '{PLOT_DIR}'")


# Data: Cavendish's Earth Density Measurements (g/cm3)
density_data = np.array([4.88, 5.07, 5.10, 5.26, 5.27, 5.29, 5.29, 5.30, 5.34, 5.34,
                        5.36, 5.39, 5.42, 5.44, 5.46, 5.47, 5.50, 5.53, 5.55, 5.57,
                        5.58, 5.61, 5.62, 5.63, 5.65, 5.68, 5.75, 5.79, 5.85])

# Create a pandas Series to make data analysis easier
data_series = pd.Series(density_data)

# ===== Part (a): Statistical Calculations =====

print("\n--- Statistical Calculations ---")
# Mean
mean = data_series.mean()

# Median
median = data_series.median()

# Quantiles
quantiles = data_series.quantile([0.25, 0.5, 0.75])

# Standard Deviation
std_dev = data_series.std()

# Upper and Lower bounds (mean ± 3 * std)
lower_bound = mean - 3 * std_dev
upper_bound = mean + 3 * std_dev

# Print the results
print(f"Mean: {mean:.4f}") # Format output for better readability
print(f"Median: {median:.4f}")
print("Quantiles (25%, 50%, 75%):")
print(quantiles.to_string(float_format="%.4f")) # Format quantile output
print(f"Standard Deviation: {std_dev:.4f}")
print(f"Lower Bound (mean - 3 * std): {lower_bound:.4f}")
print(f"Upper Bound (mean + 3 * std): {upper_bound:.4f}")
print("------------------------------\n")


# ===== Part (b): Visualizations =====

print("--- Generating and Saving Plots ---")

# --- Boxplot ---
print("1. Generating Boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_series, color="skyblue")
plt.title("Boxplot of Earth Density Measurements")
plt.ylabel("Density (g/cm3)")
# Construct filename with path
filename_boxplot = os.path.join(PLOT_DIR, "01_boxplot_density.png")
# Save the figure *before* showing it
plt.savefig(filename_boxplot, dpi=300, bbox_inches='tight') # Use dpi and bbox_inches for quality
print(f"   Saved: {filename_boxplot}")
plt.show() # Display the plot
plt.close() # Close the figure to free memory

# --- Histogram with Density ---
print("2. Generating Histogram with Density Curve...")
plt.figure(figsize=(10, 6))
sns.histplot(data_series, kde=True, color="lightgreen", bins=10)
plt.title("Histogram with Density Curve")
plt.xlabel("Density (g/cm3)")
plt.ylabel("Frequency")
filename_hist_density = os.path.join(PLOT_DIR, "02_histogram_density_curve.png")
plt.savefig(filename_hist_density, dpi=300, bbox_inches='tight')
print(f"   Saved: {filename_hist_density}")
plt.show()
plt.close()

# --- Histogram with Frequency ---
print("3. Generating Histogram with Frequency...")
plt.figure(figsize=(10, 6))
# Use plt.hist directly for frequency y-axis
plt.hist(data_series, bins=10, color="lightcoral", edgecolor='black')
plt.title("Histogram with Frequency")
plt.xlabel("Density (g/cm3)")
plt.ylabel("Frequency")
filename_hist_freq = os.path.join(PLOT_DIR, "03_histogram_frequency.png")
plt.savefig(filename_hist_freq, dpi=300, bbox_inches='tight')
print(f"   Saved: {filename_hist_freq}")
plt.show()
plt.close()

# --- Empirical Cumulative Distribution Function (ECDF) ---
print("4. Generating ECDF Plot...")
plt.figure(figsize=(10, 6))
sns.ecdfplot(data_series, color="purple")
plt.title("Empirical CDF of Earth Density Measurements")
plt.xlabel("Density (g/cm3)")
plt.ylabel("ECDF")
filename_ecdf = os.path.join(PLOT_DIR, "04_ecdf_plot.png")
plt.savefig(filename_ecdf, dpi=300, bbox_inches='tight')
print(f"   Saved: {filename_ecdf}")
plt.show()
plt.close()

# --- QQ Plot ---
print("5. Generating QQ Plot...")
# QQ Plot (to check if data follows a normal distribution)
plt.figure(figsize=(10, 6))
# stats.probplot generates the plot on the current axes
stats.probplot(data_series, dist="norm", plot=plt)
plt.title("QQ Plot (Normal Distribution Check)") # Add more descriptive title
# No separate xlabel/ylabel needed as probplot adds them
filename_qq = os.path.join(PLOT_DIR, "05_qq_plot_normal.png")
plt.savefig(filename_qq, dpi=300, bbox_inches='tight')
print(f"   Saved: {filename_qq}")
plt.show()
plt.close()

print("------------------------------")
print("All plots saved successfully.")
