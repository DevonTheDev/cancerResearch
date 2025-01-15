import os
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import re
import numpy as np

def merge_abc_and_drug_data(raw_data_dir):
    """
    Merges ABC expression data and drug response data into a single DataFrame.

    Parameters:
        raw_data_dir (str): Path to the directory containing the raw data files.

    Returns:
        pd.DataFrame: Merged DataFrame containing ABC expression and drug response data for cell lines.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    # Construct file paths
    abc_expression_path = os.path.join(raw_data_dir, "Batch_corrected_Expression_Public_24Q4_subsetted_ABC.csv")
    drug_response_path = os.path.join(raw_data_dir, "PRISM_Repurposing_Public_24Q2_subsetted.csv")

    # Check if files exist
    for file_path, file_desc in [(abc_expression_path, "ABC Expression"), (drug_response_path, "Drug Response")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_desc} file not found: {file_path}")

    # Read the CSV files
    abc_expression = pd.read_csv(abc_expression_path)
    drug_response = pd.read_csv(drug_response_path)

    # Ensure the first column is used as CellLine_ID for merging. Add a name since it has no header in the raw data and makes my life easier later down the line
    for df in [abc_expression, drug_response]:
        if df.columns[0] != "CellLine_ID":
            df.rename(columns={df.columns[0]: "CellLine_ID"}, inplace=True)

    # Merge the data on the CellLine_ID column and sort by CellLine_ID
    return pd.merge(abc_expression, drug_response, on='CellLine_ID', how='inner').sort_values(by='CellLine_ID')

def calculate_correlations(merged_data_path, gene_names):
    """
    Calculates the Pearson correlation coefficients between gene expressions
    and IC50 values for each drug, and plots the top 10 and bottom 10 drugs individually for analysis.

    Parameters:
        merged_data_path (str): Path to the merged data CSV file.
        gene_names (list): List of gene names to calculate correlations for.

    Returns:
        None
    """
    # Read the merged data
    merged_data = pd.read_csv(merged_data_path)

    # Validate gene presence in the dataset
    missing_genes = set(gene_names) - set(merged_data.columns)
    if missing_genes:
        raise ValueError(f"The following genes are missing in the dataset: {', '.join(missing_genes)}")

    # Calculate correlations for each gene and drug combination
    drug_columns = merged_data.columns[7:]  # Exclude the first 7 columns (CellLine_ID, ABC genes)
    all_gene_results = [
        {
            'Gene': gene,
            'Drug': drug,
            'Pearson_Correlation': np.corrcoef(valid_data[gene], valid_data[drug])[0, 1]
        }
        for gene in gene_names
        for drug in drug_columns
        if (valid_data := merged_data[[gene, drug]].dropna()).shape[0] >= 2
    ]

    # Save all results to CSV
    output_dir = os.path.dirname(merged_data_path)
    output_path = os.path.join(output_dir, "all_gene_pearson_correlations.csv")
    pd.DataFrame(all_gene_results).to_csv(output_path, index=False)
    print(f"Correlation results saved to: {output_path}")

    # Summarize and plot top/bottom correlations for each gene
    def summarize_and_plot(gene, results, data):
        gene_corr = results[results['Gene'] == gene]
        for label, ascending in [("Top 10 positively", False), ("Bottom 10 negatively", True)]:
            subset = gene_corr.sort_values(by='Pearson_Correlation', ascending=ascending).head(10)
            print(f"\n{label} correlated drugs for {gene}:")
            print(subset)
            plot_individual_correlations(subset, data, merged_data_path)

    all_correlation_df = pd.DataFrame(all_gene_results)
    for gene in gene_names:
        summarize_and_plot(gene, all_correlation_df, merged_data)

def plot_individual_correlations(correlation_df, merged_data, merged_data_path):
    """
    Creates individual scatter plots with a line of best fit and p-value for the top 10 and bottom 10 drugs
    for each gene, saving them in separate folders.

    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing drugs, genes, and their correlations.
        merged_data (pd.DataFrame): The merged data containing drug responses and gene expressions.
        merged_data_path (str): Path to the merged data CSV file.
    """

    # Validate input
    if 'Gene' not in correlation_df.columns:
        raise ValueError("The 'Gene' column is missing from correlation_df. Ensure it is included in the DataFrame.")

    # Output directory for plots
    plots_dir = os.path.join(os.path.dirname(merged_data_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    def generate_scatter_plots(data_subset, save_dir):
        """Generate scatter plots for a given subset of data."""
        os.makedirs(save_dir, exist_ok=True)
        for _, row in data_subset.iterrows():
            drug, gene = row['Drug'], row['Gene']

            # Drop NaN values
            valid_data = merged_data[[drug, gene]].dropna()
            if valid_data.empty:
                continue
            valid_drug_response, valid_gene_expression = valid_data[drug], valid_data[gene]

            # Calculate line of best fit
            slope, intercept = np.polyfit(valid_drug_response, valid_gene_expression, 1)
            # Calculate Pearson correlation and p-value
            corr, p_value = pearsonr(valid_drug_response, valid_gene_expression)

            # Create scatter plot
            drug_cleaned = re.sub(r"\(BRD.*$", "", drug).strip() # Remove the BRD number from the drug name for better readability and fix weird bug with file saving
            plt.figure(figsize=(8, 6))
            plt.scatter(valid_drug_response, valid_gene_expression, alpha=0.7, edgecolors='k', s=50)
            plt.plot(valid_drug_response, slope * valid_drug_response + intercept, color='red', linewidth=2)
            plt.xlabel(f'{drug_cleaned} IC50 Value', fontsize=14, fontweight='bold')
            plt.ylabel(f'{gene} Expression', fontsize=14, fontweight='bold')
            plt.title(f'{gene} vs {drug_cleaned}', fontsize=16, fontweight='bold')
            plt.text(0.05, 0.95, f'Pearson r = {corr:.2f}\np-value = {p_value:.2e}', transform=plt.gca().transAxes,
                     fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{gene}_{drug_cleaned}_scatter.png'), format='png', dpi=300)
            plt.close()

    # Generate plots for each gene
    for gene in correlation_df['Gene'].unique():
        print(f"Plotting for gene: {gene}")
        gene_dir = os.path.join(plots_dir, gene)
        gene_corr = correlation_df[correlation_df['Gene'] == gene]

        # Plot top 10 positively correlated drugs
        generate_scatter_plots(
            gene_corr.sort_values(by='Pearson_Correlation', ascending=False).head(10),
            os.path.join(gene_dir, "top_10_positive_correlations")
        )

        # Plot bottom 10 negatively correlated drugs
        generate_scatter_plots(
            gene_corr.sort_values(by='Pearson_Correlation', ascending=True).head(10),
            os.path.join(gene_dir, "bottom_10_negative_correlations")
        )

# Main script
try:
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, "Raw Data")
    merged_data_path = os.path.join(raw_data_dir, "merged_data.csv")

    # Validate Raw Data directory
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"'Raw Data' directory not found: {raw_data_dir}")

    # Merge data if necessary
    if not os.path.exists(merged_data_path):
        print(f"Merged data file not found. Creating at {merged_data_path}...")
        merged_data = merge_abc_and_drug_data(raw_data_dir)
        merged_data.to_csv(merged_data_path, index=False)
        print(f"Merged data saved to {merged_data_path}")

    # Perform correlation calculations
    gene_names = ['ABCB1', 'ABCG2', 'ABCC1', 'ABCC2', 'ABCC3', 'ABCC4']
    calculate_correlations(merged_data_path, gene_names)

except (FileNotFoundError, ValueError) as e:
    print(f"Error: {e}")
    exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    exit(1)