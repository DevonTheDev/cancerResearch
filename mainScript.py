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

    # Ensure the first column is used as CellLine_ID for merging. Add a name since it has no header in the raw data.
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

    # Ensure all genes are present
    missing_genes = [gene for gene in gene_names if gene not in merged_data.columns]
    if missing_genes:
        raise ValueError(f"The following gene data is missing in the merged dataset: {missing_genes}")

    # Prepare to store results
    all_gene_results = []

    # Loop through each gene
    for gene in gene_names:
        print(f"Processing gene: {gene}")

        # Process each drug column (skip the first 7 non-drug columns)
        for drug in merged_data.columns[7:]:
            print(f"Calculating correlation between {gene} and drug: {drug}")

            # Drop rows with NaN values and ensure sufficient valid data
            valid_data = merged_data[[gene, drug]].dropna()
            if len(valid_data) < 2:
                print(f"Not enough valid data points for drug: {drug}. Skipping.")
                continue

            # Calculate Pearson correlation and store results
            try:
                corr, _ = pearsonr(valid_data[gene], valid_data[drug])
                all_gene_results.append({'Gene': gene, 'Drug': drug, 'Pearson_Correlation': corr})
            except Exception as e:
                print(f"Error calculating correlation for drug: {drug}. Skipping. Error: {e}")

    # Convert all results to DataFrame
    all_correlation_df = pd.DataFrame(all_gene_results)

    # Save results to CSV
    output_dir = os.path.dirname(merged_data_path)
    output_path = os.path.join(output_dir, "all_gene_pearson_correlations.csv")
    all_correlation_df.to_csv(output_path, index=False)
    print(f"All gene correlation results saved to '{output_path}'.")

    # Print summary and plot correlations for each gene
    for gene in gene_names:
        # Filter correlations for the current gene
        gene_corr = all_correlation_df[all_correlation_df['Gene'] == gene]
        
        for label, ascending in [("Top 10 positively", False), ("Bottom 10 negatively", True)]:
            # Get the top/bottom 10 correlated drugs
            top_bottom_10 = gene_corr.sort_values(by='Pearson_Correlation', ascending=ascending).head(10)
            print(f"\n{label} correlated drugs for {gene}:")
            print(top_bottom_10)

            # Plot the correlations
            plot_individual_correlations(top_bottom_10, merged_data, merged_data_path)

def plot_individual_correlations(correlation_df, merged_data, merged_data_path):
    """
    Creates individual scatter plots with a line of best fit and p-value for the top 10 and bottom 10 drugs
    for each gene, saving them in separate folders.

    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing drugs, genes, and their correlations.
        merged_data (pd.DataFrame): The merged data containing drug responses and gene expressions.
        merged_data_path (str): Path to the merged data CSV file.
    """

     # Check if the 'Gene' column is present in the correlation_df
    if 'Gene' not in correlation_df.columns:
        raise ValueError("The 'Gene' column is missing from correlation_df. Ensure it is included in the DataFrame.")
    

    # Output directory for plots
    output_dir = os.path.dirname(merged_data_path)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    import re  # Importing the regex module

    def create_plot(drug, gene, valid_drug_response, valid_gene_expression, slope, intercept, corr, p_value, save_dir):

        """
        Creates a single scatter plot with line of best fit and p-value.
        """

        import matplotlib.ticker as ticker

        # Clean the drug name by removing anything after "(BRD"
        drug_cleaned = re.sub(r"\(BRD.*$", "", drug).strip()

        # Prepare line of best fit
        best_fit_line = slope * valid_drug_response + intercept

        # Set up the plot
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_drug_response, valid_gene_expression, alpha=0.7, edgecolors='k', s=50, label='Data Points')
        plt.plot(valid_drug_response, best_fit_line, color='red', linewidth=2, 
                label=f'Best Fit Line (y={slope:.2f}x + {intercept:.2f})')

        # Format axes
        plt.xlabel(f'{drug_cleaned} IC50 Value', fontsize=14, fontweight='bold')
        plt.ylabel(f'{gene} Expression', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format ticks
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))

        # Add title and annotations
        plt.title(f'{gene} vs {drug_cleaned}', fontsize=16, fontweight='bold')
        plt.text(0.05, 0.95, f'Pearson r = {corr:.2f}\np-value = {p_value:.2e}',
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Add grid with adjusted transparency
        plt.grid(alpha=0.3, linestyle='--', linewidth=0.7)

        # Add legend
        plt.legend(fontsize=12, loc='best')

        # Save the plot
        os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
        plot_path = os.path.join(save_dir, f'{gene}_{drug_cleaned}_scatter.png')
        plt.tight_layout()  # Adjust layout for better fit
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory


    print("Plotting scatter plots for each gene...")

    # Iterate through each gene
    print(f"Correlation df: {correlation_df['Gene']}")
    for gene in correlation_df['Gene'].unique():
        print(f"Plotting for gene: {gene}")
        gene_dir = os.path.join(plots_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)

        # Filter correlations for this gene
        gene_corr = correlation_df[correlation_df['Gene'] == gene]

        # Get the top 10 and bottom 10 drugs
        top_10 = gene_corr.sort_values(by='Pearson_Correlation', ascending=False).head(10)
        bottom_10 = gene_corr.sort_values(by='Pearson_Correlation', ascending=True).head(10)

        # Create individual scatter plots for top 10 positive correlations
        top_dir = os.path.join(gene_dir, "top_10_positive_correlations")
        os.makedirs(top_dir, exist_ok=True)
        for _, row in top_10.iterrows():
            drug = row['Drug']
            drug_response = merged_data[drug]
            gene_expression = merged_data[gene]

            # Drop NaN values
            valid_indices = ~drug_response.isna() & ~gene_expression.isna()
            valid_drug_response = drug_response[valid_indices]
            valid_gene_expression = gene_expression[valid_indices]

            # Calculate line of best fit
            slope, intercept = np.polyfit(valid_drug_response, valid_gene_expression, 1)

            # Calculate Pearson correlation and p-value
            corr, p_value = pearsonr(valid_drug_response, valid_gene_expression)

            # Create plot
            create_plot(drug, gene, valid_drug_response, valid_gene_expression, slope, intercept, corr, p_value, top_dir)

        # Create individual scatter plots for bottom 10 negative correlations
        bottom_dir = os.path.join(gene_dir, "bottom_10_negative_correlations")
        os.makedirs(bottom_dir, exist_ok=True)
        for _, row in bottom_10.iterrows():
            drug = row['Drug']
            drug_response = merged_data[drug]
            gene_expression = merged_data[gene]

            # Drop NaN values
            valid_indices = ~drug_response.isna() & ~gene_expression.isna()
            valid_drug_response = drug_response[valid_indices]
            valid_gene_expression = gene_expression[valid_indices]

            # Calculate line of best fit
            slope, intercept = np.polyfit(valid_drug_response, valid_gene_expression, 1)

            # Calculate Pearson correlation and p-value
            corr, p_value = pearsonr(valid_drug_response, valid_gene_expression)

            # Create plot
            create_plot(drug, gene, valid_drug_response, valid_gene_expression, slope, intercept, corr, p_value, bottom_dir)

    print("Scatter plots created for all genes.")

# Main script
try:
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to Raw Data directory
    raw_data_dir = os.path.join(script_dir, "Raw Data")
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"The directory 'Raw Data' does not exist: {raw_data_dir}")

    # Path to the merged data file
    merged_data_path = os.path.join(raw_data_dir, "merged_data.csv")

    # Merge data if the merged file does not exist
    if not os.path.exists(merged_data_path):
        print(f"Merged data file not found at {merged_data_path}. Creating it now...")
        merged_data = merge_abc_and_drug_data(raw_data_dir)
        os.makedirs(os.path.dirname(merged_data_path), exist_ok=True)
        merged_data.to_csv(merged_data_path, index=False)
        print(f"Merged data saved to {merged_data_path}")

    # Perform correlation calculations
    gene_names = ['ABCB1', 'ABCG2', 'ABCC1', 'ABCC2', 'ABCC3', 'ABCC4'] # List of gene names to calculate correlations for
    calculate_correlations(merged_data_path, gene_names)

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)
except ValueError as e:
    print(f"Value error: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)