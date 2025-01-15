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
        pd.DataFrame: Merged DataFrame containing ABC expression and drug response data.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    # Construct file paths
    abc_expression_path = os.path.join(raw_data_dir, "Batch_corrected_Expression_Public_24Q4_subsetted_ABC.csv")
    drug_response_path = os.path.join(raw_data_dir, "PRISM_Repurposing_Public_24Q2_subsetted.csv")

    # Check if files exist
    if not os.path.exists(abc_expression_path):
        raise FileNotFoundError(f"ABC Expression file not found: {abc_expression_path}")
    if not os.path.exists(drug_response_path):
        raise FileNotFoundError(f"Drug Response file not found: {drug_response_path}")

    # Read the CSV files
    # Use the leftmost column as the CellLine_ID column
    abc_expression = pd.read_csv(abc_expression_path)
    drug_response = pd.read_csv(drug_response_path)

    # Ensure the first column is used as CellLine_ID for merging
    if abc_expression.columns[0] != "CellLine_ID":
        abc_expression.rename(columns={abc_expression.columns[0]: "CellLine_ID"}, inplace=True)
    if drug_response.columns[0] != "CellLine_ID":
        drug_response.rename(columns={drug_response.columns[0]: "CellLine_ID"}, inplace=True)

    # Merge the dataframes on 'CellLine_ID'
    merged_data = pd.merge(abc_expression, drug_response, on='CellLine_ID', how='inner')

    # Sort the merged DataFrame by 'CellLine_ID'
    merged_data = merged_data.sort_values(by='CellLine_ID')

    return merged_data

def calculate_correlations(merged_data_path):
    """
    Calculates the Pearson correlation coefficients between ABCB1 expression
    and IC50 values for each drug, and plots the top 10 and bottom 10 drugs individually.

    Parameters:
        merged_data_path (str): Path to the merged data CSV file.

    Returns:
        None
    """
    # Read the merged data
    merged_data = pd.read_csv(merged_data_path)

    # Ensure ABCB1 column is present
    if 'ABCB1' not in merged_data.columns:
        raise ValueError("ABCB1 gene data is missing in the merged dataset.")

    # Extract ABCB1 expression values
    abcb1_expression = merged_data['ABCB1']

    # Prepare to store results
    correlation_results = []

    # Loop through each drug column (skip non-drug columns and gene expression columns)
    for column in merged_data.columns:
        if column not in ['CellLine_ID', 'ABCB1', 'ABCG2', 'ABCC1', 'ABCC2', 'ABCC3', 'ABCC4']:
            print(f"Calculating correlation for drug: {column}")

            # Get drug response data
            drug_response = merged_data[column]

            # Drop rows where either ABCB1 or the drug response is NaN
            valid_indices = ~abcb1_expression.isna() & ~drug_response.isna()
            valid_abcb1 = abcb1_expression[valid_indices]
            valid_drug_response = drug_response[valid_indices]

            # Skip if there are fewer than 2 valid data points
            if len(valid_abcb1) < 2:
                print(f"Not enough valid data points for drug: {column}. Skipping.")
                continue

            # Calculate Pearson correlation
            try:
                corr, _ = pearsonr(valid_abcb1, valid_drug_response)
                # Append results
                correlation_results.append({'Drug': column, 'Pearson_Correlation': corr})
            except Exception as e:
                print(f"Error calculating correlation for drug: {column}. Skipping. Error: {e}")
                continue

    # Convert results to DataFrame
    correlation_df = pd.DataFrame(correlation_results)

    # Sort by correlation coefficient
    correlation_df = correlation_df.sort_values(by='Pearson_Correlation', ascending=False)

    # Print top 10 positively correlated drugs
    print("Top 10 positively correlated drugs:")
    print(correlation_df.head(10))

    # Print bottom 10 negatively correlated drugs
    print("Bottom 10 negatively correlated drugs:")
    print(correlation_df.tail(10))

    # Plot individual scatter plots for top 10 and bottom 10 drugs
    plot_individual_correlations(correlation_df, merged_data, merged_data_path)

    # Save results to the same directory as merged_data_path
    output_dir = os.path.dirname(merged_data_path)
    output_path = os.path.join(output_dir, "pearson_correlations.csv")
    correlation_df.to_csv(output_path, index=False)
    print(f"Correlation results saved to '{output_path}'.")

def plot_individual_correlations(correlation_df, merged_data, merged_data_path):
    """
    Creates individual scatter plots with a line of best fit and p-value for the top 10 and bottom 10 drugs,
    saving them in separate folders.

    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing drugs and their correlations.
        merged_data (pd.DataFrame): The merged data containing drug responses and ABCB1 expression.
        merged_data_path (str): Path to the merged data CSV file.
    """
    # Get the top 10 and bottom 10 drugs
    top_10 = correlation_df.head(10)
    bottom_10 = correlation_df.tail(10)

    # Output directory for plots
    output_dir = os.path.dirname(merged_data_path)

    # Directories for positive and negative correlations
    top_plots_dir = os.path.join(output_dir, "plots/top_10_positive_correlations")
    bottom_plots_dir = os.path.join(output_dir, "plots/bottom_10_negative_correlations")
    os.makedirs(top_plots_dir, exist_ok=True)
    os.makedirs(bottom_plots_dir, exist_ok=True)

    # Helper function to create plots
    def create_plot(drug, valid_drug_response, valid_abcb1, slope, intercept, corr, p_value, save_dir):
        """
        Creates a single scatter plot with line of best fit and p-value.
        """
        # Calculate line of best fit
        best_fit_line = slope * valid_drug_response + intercept

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(valid_drug_response, valid_abcb1, alpha=0.7, edgecolors='k', label='Data Points')
        plt.plot(valid_drug_response, best_fit_line, color='red', label=f'Best Fit Line (y={slope:.2f}x + {intercept:.2f})')
        plt.xlabel(f'{drug} IC50 Value')
        plt.ylabel('ABCB1 Expression')
        plt.title(f'Correlation: r={corr:.2f}, p={p_value:.2e}')
        plt.grid(alpha=0.5)
        plt.legend()

        # Save the plot
        plot_path = os.path.join(save_dir, f'{drug}_scatter.png')
        plt.savefig(plot_path, format='png', dpi=300)
        print(f"Scatter plot saved to '{plot_path}'.")

        # Close the plot
        plt.close()

    # Create individual scatter plots for top 10 positive correlations
    for _, row in top_10.iterrows():
        drug = row['Drug']

        # Clean the drug name by removing everything after "(BRD"
        clean_drug_name = re.sub(r"\s*\(BRD:.*\)", "", drug)
        print(f"Plotting scatter plot for drug: {clean_drug_name} (Top 10)")

        # Extract drug response and ABCB1 data
        drug_response = merged_data[drug]
        abcb1_expression = merged_data['ABCB1']

        # Drop NaN values
        valid_indices = ~drug_response.isna() & ~abcb1_expression.isna()
        valid_drug_response = drug_response[valid_indices]
        valid_abcb1 = abcb1_expression[valid_indices]

        # Calculate line of best fit
        slope, intercept = np.polyfit(valid_drug_response, valid_abcb1, 1)

        # Calculate Pearson correlation and p-value
        corr, p_value = pearsonr(valid_drug_response, valid_abcb1)

        # Create plot
        create_plot(clean_drug_name, valid_drug_response, valid_abcb1, slope, intercept, corr, p_value, top_plots_dir)

    # Create individual scatter plots for bottom 10 negative correlations
    for _, row in bottom_10.iterrows():
        drug = row['Drug']

        # Clean the drug name by removing everything after "(BRD"
        clean_drug_name = re.sub(r"\s*\(BRD:.*\)", "", drug)
        print(f"Plotting scatter plot for drug: {clean_drug_name} (Bottom 10)")

        # Extract drug response and ABCB1 data
        drug_response = merged_data[drug]
        abcb1_expression = merged_data['ABCB1']

        # Drop NaN values
        valid_indices = ~drug_response.isna() & ~abcb1_expression.isna()
        valid_drug_response = drug_response[valid_indices]
        valid_abcb1 = abcb1_expression[valid_indices]

        # Calculate line of best fit
        slope, intercept = np.polyfit(valid_drug_response, valid_abcb1, 1)

        # Calculate Pearson correlation and p-value
        corr, p_value = pearsonr(valid_drug_response, valid_abcb1)

        # Create plot
        create_plot(clean_drug_name, valid_drug_response, valid_abcb1, slope, intercept, corr, p_value, bottom_plots_dir)

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
    calculate_correlations(merged_data_path)

except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)
except ValueError as e:
    print(f"Value error: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)