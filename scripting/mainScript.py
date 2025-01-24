import os
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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
    # Paths for raw data
    abc_expression_path = os.path.join(raw_data_dir, "Batch_corrected_Expression_Public_24Q4_subsetted_ABC.csv")
    drug_response_path = os.path.join(raw_data_dir, "PRISM_Repurposing_Public_24Q2_subsetted.csv")

    # Load files and validate
    for file_path, desc in [(abc_expression_path, "ABC Expression"), (drug_response_path, "Drug Response")]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{desc} file not found: {file_path}")

    abc_expression = pd.read_csv(abc_expression_path)
    drug_response = pd.read_csv(drug_response_path)

    # Ensure CellLine_ID is properly set
    for df in [abc_expression, drug_response]:
        if df.columns[0] != "CellLine_ID":
            df.rename(columns={df.columns[0]: "CellLine_ID"}, inplace=True)

    # Merge and sort
    return pd.merge(abc_expression, drug_response, on='CellLine_ID', how='inner').sort_values(by='CellLine_ID')

def calculate_correlations(merged_data_path, drug_data_path, gene_names, output_dir, folder_creator):
    """
    Calculates Pearson correlation coefficients between gene expressions and IC50 values for each drug,
    saves the results and merges with drug properties.

    Parameters:
        merged_data_path (str): Path to the merged data CSV file.
        drug_data_path (str): Path to the drug properties data CSV file.
        gene_names (list): List of gene names to calculate correlations for.
        output_dir (str): Path to the output directory to save results.
        folder_creator (OrderedFolderCreator): Instance of OrderedFolderCreator for creating folders.
    """
    # Load data
    parent_dir = os.path.abspath(os.path.join(output_dir, os.pardir))
    print(parent_dir)
    merged_data = pd.read_csv(merged_data_path)
    drug_properties = pd.read_csv(drug_data_path)

    # Validate genes
    missing_genes = [gene for gene in gene_names if gene not in merged_data.columns]
    if missing_genes:
        raise ValueError(f"Missing genes in dataset: {', '.join(missing_genes)}")

    # Calculate correlations
    drug_columns = merged_data.columns[7:]
    all_gene_results = [
        {'Gene': gene, 'Drug': drug, 'Pearson_Correlation': np.corrcoef(data[gene], data[drug])[0, 1]}
        for gene in gene_names
        for drug in drug_columns
        if (data := merged_data[[gene, drug]].dropna()).shape[0] >= 2
    ]

    merged_properties_path = folder_creator.create_folder(parent_dir, "properties_merged")

    for gene in gene_names:
        gene_df = pd.DataFrame([res for res in all_gene_results if res['Gene'] == gene])
        if not gene_df.empty:
            gene_df['id'] = gene_df['Drug'].str.extract(r"BRD:(BRD-.*?)-")
            gene_df.sort_values(by='Pearson_Correlation', ascending=False, inplace=True)
            gene_df.to_csv(os.path.join(output_dir, f"{gene}_pearson_correlations.csv"), index=False)

            merged_properties = pd.merge(gene_df, drug_properties, on='id', how='inner').sort_values(by='Pearson_Correlation')
            merged_properties.to_csv(os.path.join(merged_properties_path, f"{gene}_properties_merged.csv"), index=False)
            print(f"Saved results for {gene}")

    return merged_properties_path