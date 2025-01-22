import os
from tokenize import group
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

class PearsonCorrelationAnalyzer:
    def __init__(self, processed_data_dir, gene_name):
        self.processed_data_dir = processed_data_dir
        self.gene_name = gene_name

    def get_file_path(self):
        return os.path.join(self.processed_data_dir, f"{self.gene_name}_properties_merged.csv")

    def save_top_bottom_drugs(self, output_directory):
        file_path = self.get_file_path()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = pd.read_csv(file_path)
        if "Drug" not in data.columns or "Pearson_Correlation" not in data.columns:
            raise ValueError("Required columns ('Drug', 'Pearson_Correlation') are missing in the file.")

        # Sort the data by Pearson_Correlation
        sorted_data = data.sort_values(by="Pearson_Correlation", ascending=False)

        # Extract top 10 and bottom 10
        top_10 = sorted_data.head(10)
        bottom_10 = sorted_data.tail(10)

        # Combine into one DataFrame with separate columns for Top and Bottom
        combined_results = pd.DataFrame({
            "Top_10_Drug": top_10["Drug"].reset_index(drop=True),
            "Top_10_Correlation": top_10["Pearson_Correlation"].reset_index(drop=True),
            "Bottom_10_Drug": bottom_10["Drug"].reset_index(drop=True),
            "Bottom_10_Correlation": bottom_10["Pearson_Correlation"].reset_index(drop=True),
        })

        # Save to a single CSV file
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, f"{self.gene_name}_top_bottom_drugs.csv")
        combined_results.to_csv(output_file, index=False)
        print(f"Top and bottom 10 drugs saved to {output_file}")

        # Analyze properties for significance
        self.analyze_properties(top_10, bottom_10, output_directory)

    def analyze_properties(self, top_10, bottom_10, output_directory):
        """
        Analyzes the properties of the top and bottom drugs separately and compares them.
        """
        results = []
        for category, group in [("Top 10", top_10), ("Bottom 10", bottom_10)]:
            properties = group.iloc[:, 15:]

            for column in properties.columns:
                values = properties[column].dropna()

                # Use Coefficient of Variation (CV) to assess variability
                if values.dtype in [np.float64, np.int64]: # Only numeric columns
                    mean = values.mean()
                    std_dev = values.std()

                    # Calculate CV and determine significance
                    if mean != 0:  # Avoid division by zero
                        cv = std_dev / mean
                        if abs(cv) <= 0.2 and abs(cv) != 0:  # Threshold for significant values, abs because CV can be negative
                            results.append({
                                "Property": column,
                                "Category": category,
                                "Mean": mean,
                                "Standard Deviation": std_dev,
                                "Coefficient of Variation": cv,
                            })

        # Save analysis results for both categories in a single file
        analysis_output_file = os.path.join(output_directory, f"{self.gene_name}_significant_properties.csv")
        pd.DataFrame(results).to_csv(analysis_output_file, index=False)
        print(f"Significant property analysis saved to {analysis_output_file}")

        self.calculate_correlation_stats(output_directory)

    def calculate_correlation_stats(self, output_directory):
        """
        Calculates the Pearson correlation for each property against Pearson_Correlation,
        extracts the top and bottom 10 correlations, and saves them to a CSV file.
        """
        file_path = self.get_file_path()
        data = pd.read_csv(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if "Pearson_Correlation" not in data.columns:
            raise ValueError("'Pearson_Correlation' column is missing in the file.")

        results = []
        drug_properties = data.iloc[:, 15:]  # Assuming property columns start at index 15

        for property in drug_properties.columns:
            property_values = data[property].dropna()
            correlation_values = data["Pearson_Correlation"]

            # Ensure there are enough data points
            if len(property_values) > 2 and property_values.dtype in [np.float64, np.int64]:
                try:
                    pearson_corr, _ = pearsonr(correlation_values, property_values)
                    if not np.isnan(pearson_corr):  # Exclude NaN correlations
                        results.append({
                            "Property": property,
                            "Pearson_Correlation": pearson_corr
                        })
                except Exception as e:
                    print(f"Error calculating correlation for {property}: {e}")

        # Ensure results are valid before proceeding
        if not results:
            print("No valid correlations found.")
            return

        # Create a DataFrame of results and extract top and bottom 10 correlations
        results_df = pd.DataFrame(results).dropna().sort_values(by="Pearson_Correlation", ascending=False)
        top_10 = results_df.head(10)
        bottom_10 = results_df.tail(10)

        # Save top and bottom correlations to CSV
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, f"{self.gene_name}_top_bottom_correlations.csv")
        pd.concat([top_10, bottom_10]).to_csv(output_file, index=False)
        print(f"Top and bottom 10 correlations saved to {output_file}")
