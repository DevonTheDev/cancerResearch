import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, anderson

class PearsonCorrelationAnalyzer:
    def __init__(self, processed_data_dir, gene_name):
        self.processed_data_dir = processed_data_dir
        self.gene_name = gene_name

    def get_file_path(self):
        """Generates the file path for the gene properties file."""
        return os.path.join(self.processed_data_dir, f"{self.gene_name}_properties_merged.csv")

    def save_top_bottom_drugs(self, output_directory):
        """Saves the top 10 and bottom 10 drugs based on Pearson Correlation to a CSV file."""
        file_path = self.get_file_path()

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        data = pd.read_csv(file_path)
        required_columns = {"Drug", "Pearson_Correlation"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Required columns {required_columns} are missing in the file.")

        # Sort the data by Pearson Correlation
        sorted_data = data.sort_values(by="Pearson_Correlation", ascending=False)

        # Extract top 10 and bottom 10 entries
        top_10 = sorted_data.head(10)
        bottom_10 = sorted_data.tail(10)

        # Combine the results into a single DataFrame
        combined_results = pd.DataFrame({
            "Top_10_Drug": top_10["Drug"].reset_index(drop=True),
            "Top_10_Correlation": top_10["Pearson_Correlation"].reset_index(drop=True),
            "Bottom_10_Drug": bottom_10["Drug"].reset_index(drop=True),
            "Bottom_10_Correlation": bottom_10["Pearson_Correlation"].reset_index(drop=True),
        })

        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, f"{self.gene_name}_top_bottom_drugs.csv")
        combined_results.to_csv(output_file, index=False)
        print(f"Top and bottom 10 drugs saved to {output_file}")

        # Analyze properties for significant differences
        self.analyze_properties(data, top_10, bottom_10, output_directory)

    def analyze_properties(self, all_drugs, top_10, bottom_10, output_directory):
        """Analyzes the properties of top and bottom drugs and compares them to the rest, as well as to each other."""
        top_10_results = []
        bottom_10_results = []
        comparison_results = []
        
        def biserial_correlation(test_statistic, group1, group2):
            """Calculates biserial correlation effect size for two groups."""
            return (1 - ((2 * test_statistic) / (group1.size + group2.size)))

        # Combine all comparisons (Top 10 vs rest, Bottom 10 vs rest, and Top 10 vs Bottom 10)
        for category, group, comparison_group in [
            ("Top 10", top_10, all_drugs[~all_drugs["Drug"].isin(top_10["Drug"])]),
            ("Bottom 10", bottom_10, all_drugs[~all_drugs["Drug"].isin(bottom_10["Drug"])]),
            ("Top 10 vs Bottom 10", top_10, bottom_10)
        ]:
            properties = group.iloc[:, 15:].dropna(axis=1, how="all")  # Drop columns with all NaN values

            for column in properties.columns:
                group_values = group[column].dropna()
                comparison_values = comparison_group[column].dropna()

                # Remove non-finite values (NaN, inf, -inf)
                group_values = group_values[np.isfinite(group_values)]
                comparison_values = comparison_values[np.isfinite(comparison_values)]

                # Skip properties with insufficient variation or no valid data left
                if (group_values.nunique() <= 1) or (comparison_values.nunique() <= 1) or group_values.empty or comparison_values.empty:
                    continue

                if np.issubdtype(group_values.dtype, np.number):
                    # Clip extreme values to prevent numeric errors
                    group_values = np.clip(group_values, -1e10, 1e10)
                    comparison_values = np.clip(comparison_values, -1e10, 1e10)

                    try:
                        group_normality = anderson(group_values).statistic < 1.0
                        comparison_normality = anderson(comparison_values).statistic < 1.0
                    except Exception as e:
                        print(f"Error in normality test for column '{column}': {e}")
                        group_normality, comparison_normality = False, False

                    try:
                        if group_normality and comparison_normality:
                            test_type = "t-test"
                            t_stat, p_value = ttest_ind(group_values, comparison_values, equal_var=False, nan_policy='omit')
                        else:
                            test_type = "Mann-Whitney U"
                            t_stat, p_value = mannwhitneyu(group_values, comparison_values, alternative='two-sided')
                        
                        # Use Biserial Correlation for both groups to allow for comparison
                        effect_size = biserial_correlation(t_stat, group_values, comparison_values)

                        if p_value < 0.05:
                            result = {
                                "Property": column,
                                "Category": category,
                                "Test Type": test_type,
                                "T-Statistic": t_stat,
                                "P-Value": p_value,
                                "Effect Size": effect_size,
                                "Group Mean": group_values.mean(),
                                "Comparison Mean": comparison_values.mean(),
                            }

                            if (category == "Top 10"):
                                top_10_results.append(result)
                            elif (category == "Bottom 10"):
                                bottom_10_results.append(result)
                            else:
                                comparison_results.append(result)

                    except Exception as e:
                        print(f"Error in statistical test for column '{column}': {e}")

        # Sort each result by Effect Size, then concat for saving to csv
        top_10_results_df = pd.DataFrame(top_10_results).sort_values(by="Effect Size", ascending=True)
        bottom_10_results_df = pd.DataFrame(bottom_10_results).sort_values(by="Effect Size", ascending=True)
        comparison_results_df = pd.DataFrame(comparison_results).sort_values(by="Effect Size", ascending=True)

        results_df = pd.concat([top_10_results_df, bottom_10_results_df, comparison_results_df], ignore_index=True)

        # Save results to CSV
        output_file = os.path.join(output_directory, f"{self.gene_name}_properties_analysis.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Property analysis saved to {output_file}")