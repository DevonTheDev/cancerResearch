import os
import pandas as pd
import numpy as np

class MLFolderFinder:
    def __init__(self, target_folder="cancerResearch"):
        self.target_folder = target_folder
        self.parent_dir = self.find_parent_directory()
        self.processed_folder = os.path.join(
            self.parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties"
        )
        os.makedirs(self.processed_folder, exist_ok=True)

    def find_parent_directory(self):
        """Finds the absolute path of the specified parent directory, ensuring it returns the target folder itself."""
        current_dir = os.path.abspath(__file__)  # Get current script path
        
        while True:
            parent_dir, last_folder = os.path.split(current_dir)  # Split path into parent & last directory
            if last_folder == self.target_folder:
                return current_dir  # Return "cancerResearch" directory itself
            if not last_folder:  # Stop if we reach the root directory
                raise FileNotFoundError(f"Parent folder '{self.target_folder}' not found in path hierarchy.")
            current_dir = parent_dir  # Move up one level

    def get_processed_folder(self):
        """Returns the path to the processed folder."""
        return self.processed_folder


# Constants
FIXED_COLUMNS = ["Drug", "Pearson_Correlation"]  # Columns to validate CSV structure
DROPPED_COLUMNS = ["ExactMolWt", "FractionCSP3", "HallKierAlpha", "HeavyAtomCount", "HeavyAtomMolWt", "LabuteASA", "Phi", "SPS", "TPSA", "NumValenceElectrons"]  # Columns to drop from properties
DROPPED_PREFIXES = ["AUTOCORR", "BCUT2D", "BalabanJ", "BertzCT", "Chi", "EState_VSA", "FpDensityMorgan", "Kappa", "PEOE_VSA", "SMR_VSA", "VSA_EState"]
CUTOFF_PERCENT = 0.01  # Percentage of top and bottom data to select

# Directories
parent_dir = MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged")
output_folder = os.path.join(processed_folder, "ml_processed_properties")
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

class MLFileCleaner:
    @staticmethod
    def load_and_process_csv(csv_file):
        """Loads a CSV file, filters data, and selects relevant columns."""
        file_path = os.path.join(processed_folder, csv_file)
        
        try:
            df = pd.read_csv(file_path)

            # Check for missing required columns
            missing_columns = [col for col in FIXED_COLUMNS if col not in df.columns]
            if missing_columns:
                print(f"Skipping {csv_file} - Missing columns: {missing_columns}")
                return None

            # Calculate cutoff index for top and bottom %
            cutoff = max(1, int(CUTOFF_PERCENT * len(df)))  # Ensure at least 1 row is selected
            top_1_percent = df.iloc[-cutoff:]
            bottom_1_percent = df.iloc[:cutoff]
            filtered_df = pd.concat([bottom_1_percent, top_1_percent])

            # Select fixed columns
            processed_df = filtered_df[FIXED_COLUMNS].copy()

            # Label classification based on Pearson_Correlation
            processed_df["Label"] = processed_df["Pearson_Correlation"].apply(lambda x: 0 if x < 0 else 1)

            # Add additional properties from column index 15 onwards
            if len(df.columns) > 15:
                additional_columns = filtered_df.iloc[:, 15:].copy()  # Get drug properties

                # Step 1: Drop columns with NaN, Inf, or extremely large values
                max_float = np.finfo(np.float32).max
                def is_valid_column(col):
                    return col.notna().all() and np.isfinite(col).all() and (col.abs() < max_float).all()
                
                additional_columns = additional_columns.loc[:, additional_columns.apply(is_valid_column)]

                # Step 2: Drop columns with disallowed prefixes
                additional_columns = additional_columns.loc[:, ~additional_columns.columns.str.startswith(tuple(DROPPED_PREFIXES))]

                # Step 3: Drop specific unwanted columns
                additional_columns = additional_columns.drop(columns=DROPPED_COLUMNS, errors='ignore')

                # Step 4: Drop columns with fewer than 3 unique values
                additional_columns = additional_columns.loc[:, additional_columns.nunique() >= 3]

                # Step 5: Concatenate cleaned columns
                processed_df = pd.concat([processed_df, additional_columns], axis=1)

            # Save processed CSV
            processed_file_path = os.path.join(output_folder, csv_file)
            processed_df.to_csv(processed_file_path, index=False)
            print(f"Processed and saved: {csv_file}")

            return processed_file_path

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None

    @staticmethod
    def run_file_clean():
        """Processes all CSV files in the input directory."""
        csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

        for csv_file in csv_files:
            MLFileCleaner.load_and_process_csv(csv_file)  # Fixed function call