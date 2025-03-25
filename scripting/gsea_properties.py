import pandas as pd
import numpy as np
import os

# Define the columns to always extract
FIXED_COLUMNS = ["Drug", "Pearson_Correlation"]

# Get the parent directory of the current script
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Path for processed CSV folder
processed_folder = os.path.join(os.path.dirname(parent_dir), "Processed_Data", "3_properties_merged")
output_folder = os.path.join(processed_folder, "processed_properties")
os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists

# Get all CSV files in the parent directory
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# List to store output file paths
processed_files = []

def generate_files():
    for csv_file in csv_files:
        file_path = os.path.join(processed_folder, csv_file)

        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Check if required columns exist
            missing_columns = [col for col in FIXED_COLUMNS if col not in df.columns]
            if missing_columns:
                print(f"Skipping {csv_file} - Missing columns: {missing_columns}")
                continue

            # Calculate cutoff index for top and bottom %
            num_rows = len(df)
            cutoff = int(0.01 * num_rows)  # 1% cutoff
            
            top_10_percent = df.iloc[-cutoff:]
            bottom_10_percent = df.iloc[:cutoff]
            filtered_df = pd.concat([bottom_10_percent, top_10_percent])

            # Select the fixed columns
            selected_columns = filtered_df[FIXED_COLUMNS].copy()

            # Add the Label column based on Pearson_Correlation
            selected_columns["Label"] = selected_columns["Pearson_Correlation"].apply(lambda x: 0 if x < 0 else 1)  # 0 = Susceptible, 1 = Resistant

            # Select additional columns from index 15 onward
            if len(df.columns) > 15:
                additional_columns = filtered_df.iloc[:, 15:]

                # Identify and remove columns with NaN, Inf, or extremely large values
                valid_additional_columns = additional_columns.loc[:, additional_columns.apply(
                    lambda x: x.notna().all() and np.isfinite(x).all() and (x.abs() < np.finfo(np.float32).max).all()
                )]

                processed_df = pd.concat([selected_columns, valid_additional_columns], axis=1)
            else:
                processed_df = selected_columns  # If not enough columns, just use fixed columns

            # Save processed CSV to new folder
            processed_file_path = os.path.join(output_folder, csv_file)
            processed_df.to_csv(processed_file_path, index=False)

            # Store processed file path for later use
            processed_files.append(processed_file_path)

            print(f"Processed and saved: {csv_file}")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

# Run the file generation process
generate_files()