import os
import glob
import pandas as pd
import file_manager
from scipy.stats import zscore

# Initialize the folder creator
folder_creator = file_manager.OrderedFolderCreator()

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the path to the Processed_Data folder
processed_data_path = os.path.join(parent_dir, 'Processed_Data', '*_properties_merged')

# Get a list of all .csv files in the directory
csv_files = glob.glob(os.path.join(processed_data_path, '*.csv'))

ranked_files = glob.glob(os.path.join(parent_dir, "*_gmt_files", "*.csv"))

# Define the output directory for GMT files
output_directory = folder_creator.create_folder(os.path.join(parent_dir, 'Processed_Data'), 'gmt_files')

# Parameters
top_percentile = 0.1
bottom_percentile = 0.1

# Function to create gene sets (drug sets) for each property
def create_gmt(data, top_percentile, bottom_percentile, output_file):
    drugs = data.iloc[:, 1]  # Extract drug names from the second column
    properties = data.columns[14:]  # Properties start from the 15th column (index 14)

    # Normalize the properties using Z-score
    normalized_data = data.copy()

    for prop in properties:
        if data[prop].dtype in ['float64', 'int64']:  # Ensure the column is numeric
            col_values = data[prop].dropna()  # Drop NaN values
            if len(col_values.unique()) > 1:  # Check if there are at least two unique values
                normalized_data[prop] = zscore(data[prop], nan_policy='omit')
            else:
                # If all values are identical or the column is entirely NaN, set the column to NaN
                normalized_data[prop] = float('nan')

    gmt_entries = []

    for prop in properties:
        if normalized_data[prop].isna().all():  # Skip properties where all values are NaN
            continue

        # Sort drugs by the normalized property value
        sorted_data = normalized_data.sort_values(by=prop, ascending=False)
        
        # Define top and bottom drugs
        top_drugs = sorted_data.iloc[:int(len(sorted_data) * top_percentile), 1].tolist()  # Drug names
        bottom_drugs = sorted_data.iloc[-int(len(sorted_data) * bottom_percentile):, 1].tolist()  # Drug names

        # Add to GMT format
        gmt_entries.append(f"{prop}_HIGH\tdrugs_with_high_{prop}\t" + "\t".join(top_drugs))
        gmt_entries.append(f"{prop}_LOW\tdrugs_with_low_{prop}\t" + "\t".join(bottom_drugs))

    # Write to GMT file
    with open(output_file, "w") as gmt_file:
        gmt_file.write("\n".join(gmt_entries))
    print(f".gmt file created successfully: {output_file}")

def create_ranked_data(data, output_directory):
    # Extract the Drug and Pearson_correlation columns
    ranked_data = data[['Drug', 'Pearson_Correlation']]
    
    # Sort the data by Pearson_correlation in descending order
    ranked_data = ranked_data.sort_values(by='Pearson_Correlation', ascending=False)
    
    # Get the base name of the file (without extension) for the output file name
    file_name = os.path.basename(csv_file).replace('.csv', '')
    ranked_output_file = os.path.join(output_directory, f"{file_name}_ranked.csv")
    
    # Save the ranked data to a new CSV file
    ranked_data.to_csv(ranked_output_file, index=False)
    print(f"Ranked data CSV file created successfully: {ranked_output_file}")

# Loop through each .csv file
for csv_file in csv_files:
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)
    
    # Get the base name of the file (without extension) for the output file name
    file_name = os.path.basename(csv_file).replace('.csv', '')
    gmt_output_file = os.path.join(output_directory, f"{file_name}_chemical_properties.gmt")
    
    # Call the create_gmt function
    create_gmt(data, top_percentile, bottom_percentile, gmt_output_file)
    
    # Call the create_ranked_data function
    create_ranked_data(data, output_directory)