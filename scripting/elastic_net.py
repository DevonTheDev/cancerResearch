import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Constants
RANDOM_STATE = 42

# Directories
parent_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(os.path.dirname(parent_dir), "Processed_Data", "3_properties_merged")
os.makedirs(processed_folder, exist_ok=True)  # Ensure output directory exists

def calculate_r2_for_each_feature(processed_file):
    """Calculates the R² value for each feature with Pearson correlation."""
    
    df = pd.read_csv(processed_file)

    df = df[df.columns.drop(list(df.filter(regex='AUTOCORR')))]
    
    # Handle missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf with NaN
    df.dropna(axis=1, inplace=True)  # Drop columns with NaN values
    df.fillna(0, inplace=True)  # Replace remaining NaNs with zero
    
    # Ensure valid columns
    target_column = "Pearson_Correlation"
    feature_columns = df.columns[15:]  # Skip first 15 columns (adjust if needed)
    
    if target_column not in df.columns:
        print(f"Skipping {processed_file}: Pearson_Correlation column missing.")
        return None
    
    results = []
    
    for feature in feature_columns:
        X = df[[feature]].astype(float)  # Single feature as X
        y = df[target_column].astype(float)  # Pearson correlation as target
        
        # Fit a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Compute R² score
        r2 = r2_score(y, y_pred)
        results.append((feature, r2))
    
    # Convert results to a DataFrame and sort by R²
    results_df = pd.DataFrame(results, columns=["Feature", "R² Value"])
    results_df = results_df.sort_values(by="R² Value", ascending=False)
    
    return results_df

def run_analysis():
    """Runs the feature R² analysis for all genes."""
    
    processed_files = [os.path.join(processed_folder, file) for file in os.listdir(processed_folder) if file.endswith(".csv")]
    
    if not processed_files:
        print("No processed files available. Exiting analysis.")
        return None
    
    for processed_file in processed_files:
        print(f"Processing file: {processed_file}")
        results_df = calculate_r2_for_each_feature(processed_file)
        
        if results_df is not None:
            print(f"\nTop Features Ranked by R² for {processed_file}:\n")
            print(results_df.head(10))  # Print top 10 most predictive features
        
            # Plot R² values
            plt.figure(figsize=(10, 5))
            plt.barh(results_df["Feature"].head(10), results_df["R² Value"].head(10), color='blue')
            plt.xlabel("R² Value")
            plt.ylabel("Feature")
            plt.title(f"Top 10 Features by R² for {os.path.basename(processed_file)}")
            plt.gca().invert_yaxis()
            plt.show()

def main():
    """Main function to start analysis."""
    print("Starting feature R² ranking analysis...")
    run_analysis()

if __name__ == "__main__":
    main()