import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import logging

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

            # Calculate cutoff index for top and bottom 10%
            num_rows = len(df)
            cutoff = int(0.1 * num_rows)  # 10% cutoff
            
            # Select the top and bottom 10% of Pearson_Correlation
            top_10_percent = df.iloc[-cutoff:]  # Last 10%
            bottom_10_percent = df.iloc[:cutoff]  # First 10%
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hyperparameter tuning space (randomized search for efficiency)
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['sqrt', 'log2']  # Reducing overfitting
}

random_state_value = 1

for processed_file in processed_files:
    logging.info(f"Processing file: {processed_file}")

    # Load the dataset
    df = pd.read_csv(processed_file)

    # Select features (exclude first three columns: 'Drug', 'Pearson_Correlation', 'Resistance')
    X = df.iloc[:, 3:]
    y = df.iloc[:, 2]  # Assuming 'Resistance' is in column index 2

    # Train-test split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state_value
    )

    # Hyperparameter tuning using RandomizedSearchCV (faster than GridSearchCV)
    rf_model = RandomForestClassifier(random_state=random_state_value)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=20,  # Reducing iterations to balance speed/quality
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=random_state_value
    )
    
    random_search.fit(X_train, y_train)

    # Best parameters from the search
    best_params = random_search.best_params_
    logging.info(f"Best Hyperparameters: {best_params}")

    # Train final model with best hyperparameters
    best_rf_model = RandomForestClassifier(**best_params, random_state=random_state_value)
    best_rf_model.fit(X_train, y_train)

    # Save the trained model
    base_filename = os.path.basename(processed_file).replace('.csv', '.joblib')
    model_filename = os.path.join(os.getcwd(), f"random_forest_{base_filename}")
    joblib.dump(best_rf_model, model_filename)
    logging.info(f"Model saved as {model_filename}")

    # Make predictions
    y_pred = best_rf_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Final Model Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")