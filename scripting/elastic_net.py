import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Constants
RANDOM_STATE = 42
FEATURE_IMPORTANCE_THRESHOLD = 0.001  # Minimum importance for feature selection
MODEL_FOLDER = os.path.join(os.getcwd(), "ml_models", "elastic_net_models")
PROCESSED_FOLDER = os.path.join(os.getcwd(), "Processed_Data", "3_properties_merged")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hyperparameter grid for Elastic Net
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Balance between L1 (Lasso) & L2 (Ridge)
}

# Get all CSV files in the processed folder
csv_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

if not csv_files:
    logging.error("No processed gene-specific CSV files found. Exiting...")
    exit()

def clean_data(df, prefixes_to_remove=None):
    """
    Cleans the dataset by:
    1. Removing columns with NaN values.
    2. Removing columns with fewer than 2 unique values.
    3. Removing columns with values exceeding dtype('float64') max.
    4. Removing columns that start with specific prefixes.
    """
    initial_cols = df.shape[1]

    # Remove columns with NaN values
    df = df.dropna(axis=1)

    # Remove columns with fewer than 2 unique values
    df = df.loc[:, df.nunique() > 1]

    # Ensure only numeric columns are processed
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]  # Keep only numeric columns

    # Remove columns with extremely large values that exceed float64 limits
    float64_max = np.finfo(np.float64).max
    df_numeric = df_numeric.loc[:, (df_numeric.abs() < float64_max).all(axis=0)]

    # Remove columns with specific prefixes
    if prefixes_to_remove:
        df_numeric = df_numeric.loc[:, ~df_numeric.columns.str.startswith(tuple(prefixes_to_remove))]

    # Merge back with non-numeric columns (if necessary)
    df = pd.concat([df[df.columns.difference(numeric_cols)], df_numeric], axis=1)

    final_cols = df.shape[1]
    logging.info(f"Data cleaned: Removed {initial_cols - final_cols} columns.")

    return df

def train_elastic_net(X_train, y_train):
    """
    Trains an Elastic Net model using GridSearchCV for hyperparameter tuning.
    """
    model = ElasticNet(random_state=RANDOM_STATE, max_iter=10000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = ElasticNet(**grid_search.best_params_, random_state=RANDOM_STATE, max_iter=10000)
    best_model.fit(X_train, y_train)

    return best_model

def select_important_features(model, X):
    """
    Selects features based on non-zero coefficients from the trained Elastic Net model.
    """
    feature_importances = np.abs(model.coef_)
    selected_indices = np.where(feature_importances >= FEATURE_IMPORTANCE_THRESHOLD)[0]
    
    if len(selected_indices) == 0:
        logging.warning("No features met the threshold. Selecting top 5 features instead.")
        selected_indices = np.argsort(feature_importances)[-5:]  # Select top 5 features as fallback

    selected_features = X.columns[selected_indices]
    return selected_features

def evaluate_and_save_model(model, X_test, y_test, gene_name):
    """
    Evaluates model performance and saves the trained model.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    logging.info(f"{gene_name} | R² Score: {r2:.4f} | MSE: {mse:.4f}")

    # Save model
    model_filename = os.path.join(MODEL_FOLDER, f"elastic_net_{gene_name}.joblib")
    joblib.dump(model, model_filename)
    logging.info(f"Model saved as {model_filename}")

    return {
        "gene": gene_name,
        "model_path": model_filename,
        "r2_score": r2,
        "mse": mse
    }

def run_elastic_net():
    """
    Loads data, standardizes it, trains Elastic Net models with feature selection, and saves them.
    """
    results = []

    for csv_file in csv_files:
        gene_name = os.path.splitext(csv_file)[0]  # Extract gene name from filename
        logging.info(f"Processing: {gene_name}")

        # Load dataset
        file_path = os.path.join(PROCESSED_FOLDER, csv_file)
        df = pd.read_csv(file_path)

        if "Pearson_Correlation" not in df.columns:
            logging.warning(f"Skipping {csv_file} - No 'Pearson_Correlation' column found.")
            continue

        # Clean dataset
        prefixes_to_exclude = ["AUTOCORR", "ExactMolWt"]
        df = clean_data(df, prefixes_to_exclude)

        # Define features (X) and target (y)
        X = df.iloc[:, 15:]  # Features (assuming first 15 columns are metadata)
        y = df["Pearson_Correlation"]  # Target

        if X.shape[1] == 0:
            logging.warning(f"Skipping {csv_file} - No valid features found after cleaning.")
            continue

        # Standardize features using Z-score normalization
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        # Train initial Elastic Net model
        initial_model = train_elastic_net(X_train, y_train)

        # Feature Selection Based on Coefficients
        selected_features = select_important_features(initial_model, X_train)

        logging.info(f"Selected {len(selected_features)} features for retraining.")

        # Filter dataset to selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        # Train final Elastic Net model with selected features
        final_model = train_elastic_net(X_train_selected, y_train)

        # Evaluate and save results
        result = evaluate_and_save_model(final_model, X_test_selected, y_test, gene_name)
        results.append(result)

    return results

# Run the model training
if __name__ == "__main__":
    results = run_elastic_net()
    logging.info("Elastic Net training completed.")