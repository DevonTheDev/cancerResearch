import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from scripting import ml_file_cleaner as mlfc

# Constants
RANDOM_STATE = 42
FEATURE_IMPORTANCE_CUTOFF = 0.005  # Minimum feature importance threshold

# Directories
processed_folder = os.path.join(os.getcwd(), "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# Only create cleaned files if none exist currently
if not (csv_files):
    mlfc.MLFileCleaner.run_file_clean()

def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

param_grid = {
    'n_estimators': [200, 500, 800, 1000],
    'max_depth': [4, 6, 8, 10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1]
}

def train_ml_model(X_train, y_train):
    """Trains an XGBoost classifier with optimized hyperparameters."""
    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model

def evaluate_and_save_model(model, X_test, y_test, selected_features, processed_file):
    """Evaluates the trained model, saves results, and returns key metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Final Model Accuracy: {accuracy:.4f}")

    # Feature importances
    feature_importance = model.feature_importances_
    selected_feature_importances = [feature_importance[i] for i, col in enumerate(X_test.columns) if col in selected_features]

    # Save model
    base_filename = os.path.basename(processed_file).replace(".csv", ".joblib")
    model_filepath = os.path.join(os.getcwd(), "ml_models", "xgboost_models")
    os.makedirs(model_filepath, exist_ok=True)

    model_filename = os.path.join(model_filepath, f"xgboost_{base_filename}")

    model_data = {
        "model": model,
        "accuracy": accuracy,
        "feature_importances": selected_feature_importances,
        "selected_features": list(selected_features),
        "best_params": model.get_params(),
        "classification_report": classification_rep,
    }

    joblib.dump(model_data, model_filename)
    logging.info(f"Model saved as {model_filename}")

    return {
        "processed_file": processed_file,
        "model_path": model_filename,
        "accuracy": accuracy,
        "classification_report": classification_rep
    }

def run_ml_model():
    """Runs the ML model, processes data, saves the trained model, and returns key results."""
    setup_logging()

    if not csv_files:
        logging.error("No processed files available. Exiting ML process.")
        return None

    results = []

    for csv_file in csv_files:
        logging.info(f"Processing file: {csv_file}")

        # Load dataset
        file_path = os.path.join(processed_folder, csv_file)
        df = pd.read_csv(file_path)

        if "Label" not in df.columns:
            logging.warning(f"Skipping {csv_file} - 'Label' column not found.")
            continue

        # Extract features and target variable
        feature_columns = df.columns[3:]  # Feature names
        X = df.iloc[:, 3:]  # Feature matrix
        y = df["Label"]  # Target variable

        # Ensure valid dataset
        if X.shape[1] == 0:
            logging.warning(f"Skipping {csv_file} - No valid features found.")
            continue

        # Handle NaN, Inf, and extremely large values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        # First Model Training (All Features)
        initial_model = train_ml_model(X_train, y_train)

        # Feature Selection Based on Importance
        feature_importances = initial_model.feature_importances_
        selected_feature_indices = np.where(feature_importances >= FEATURE_IMPORTANCE_CUTOFF)[0]
        selected_features = feature_columns[selected_feature_indices]

        # Ensure at least 5 features are selected
        if len(selected_features) == 0:
            logging.warning(f"No features met the cutoff of {FEATURE_IMPORTANCE_CUTOFF}, selecting top 5 features instead.")
            selected_feature_indices = np.argsort(feature_importances)[-5:]
            selected_features = feature_columns[selected_feature_indices]

        logging.info(f"Selected {len(selected_features)} features for retraining.")

        # Filter dataset to selected features
        X_train_selected = pd.DataFrame(X_train[:, selected_feature_indices], columns=selected_features)
        X_test_selected = pd.DataFrame(X_test[:, selected_feature_indices], columns=selected_features)


        # Second Model Training (Only Selected Features)
        final_model = train_ml_model(X_train_selected, y_train)

        # Evaluate and save results
        result = evaluate_and_save_model(final_model, X_test_selected, y_test, selected_features, csv_file)
        results.append(result)

    return results[-1] if results else None