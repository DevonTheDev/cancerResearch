import os
import logging
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scripting import ml_file_cleaner as mlfc

# Constants
RANDOM_STATE = 42  # Ensures reproducibility
FEATURE_CUTOFF = 0.005  # Threshold for selecting important features

# Directories
parent_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(os.path.dirname(parent_dir), "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# Only create cleaned files if none exist currently
if not (csv_files):
    mlfc.MLFileCleaner.run_file_clean()
    csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_ml_model(X_train, y_train):
    """Performs hyperparameter tuning and trains a Random Forest model."""
    param_dist = {
        "n_estimators": [50, 100, 200, 500, 1000],
        "max_depth": [5, 10, 20, 50, 100, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 10],
        "bootstrap": [True, False]
    }

    # RandomizedSearchCV for hyperparameter tuning
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=300,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE
    )

    random_search.fit(X_train, y_train)
    logging.info(f"Best Hyperparameters: {random_search.best_params_}")

    # Train final model with best parameters
    best_rf_model = RandomForestClassifier(**random_search.best_params_, random_state=RANDOM_STATE)
    best_rf_model.fit(X_train, y_train)

    return best_rf_model


def evaluate_and_save_model(model, X_test, y_test, selected_features, processed_file):
    """Evaluates the trained model, saves results, and returns key metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    logging.info(f"Final Model Accuracy: {accuracy:.4f}")

    # Save only selected features' importances
    feature_importance = model.feature_importances_
    selected_feature_importances = feature_importance[np.isin(X_test.columns, selected_features)]

    # Save model
    base_filename = os.path.basename(processed_file).replace(".csv", ".joblib")
    model_filepath = os.path.join(os.getcwd(), "ml_models", "random_forest_models")
    os.makedirs(model_filepath, exist_ok=True)

    model_filename = os.path.join(model_filepath, f"random_forest_{base_filename}")

    model_data = {
        "model": model,
        "accuracy": accuracy,
        "feature_importances": selected_feature_importances,
        "selected_features": selected_features.tolist(),
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
    sns.set_style("whitegrid")

    if not csv_files:
        logging.error("No processed files available. Exiting ML process.")
        return None

    results = []

    for processed_file in csv_files:
        logging.info(f"Processing file: {processed_file}")

        # Define expected model filename
        base_filename = os.path.basename(processed_file).replace(".csv", ".joblib")
        model_filepath = os.path.join(os.getcwd(), "ml_models", "random_forest_models")
        model_filename = os.path.join(model_filepath, f"random_forest_{base_filename}")

        # ✅ **Check if model already exists**
        if os.path.exists(model_filename):
            logging.info(f"Model already exists for {processed_file}. Loading model from {model_filename}.")
            model_data = joblib.load(model_filename)
            results.append(model_data)  # Store results from existing model
            continue  # Skip training and move to the next file

        # Load dataset
        df = pd.read_csv(os.path.join(processed_folder, processed_file))
        
        X = df.iloc[:, 3:]  # Features (excluding first three columns)
        y = df.iloc[:, 2]  # Target variable (Binary Resistance)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )

        # Train ML model
        model = train_ml_model(X_train, y_train)

        # Feature selection based on importance
        feature_importance = model.feature_importances_
        selected_feature_indices = np.where(feature_importance >= FEATURE_CUTOFF)[0]
        selected_features = X.columns[selected_feature_indices]

        # Ensure at least 5 features are selected
        if len(selected_features) == 0:
            logging.warning(f"No features met the cutoff of {FEATURE_CUTOFF}, selecting top 5 features instead.")
            selected_feature_indices = np.argsort(feature_importance)[-5:]
            selected_features = X.columns[selected_feature_indices]

        # Retrain with selected features
        X_train_selected, X_test_selected = X_train[selected_features], X_test[selected_features]
        model.fit(X_train_selected, y_train)

        # Evaluate and save results
        result = evaluate_and_save_model(model, X_test_selected, y_test, selected_features, processed_file)
        results.append(result)

    return results[-1] if results else None