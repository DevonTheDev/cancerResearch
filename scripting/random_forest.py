from itertools import combinations
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

# Constants
FIXED_COLUMNS = ["Drug", "Pearson_Correlation"]
RANDOM_STATE = 42  # Ensures reproducibility
FEATURE_CUTOFF = 0.005  # Threshold for selecting important features
CUTOFF_PERCENT = 0.01  # Percentage of top and bottom data to select

# Directories
parent_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(os.path.dirname(parent_dir), "Processed_Data", "3_properties_merged")
output_folder = os.path.join(processed_folder, "processed_properties")
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]


def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_process_csv(csv_file):
    """Loads a CSV file, filters data, and selects relevant columns."""
    file_path = os.path.join(processed_folder, csv_file)
    
    try:
        df = pd.read_csv(file_path)

        # Check for missing required columns
        missing_columns = [col for col in FIXED_COLUMNS if col not in df.columns]
        if missing_columns:
            logging.warning(f"Skipping {csv_file} - Missing columns: {missing_columns}")
            return None

        # Calculate cutoff index for top and bottom 1%
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
            additional_columns = filtered_df.iloc[:, 15:]

            # Remove NaN, Inf, or extremely large values
            valid_columns = additional_columns.loc[:, additional_columns.apply(
                lambda x: x.notna().all() and np.isfinite(x).all() and (x.abs() < np.finfo(np.float32).max).all()
            )]

            processed_df = pd.concat([processed_df, valid_columns], axis=1)

        # Save processed CSV
        processed_file_path = os.path.join(output_folder, csv_file)
        processed_df.to_csv(processed_file_path, index=False)
        logging.info(f"Processed and saved: {csv_file}")

        return processed_file_path

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}")
        return None


def generate_files():
    """Processes all CSV files and returns a list of processed file paths."""
    processed_files = [load_and_process_csv(file) for file in csv_files]
    return [file for file in processed_files if file is not None]  # Remove None values


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
        n_iter=100,
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
    model_filename = os.path.join(os.getcwd(), f"random_forest_{base_filename}")

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


# Possible feature prefixes to exclude (AUTOCORR is always included)
EXCLUDE_PREFIXES = ["BCUT", "EState_", "PEOE_", "SMR_", "SlogP_", "VSA_EState", "qed"]

def generate_exclusion_combinations():
    """Generate all combinations of EXCLUDE_PREFIXES while always including 'AUTOCORR'."""
    all_combinations = []
    
    # Generate all possible subsets of EXCLUDE_PREFIXES
    for r in range(len(EXCLUDE_PREFIXES) + 1):
        for subset in combinations(EXCLUDE_PREFIXES, r):
            all_combinations.append(("AUTOCORR",) + subset)  # Ensure 'AUTOCORR' is always included

    return all_combinations

def run_ml_model(exclude_autocorr):
    """Runs the ML model for every processed file across all exclusion combinations and returns the best result."""
    
    setup_logging()
    sns.set_style("whitegrid")

    # Generate processed files
    processed_files = generate_files()
    if not processed_files:
        logging.error("No processed files available. Exiting ML process.")
        return None

    # Get all exclusion combinations
    exclusion_combinations = generate_exclusion_combinations()
    best_combination = None
    best_avg_accuracy = -1

    # Dictionary to store results
    accuracy_results = {}

    # Iterate through all exclusion combinations
    for exclude_set in exclusion_combinations:
        logging.info(f"Testing exclusion set: {exclude_set}")
        accuracies = []

        for processed_file in processed_files:
            logging.info(f"Processing file: {processed_file} with exclusions: {exclude_set}")

            # Load dataset
            df = pd.read_csv(processed_file)

            # Drop columns that start with any prefix in exclude_set
            df = df.drop(columns=[col for col in df.columns if any(col.startswith(prefix) for prefix in exclude_set)], errors="ignore")

            X = df.iloc[:, 3:]  # Features (excluding first three columns)
            y = df.iloc[:, 2]  # Target variable (Resistance)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
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

            # Evaluate the model
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # Compute the average accuracy for this exclusion set
        avg_accuracy = np.mean(accuracies)
        accuracy_results[exclude_set] = avg_accuracy

        logging.info(f"Exclusion set {exclude_set} achieved average accuracy: {avg_accuracy:.4f}")

        # Update the best combination
        if avg_accuracy > best_avg_accuracy or (avg_accuracy == best_avg_accuracy and len(exclude_set) < len(best_combination)):
            best_avg_accuracy = avg_accuracy
            best_combination = exclude_set

    logging.info(f"Best exclusion set: {best_combination} with accuracy: {best_avg_accuracy:.4f}")
    return {"best_exclusion_set": best_combination, "best_accuracy": best_avg_accuracy}