import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scripting.MachineLearning import ml_file_cleaner as mlfc

# Constants
FEATURE_CUTOFF = 0.005  # Threshold for selecting important features

def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_ml_model(X_train, y_train, RANDOM_STATE):
    """Performs hyperparameter tuning and trains a Random Forest model."""
    param_dist = {
        "n_estimators": [50, 100, 200, 500, 800, 1000],
        "max_depth": [5, 10, 20, 50, 100],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 10],
        "bootstrap": [True, False]
    }

    # RandomizedSearchCV for hyperparameter tuning
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=250,
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


def evaluate_model(model, data, selected_features):
    """Evaluates the trained model using only the selected features."""

    X_test = data["X_test"]

    X_test_selected = X_test[selected_features]

    y_pred = model.predict(X_test_selected)
    return y_pred


def run_ml_model(result, RANDOM_STATE):
    """Runs the ML model, processes data, saves the trained model, and returns key results."""
    setup_logging()

    # Extract training data
    X_train = result["X_train"]
    y_train = result["y_train"]
    X = result["X"]

    # Train ML model
    model = train_ml_model(X_train, y_train, RANDOM_STATE)

    # Feature selection based on importance
    feature_importance = model.feature_importances_
    selected_feature_indices = np.where(feature_importance >= FEATURE_CUTOFF)[0]
    selected_features = list(X.columns[selected_feature_indices])

    # Ensure at least 5 features are selected
    if len(selected_features) == 0:
        logging.warning(f"No features met the cutoff of {FEATURE_CUTOFF}, selecting top 5 features instead.")
        selected_feature_indices = np.argsort(feature_importance)[-5:]
        selected_features = list(X.columns[selected_feature_indices])

    scaler = StandardScaler()
    scaled_feature_importance = scaler.fit_transform(feature_importance.reshape(-1, 1)).flatten()

    # Create feature importance dictionary for selected features
    selected_feature_importance = {feature: scaled_feature_importance[idx] for feature, idx in zip(selected_features, selected_feature_indices)}

    # Retrain with selected features
    X_train_selected = X_train[selected_features]
    final_model = train_ml_model(X_train_selected, y_train, RANDOM_STATE)

    return final_model, selected_features, selected_feature_importance
