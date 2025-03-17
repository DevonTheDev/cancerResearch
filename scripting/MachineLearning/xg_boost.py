import logging
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from xgboost import XGBClassifier

# Constants
FEATURE_IMPORTANCE_CUTOFF = 0.005  # Minimum feature importance threshold

def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

param_grid = {
    'n_estimators': [50, 100, 200, 500, 800, 1000],
    'max_depth': [4, 6, 8, 10, 15, 20, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1, 0.5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1]
}

def train_ml_model(X_train, y_train, RANDOM_STATE):
    """Trains an XGBoost classifier with optimized hyperparameters."""
    model = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model

def evaluate_model(model, data, xg_features):

    X_test = data["X_test"]

    X_test_selected = X_test[xg_features]

    """Evaluates the trained model, saves results, and returns key metrics."""
    y_pred = model.predict(X_test_selected)

    return y_pred

def run_ml_model(result, RANDOM_STATE):
    """Runs the ML model, processes data, saves the trained model, and returns key results."""
    setup_logging()

    # Structure as per MLWorker.py function extract_data(self, processed_file)
    X_train = result["X_train"]
    y_train = result["y_train"]
    X = result["X"]

    # First Model Training (All Features)
    initial_model = train_ml_model(X_train, y_train, RANDOM_STATE)

    # Feature Selection Based on Importance
    feature_importances = initial_model.feature_importances_
    selected_feature_indices = np.where(feature_importances >= FEATURE_IMPORTANCE_CUTOFF)[0]
    selected_features = X.columns[selected_feature_indices]

    # Ensure at least 5 features are selected
    if len(selected_features) == 0:
        logging.warning(f"No features met the cutoff of {FEATURE_IMPORTANCE_CUTOFF}, selecting top 5 features instead.")
        selected_feature_indices = np.argsort(feature_importances)[-5:]
        selected_features = X.columns[selected_feature_indices]

    scaler = StandardScaler()
    scaled_feature_importance = scaler.fit_transform(feature_importances.reshape(-1, 1)).flatten()

    # Create feature importance dictionary for selected features
    selected_feature_importance = {feature: scaled_feature_importance[idx] for feature, idx in zip(selected_features, selected_feature_indices)}

    # Filter dataset to selected features
    X_train_selected = X_train[selected_features]

    # Second Model Training (Only Selected Features)
    final_model = train_ml_model(X_train_selected, y_train, RANDOM_STATE)

    return (final_model, selected_features, selected_feature_importance)