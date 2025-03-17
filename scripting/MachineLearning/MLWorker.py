from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from scripting.MachineLearning import random_forest, xg_boost, neural_network
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from scripting.MachineLearning import ml_file_cleaner as mlfc
import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42

parent_dir = mlfc.MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# Only create cleaned files if none exist currently
if not (csv_files):
    mlfc.MLFileCleaner.run_file_clean()
    csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]
    

class MLWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

        self.results = []

        for processed_file in csv_files:
            self.extract_data(processed_file)

    def extract_data(self, processed_file):
        print(f"Processing file: {processed_file}")

        df = pd.read_csv(os.path.join(processed_folder, processed_file))
        
        X = df.iloc[:, 3:]  # Features (excluding first three columns)
        y = df.iloc[:, 2]  # Target variable (Binary Resistance)

        # First split: Train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE
        )

        # Second split: Train-validation (taking a portion of training data for neural network)
        X_train_for_val, X_val, y_train_for_val, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=RANDOM_STATE
        )

        # 🚀 Apply Standard Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # Fit & transform training data
        X_val_scaled = scaler.transform(X_val)  # Transform validation data
        X_test_scaled = scaler.transform(X_test)  # Transform test data

        X_train = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_val = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

        self.results.append({
            "X_train": X_train,
            "X_test": X_test,
            "X_val": X_val,
            "y_train": y_train,
            "y_test": y_test,
            "y_val": y_val,
            "X": X,
            "y": y,
            "file_name": processed_file
        })

    def run(self):
        print("Running Machine Learning Models...")

        final_results = {}
        feature_importances = {}  # Use a dictionary to store cumulative feature importances

        for data in self.results:
            
            # Run Random Forest Model
            rf_model, rf_features, rf_importances = random_forest.run_ml_model(data, RANDOM_STATE)
            rf_preds = random_forest.evaluate_model(rf_model, data, rf_features)

            # Update feature importances for Random Forest
            for feature, importance in rf_importances.items():
                if feature in feature_importances:
                    feature_importances[feature] += importance
                else:
                    feature_importances[feature] = importance

            # Run XGBoost Model
            xg_model, xg_features, xg_importances = xg_boost.run_ml_model(data, RANDOM_STATE)
            xg_preds = xg_boost.evaluate_model(xg_model, data, xg_features)

            # Update feature importances for XGBoost
            for feature, importance in xg_importances.items():
                if feature in feature_importances:
                    feature_importances[feature] += importance
                else:
                    feature_importances[feature] = importance

            # Run Neural Network Model
            nn_model = neural_network.run_mlp(data)
            nn_preds = neural_network.evaluate_model(nn_model, data).squeeze()  # Ensure shape consistency

            # Ensemble Predictions (Bagging)
            stacked_predictions = np.vstack((rf_preds, xg_preds, nn_preds)).T.astype(int)
            bagged_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=stacked_predictions)

            y_test = np.asarray(data["y_test"])

            print(f"Type of y_test: {type(y_test)}")
            print(f"Type of bagged_preds: {type(bagged_preds)}")

            # Store final results
            final_results[data["file_name"]] = {
                "random_forest_accuracy": accuracy_score(y_test, rf_preds),
                "xg_boost_accuracy": accuracy_score(y_test, xg_preds),
                "neural_network_accuracy": accuracy_score(y_test, nn_preds),
                "bagged_accuracy": accuracy_score(y_test, bagged_preds),
            }

        # Print or store feature importance values for debugging
        feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        print("Final Cumulative Feature Importances:", feature_importances)

        self.finished.emit(final_results)