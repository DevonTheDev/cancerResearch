from sklearn.discriminant_analysis import StandardScaler
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from scripting.MachineLearning import ml_file_cleaner as mlfc
import pandas as pd
from pycaret.classification import *
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

        saved_models = []

        for result in self.results:
            # Load dataset
            print (f"Processing file: {result['file_name']}")
            df = pd.read_csv(os.path.join("Processed_Data", "3_properties_merged", "ml_processed_properties", result["file_name"]))
            data = df.iloc[:, 2:]

            # Step 1: Setup PyCaret with full dataset
            s = setup(data, target='Label', session_id=RANDOM_STATE)

            # Step 2: Train an initial model
            initial_model = compare_models(sort="Accuracy", fold=10)

            # Step 3: Check if feature importance is available
            if hasattr(initial_model, "feature_importances_"):
                # Extract trained feature names from PyCaret
                trained_feature_names = get_config('X_train').columns.tolist()

                # Ensure feature importance size matches feature names
                if len(initial_model.feature_importances_) == len(trained_feature_names):
                    feature_importance_df = pd.DataFrame({
                        'Feature': trained_feature_names,
                        'Importance': initial_model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)

                    # Step 4: Select top 10% features dynamically
                    num_features = max(5, int(len(feature_importance_df) * 0.1))  # Ensure at least 5 features
                    selected_features = feature_importance_df.iloc[:num_features]['Feature'].tolist()

                    # Step 5: Retrain model using only the selected features
                    refined_data = data[selected_features + ['Label']]

                    # Step 6: Setup PyCaret again with refined dataset (avoid redundant transformations)
                    s = setup(refined_data, target='Label', remove_multicollinearity=True, 
                            multicollinearity_threshold=0.85, session_id=RANDOM_STATE)

                    # Step 7: Train top models again (using parallel processing)
                    best_models = compare_models(sort="Accuracy", fold=10, n_select=5)

                    # Step 8: Tune models efficiently with early stopping
                    tuned_models = [tune_model(model, optimize="Accuracy", n_iter=300, fold=10, 
                                            search_library="scikit-optimize", search_algorithm="bayesian", 
                                            early_stopping=True) for model in best_models]

                    # Step 9: Blend models instead of stacking (better generalization)
                    blended = blend_models(tuned_models, fold=10, method="soft")

                    # Step 10: Print accuracy
                    saved_models.append({result["file_name"]: blended})
                else:
                    print("⚠️ Feature Importance Count Mismatch! Skipping feature selection.")
            else:
                print("⚠️ Selected model does not support feature importance. Skipping feature selection.")

        print(saved_models)