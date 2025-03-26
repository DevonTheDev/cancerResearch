from collections import defaultdict
import json
import time
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib import pyplot as plt
import numpy as np
import shap
from sklearn.discriminant_analysis import StandardScaler
from scripting.MachineLearning import ml_file_cleaner as mlfc
import pandas as pd
from pycaret.classification import *
import os
import warnings

os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker.*")

RANDOM_STATE = 42
NUMBER_OF_MODELS = 5
NUMBER_OF_FOLDS = 3
NUMBER_OF_ITERATIONS = 5
FEATURE_PERCENTAGE = 0.25

# Get processed files
parent_dir = mlfc.MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]
if not csv_files:
    mlfc.MLFileCleaner.run_file_clean()
    csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]


# Directories to save models to and look for
# Directory 1 - Saved Independent Models and Metrics
saved_independent_model_path = os.path.join("Processed_Data", "ml_saved_models", "independent_models")
os.makedirs(saved_independent_model_path, exist_ok=True)

# Directory 2 - Saved Shared Models and Metrics
saved_shared_model_path = os.path.join("Processed_Data", "ml_saved_models", "shared_models")
os.makedirs(saved_shared_model_path, exist_ok=True)

# Model name to abbreviation map
model_name_to_abbr = {
    "Logistic Regression": "lr",
    "K Neighbors Classifier": "knn",
    "Naive Bayes": "nb",
    "Decision Tree Classifier": "dt",
    "SVM - Linear Kernel": "svm",
    "SVM - Radial Kernel": "rbfsvm",
    "Gaussian Process Classifier": "gpc",
    "MLP Classifier": "mlp",
    "Ridge Classifier": "ridge",
    "Random Forest Classifier": "rf",
    "Quadratic Discriminant Analysis": "qda",
    "Ada Boost Classifier": "ada",
    "Gradient Boosting Classifier": "gbc",
    "Linear Discriminant Analysis": "lda",
    "Extra Trees Classifier": "et",
    "Extreme Gradient Boosting": "xgboost",
    "Light Gradient Boosting Machine": "lightgbm",
    "CatBoost Classifier": "catboost"
}

class MLWorker(QThread):
    finished = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.results = [self.extract_data(f) for f in csv_files]
        self.feature_importances = defaultdict(float)
    def extract_data(self, processed_file):
        df = pd.read_csv(os.path.join(processed_folder, processed_file))
        X = df.iloc[:, 3:]
        y = df.iloc[:, 2]
        return {"X": X, "y": y, "file_name": processed_file}

    def setup_data(self, data):
        return setup(data, target='Label', session_id=RANDOM_STATE, verbose=False)

    def tune_and_blend_models(self, models):
        tuned_models = [
            tune_model(m, optimize="Accuracy", n_iter=NUMBER_OF_ITERATIONS, fold=NUMBER_OF_FOLDS,
                       search_library="scikit-optimize", search_algorithm="bayesian",
                       early_stopping=True, tuner_verbose=False) for m in models
        ]
        return blend_models(tuned_models, fold=NUMBER_OF_FOLDS, verbose=False)

    def save_metrics(self, path, metrics):
        with open(path, "w") as f:
            json.dump(metrics, f)

    def run(self):
        start_time = time.time()
        print("Running Machine Learning Models...")
        self.run_independent_models()
        self.run_shared_models()
        end_time = time.time()

        print(f"Time Taken: {end_time - start_time:.2f} seconds")

    def run_independent_models(self):
        for result in self.results:
            gene = result["file_name"].replace("_properties_merged.csv", "")
            model_path = os.path.join(saved_independent_model_path, gene)
            print(f"\nProcessing gene: {gene}")

            data = result['X'].copy()
            data['Label'] = result['y']

            if os.path.exists(model_path + ".pkl"):
                metrics_path = model_path + "_independent_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        accuracy = json.load(f).get("accuracy")
                        print(f"✔ Loaded model with stored accuracy: {accuracy}")
                continue

            # Initial setup and comparison
            self.setup_data(data)
            best_models = compare_models(sort="Accuracy", fold=NUMBER_OF_FOLDS, n_select=NUMBER_OF_MODELS, turbo=False, exclude=["knn"])

            # Collect feature importances
            importance_df = pd.DataFrame()
            for model in best_models:
                if hasattr(model, "feature_importances_"):
                    feature_names = get_config("X").columns.tolist()
                    importances = model.feature_importances_
                    if len(importances) == len(feature_names):
                        temp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                        importance_df = pd.concat([importance_df, temp_df])

            if importance_df.empty:
                print(f"⚠️ No valid feature importances for gene: {gene}. Skipping.")
                continue

            # Process feature importances
            importance_df = importance_df.groupby("Feature").mean().reset_index()
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            num_features = max(5, int(len(importance_df) * FEATURE_PERCENTAGE))
            top_features = importance_df.copy().iloc[:num_features]["Feature"].tolist()

            # Make a copy and select top features
            scaled_df = importance_df.copy().iloc[:num_features].reset_index(drop=True)

            # Apply scaling to just the 'Importance' column
            scaler = StandardScaler()
            scaled_importance = scaler.fit_transform(scaled_df[["Importance"]])

            # Add scaled values back into the DataFrame
            scaled_df["ScaledImportance"] = scaled_importance

            # Now you can iterate safely
            for _, row in scaled_df.iterrows():
                name = row["Feature"]
                importance = row["ScaledImportance"]
                self.feature_importances[name] += importance

            # Filter data to top features
            refined_data = result['X'][top_features].copy()
            refined_data['Label'] = result['y']

            # Setup with reduced features
            self.setup_data(refined_data)

            # Tune and blend models on reduced features
            blended = self.tune_and_blend_models(best_models)
            score = pull()
            accuracy = float(score.iloc[NUMBER_OF_FOLDS, 1])

            # Save everything
            save_model(blended, model_path)
            self.save_metrics(model_path + "_independent_metrics.json", {"accuracy": accuracy, "top_features": importance_df.iloc[:num_features].to_json(orient="records")})

            # Generate and save SHAP values and plots
            try:
                X = get_config("X")
                y = get_config("y")

                shap_output_path = os.path.join(model_path + "_shap_outputs")
                os.makedirs(shap_output_path, exist_ok=True)

                # Tree-based SHAP (fast)
                try:
                    explainer = shap.Explainer(blended)
                    shap_values = explainer(X)
                    shap_X = X
                except Exception as tree_error:
                    # Fallback for non-tree models
                    background = shap.utils.sample(X, 100)
                    explainer = shap.KernelExplainer(blended.predict, background)

                    shap_X = X.sample(100)
                    shap_values = explainer.shap_values(shap_X)

                plt.figure()
                shap.summary_plot(shap_values, shap_X, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(shap_output_path, "shap_summary.png"), dpi=300)
                plt.close()

            except Exception as shap_error:
                print(f"⚠️ Could not generate SHAP summaries for {model_path}: {shap_error}")


        self.save_metrics(
            saved_independent_model_path + "_feature_importances.json",
            {
                "feature_importances": dict(
                    pd.Series(self.feature_importances).sort_values(ascending=False)
                )
            }
        )
        self.feature_importances = [] # Reset for shared model run

    def run_shared_models(self):
        model_avg_acc = defaultdict(list)

        # First pass: identify top-performing models
        for result in self.results:
            data = result['X'].copy()
            data['Label'] = result['y']
            self.setup_data(data)

            compare_models(sort="Accuracy", fold=NUMBER_OF_FOLDS, turbo=False, n_select=len(models()), exclude=["knn"])
            performance = pull().iloc[:, [0, 1]]  # [Model Abbreviation, Accuracy]

            for _, row in performance.iterrows():
                model_avg_acc[row[0]].append(row[1])

        top_5 = sorted(model_avg_acc.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]

        # Second pass: create tuned blended models with feature selection
        for result in self.results:
            file_name = result["file_name"]
            gene = file_name.replace("_properties_merged.csv", "")
            model_path = os.path.join(saved_shared_model_path, gene)

            data = result['X'].copy()
            data['Label'] = result['y']

            # Step 1: Create initial models to gather feature importance
            importance_df = pd.DataFrame()
            for model_name, _ in top_5:
                abbr = model_name_to_abbr.get(model_name)
                if abbr:
                    model = create_model(abbr, fold=NUMBER_OF_FOLDS, verbose=False)
                    if hasattr(model, "feature_importances_"):
                        feature_names = get_config("X").columns.tolist()
                        importances = model.feature_importances_
                        if len(importances) == len(feature_names):
                            temp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                            importance_df = pd.concat([importance_df, temp_df])
                else:
                    print(f"⚠️ No abbreviation found for model: {model_name}")

            if importance_df.empty:
                print(f"⚠️ Skipping {gene} — no feature importance available.")
                continue

            # Step 2: Average and select top features
            importance_df = importance_df.groupby("Feature").mean().reset_index()
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            num_features = max(5, int(len(importance_df) * FEATURE_PERCENTAGE))
            top_features = importance_df.iloc[:num_features]["Feature"].tolist()

            # Step 3: Refine data and rerun setup
            refined_data = result['X'][top_features].copy()
            refined_data['Label'] = result['y']
            self.setup_data(refined_data)

            # Step 4: Recreate, tune and blend top models on reduced features
            models_to_blend = []
            for model_name, _ in top_5:
                abbr = model_name_to_abbr.get(model_name)
                if abbr:
                    model = create_model(abbr, fold=NUMBER_OF_FOLDS, verbose=False)
                    tuned = tune_model(model, optimize="Accuracy", n_iter=NUMBER_OF_ITERATIONS, fold=NUMBER_OF_FOLDS,
                                    search_library="scikit-optimize", search_algorithm="bayesian",
                                    early_stopping=True, tuner_verbose=False)
                    models_to_blend.append(tuned)

            blended = blend_models(models_to_blend, fold=NUMBER_OF_FOLDS, choose_better=True, method="soft")
            score = pull()
            accuracy = float(score.iloc[NUMBER_OF_FOLDS, 1])

            save_model(blended, model_path)
            self.save_metrics(model_path + "shared_metrics.json", {"accuracy": accuracy, "top_features": importance_df.iloc[:num_features].to_json(orient="records")})