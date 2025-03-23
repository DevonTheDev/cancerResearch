from collections import defaultdict
import json
from sklearn.discriminant_analysis import StandardScaler
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.model_selection import train_test_split
from scripting.MachineLearning import ml_file_cleaner as mlfc
import pandas as pd
from pycaret.classification import *
import os

RANDOM_STATE = 42
NUMBER_OF_MODELS = 5

parent_dir = mlfc.MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)

# Get all CSV files in the directory
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# Only create cleaned files if none exist currently
if not (csv_files):
    mlfc.MLFileCleaner.run_file_clean()
    csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

# Mapping from full model name to PyCaret abbreviation
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
        self.results = []

        for processed_file in csv_files:
            self.extract_data(processed_file)

    def extract_data(self, processed_file):
        print(f"Processing file: {processed_file}")

        df = pd.read_csv(os.path.join(processed_folder, processed_file))

        X = df.iloc[:, 3:]
        y = df.iloc[:, 2]

        self.results.append({
            "X": X,
            "y": y,
            "file_name": processed_file
        })

    def run(self):
        print("Running Machine Learning Models...")
        self.run_independent_models()
        self.run_shared_models()

    def run_independent_models(self):
        saved_model_dir = os.path.join("Processed_Data", "ml_saved_models")
        os.makedirs(saved_model_dir, exist_ok=True)

        print("Yes" if os.path.exists(saved_model_dir) else "No")

        self.accuracy_results = []

        for result in self.results:
            file_name = result["file_name"]
            gene_name = file_name.replace("_properties_merged.csv", "")
            model_path = os.path.join(saved_model_dir, gene_name)

            print(f"\nProcessing gene: {gene_name}")

            data = result['X']
            data['Label'] = result['y']

            # Run setup to match PyCaret requirements before loading
            setup(data, target='Label', session_id=RANDOM_STATE)

            if os.path.exists(model_path + ".pkl"):
                metrics_path = model_path + "_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                        mean_accuracy = metrics.get("accuracy")
                        print(f"✔ Loaded model with stored accuracy: {mean_accuracy}")

                self.accuracy_results.append({"gene": gene_name, "accuracy": mean_accuracy})
                continue

            # Compare and get top 5 models
            best_models = compare_models(sort="Accuracy", fold=10, n_select=5)

            # Get average feature importance from top models
            importance_df = pd.DataFrame()
            for model in best_models:
                if hasattr(model, "feature_importances_"):
                    feature_names = get_config("X_train").columns.tolist()
                    importances = model.feature_importances_
                    if len(importances) == len(feature_names):
                        temp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                        importance_df = pd.concat([importance_df, temp_df])

            if importance_df.empty:
                print(f"⚠️ No models with valid feature importances for {gene_name}. Skipping.")
                continue

            importance_df = importance_df.groupby("Feature").mean().reset_index()
            top_features = importance_df.sort_values(by="Importance", ascending=False)
            num_features = max(5, int(len(top_features) * 0.1))
            selected_features = top_features.iloc[:num_features]['Feature'].tolist()

            # Retrain on selected features
            refined_data = result['X'][selected_features].copy()
            refined_data['Label'] = result['y']

            setup(refined_data, target='Label', remove_multicollinearity=True, 
                  multicollinearity_threshold=0.85, session_id=RANDOM_STATE)

            tuned_models = []
            for model in best_models:
                tuned = tune_model(
                    model, optimize="Accuracy", n_iter=10, fold=10,
                    search_library="scikit-optimize", search_algorithm="bayesian",
                    early_stopping=True
                )
                tuned_models.append(tuned)

            # Blend and save
            blended = blend_models(tuned_models, fold=10, method="soft")

            score = pull()
            mean_accuracy = float(score.iloc[10, 1]) # Extract mean accuracy from df from pull()

            save_model(blended, model_path)

            metrics_path = model_path + "_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"accuracy": mean_accuracy if mean_accuracy else "failed to extract"}, f)

            self.accuracy_results.append({"gene": gene_name, "accuracy": mean_accuracy})

        print(self.accuracy_results)

    def run_shared_models(self):

        model_average_accuracies = defaultdict(list)

        # First loop to determine best models across all genes
        for result in self.results:
            # Run all models and append average accuracy to a dict object
            data = result['X']
            data['Label'] = result['y']
            
            setup(data, target="Label", session_id=RANDOM_STATE, verbose=False)
            compare_models(sort="Accuracy", fold=10, n_select=13)

            performance_df = pull()
            performance_df.iloc[:, [1,2]] # Extract [Model Abbreviation, Accuracy]

            for _, item in performance_df.iterrows():
                abbr = item[0]
                acc = item[1]
                model_average_accuracies[abbr].append(acc)

        top_5_average_models = sorted(model_average_accuracies.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]
        print(top_5_average_models)

        # Second loop to create top models for every gene
        blended_models = {}

        for result in self.results:
            created_models = []

            data = result['X']
            data['Label'] = result['y']

            setup(data, target="Label", session_id=RANDOM_STATE, verbose=False)

            for model_name, _ in top_5_average_models:
                abbr = model_name_to_abbr.get(model_name)
                if abbr:
                    created_model = create_model(abbr, fold=10)
                    tuned_model = tune_model(created_model, fold=10, n_iter=10, early_stopping=True)
                    created_models.append(tuned_model)
                else:
                    print("Error - No Model Abbreviation Found")
            blend_models(created_models, fold=10, choose_better=True)
            blended_models[result["file_name"]] = pull()
        
        print(blended_models)