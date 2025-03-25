from collections import defaultdict
import json
from PyQt5.QtCore import QThread, pyqtSignal
from scripting.MachineLearning import ml_file_cleaner as mlfc
import pandas as pd
from pycaret.classification import *
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker.*")

RANDOM_STATE = 42
NUMBER_OF_MODELS = 5
NUMBER_OF_FOLDS = 20
SAVED_MODEL_DIR = os.path.join("Processed_Data", "ml_saved_models")

# Get processed files
parent_dir = mlfc.MLFolderFinder().parent_dir
processed_folder = os.path.join(parent_dir, "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(processed_folder, exist_ok=True)
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]
if not csv_files:
    mlfc.MLFileCleaner.run_file_clean()
    csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

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

    def extract_data(self, processed_file):
        df = pd.read_csv(os.path.join(processed_folder, processed_file))
        X = df.iloc[:, 3:]
        y = df.iloc[:, 2]
        return {"X": X, "y": y, "file_name": processed_file}

    def setup_data(self, data):
        return setup(data, target='Label', session_id=RANDOM_STATE, verbose=False)

    def extract_top_features(self, models):
        importance_df = pd.DataFrame()
        for model in models:
            if hasattr(model, "feature_importances_"):
                features = get_config("X_train").columns
                importances = model.feature_importances_
                if len(importances) == len(features):
                    temp_df = pd.DataFrame({"Feature": features, "Importance": importances})
                    importance_df = pd.concat([importance_df, temp_df])
        if importance_df.empty:
            return []
        grouped = importance_df.groupby("Feature").mean().reset_index()
        top = grouped.sort_values(by="Importance", ascending=False)
        num_features = max(5, int(len(top) * 0.1))
        return top.iloc[:num_features]['Feature'].tolist()

    def tune_and_blend_models(self, models):
        tuned_models = [
            tune_model(m, optimize="Accuracy", n_iter=50, fold=NUMBER_OF_FOLDS,
                       search_library="scikit-optimize", search_algorithm="bayesian",
                       early_stopping=True, tuner_verbose=False) for m in models
        ]
        return blend_models(tuned_models, fold=NUMBER_OF_FOLDS)

    def save_metrics(self, path, metrics):
        with open(path, "w") as f:
            json.dump(metrics, f)

    def run(self):
        print("Running Machine Learning Models...")
        self.run_independent_models()
        self.run_shared_models()

    def run_independent_models(self):
        os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        self.accuracy_results = []

        for result in self.results:
            gene = result["file_name"].replace("_properties_merged.csv", "")
            model_path = os.path.join(SAVED_MODEL_DIR, gene)
            print(f"\nProcessing gene: {gene}")

            data = result['X'].copy()
            data['Label'] = result['y']
            self.setup_data(data)

            if os.path.exists(model_path + ".pkl"):
                metrics_path = model_path + "_independent_metrics.json"
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        accuracy = json.load(f).get("accuracy")
                        print(f"✔ Loaded model with stored accuracy: {accuracy}")
                        self.accuracy_results.append({"gene": gene, "accuracy": accuracy})
                continue

            best_models = compare_models(sort="Accuracy", fold=NUMBER_OF_FOLDS, n_select=NUMBER_OF_MODELS, turbo=False)
            selected_features = self.extract_top_features(best_models)
            if not selected_features:
                print(f"⚠️ No valid feature importances for {gene}. Skipping.")
                continue

            refined_data = result['X'][selected_features].copy()
            refined_data['Label'] = result['y']
            self.setup_data(refined_data)

            blended = self.tune_and_blend_models(best_models)
            score = pull()
            accuracy = float(score.iloc[NUMBER_OF_FOLDS, 1])

            save_model(blended, model_path)
            self.save_metrics(model_path + "_independent_metrics.json", {"accuracy": accuracy})
            self.accuracy_results.append({"gene": gene, "accuracy": accuracy})

        print(self.accuracy_results)

    def run_shared_models(self):
        model_avg_acc = defaultdict(list)

        for result in self.results:
            data = result['X'].copy()
            data['Label'] = result['y']
            self.setup_data(data)

            compare_models(sort="Accuracy", fold=NUMBER_OF_FOLDS, turbo=False, n_select=len(models()))
            performance = pull().iloc[:, [0, 1]]

            for _, row in performance.iterrows():
                model_avg_acc[row[0]].append(row[1])

        top_5 = sorted(model_avg_acc.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:5]

        for result in self.results:
            file_name = result["file_name"]
            gene = file_name.replace("_properties_merged.csv", "")
            model_path = os.path.join(SAVED_MODEL_DIR, gene)

            data = result['X'].copy()
            data['Label'] = result['y']

            self.setup_data(data)
            compare_models(sort="Accuracy", fold=NUMBER_OF_FOLDS, turbo=False, n_select=NUMBER_OF_MODELS)
            features = self.extract_top_features(compare_models(n_select=NUMBER_OF_MODELS))
            if not features:
                continue

            refined_data = result['X'][features].copy()
            refined_data['Label'] = result['y']
            self.setup_data(refined_data)

            models_to_blend = []
            for name, _ in top_5:
                abbr = model_name_to_abbr.get(name)
                if abbr:
                    model = create_model(abbr, fold=NUMBER_OF_FOLDS, verbose=False)
                    tuned = tune_model(model, optimize="Accuracy", n_iter=100, fold=NUMBER_OF_FOLDS,
                                       search_library="scikit-optimize", search_algorithm="bayesian",
                                       early_stopping=True, tuner_verbose=False)
                    models_to_blend.append(tuned)

            blend_models(models_to_blend, fold=NUMBER_OF_FOLDS, choose_better=True, method="soft")
            metrics = pull()
            self.save_metrics(model_path + "_shared_metrics.json", {gene: metrics.to_dict()})

        print("Shared models generated.")