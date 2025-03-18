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

        for result in self.results:
          df = pd.read_csv(os.path.join("Processed_Data", "3_properties_merged", "ml_processed_properties", result["file_name"]))
          data = df.iloc[:, 2:]

        s = setup(data, target = 'Label', session_id = RANDOM_STATE)

        print(s)