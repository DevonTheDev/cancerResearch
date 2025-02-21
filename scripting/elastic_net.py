import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Constants
FIXED_COLUMNS = ["Drug", "Pearson_Correlation"]
RANDOM_STATE = 42
FEATURE_CUTOFF = 0.0005  # Minimum coefficient threshold for feature selection
POLY_DEGREE = 2  # Initial polynomial degree, dynamically adjusted

# Directories
parent_dir = os.path.dirname(os.path.abspath(__file__))
processed_folder = os.path.join(os.path.dirname(parent_dir), "Processed_Data", "3_properties_merged")
output_folder = os.path.join(processed_folder, "processed_properties")
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

def setup_logging():
    """Sets up logging for the script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ElasticNetModelTrainer:
    def __init__(self, processed_file):
        self.processed_file = processed_file
        self.model = None
        self.selected_features = None
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=POLY_DEGREE, interaction_only=True, include_bias=False)
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_data()

    def load_data(self):
        """Loads dataset and performs train-test split."""
        df = pd.read_csv(self.processed_file)
        X = df.iloc[:, 3:]
        y = df.iloc[:, 2]
        return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    def preprocess_features(self):
        """Applies standard scaling and polynomial feature expansion."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        if self.X_train.shape[1] < 50:
            self.poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
        
        self.X_train = self.poly.fit_transform(self.X_train)
        self.X_test = self.poly.transform(self.X_test)

    def select_features(self):
        """Selects features based on ElasticNet importance."""
        initial_model = ElasticNet(random_state=RANDOM_STATE, max_iter=10000)
        initial_model.fit(self.X_train, self.y_train)
        selector = SelectFromModel(initial_model, threshold='mean', prefit=True)
        
        self.X_train = selector.transform(self.X_train)
        self.X_test = selector.transform(self.X_test)
        self.selected_features = np.array(self.poly.get_feature_names_out())[selector.get_support()]

    def train_model(self):
        """Performs hyperparameter tuning and trains the ElasticNet model."""
        param_dist = {
            "alpha": np.logspace(-4, 3, 50),
            "l1_ratio": np.linspace(0.05, 0.95, 10)
        }

        en_model = ElasticNet(random_state=RANDOM_STATE, max_iter=10000)
        random_search = RandomizedSearchCV(en_model, param_dist, n_iter=20, cv=5, scoring="r2", n_jobs=-1, random_state=RANDOM_STATE)
        random_search.fit(self.X_train, self.y_train)

        self.model = random_search.best_estimator_
        logging.info(f"Best Hyperparameters: {random_search.best_params_}")

    def evaluate_model(self):
        """Evaluates the model and returns performance metrics."""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        explained_var = explained_variance_score(self.y_test, y_pred)
        
        cv_r2 = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring="r2").mean()
        
        logging.info(f"Model MSE: {mse:.4f}, MAE: {mae:.4f}, R² Score: {r2:.4f}, Explained Variance: {explained_var:.4f}, Cross-Validated R²: {cv_r2:.4f}")
        return mse, r2

    def visualize_model_performance(self):
        """Visualizes model performance."""
        y_pred = self.model.predict(self.X_test)
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.5, color='red', edgecolors='black')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        
        plt.subplot(1, 2, 2)
        residuals = self.y_test - y_pred
        plt.hist(residuals, bins=20, color="green", edgecolor="black", alpha=0.7)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title("Residual Histogram")
        
        plt.tight_layout()
        plt.show()


def run_ml_pipeline():
    setup_logging()
    processed_files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".csv")]

    if not processed_files:
        logging.error("No processed files available. Exiting ML process.")
        return None

    for processed_file in processed_files:
        logging.info(f"Processing file: {processed_file}")
        trainer = ElasticNetModelTrainer(processed_file)
        trainer.preprocess_features()
        trainer.select_features()
        trainer.train_model()
        trainer.evaluate_model()
        trainer.visualize_model_performance()


def main():
    logging.info("Starting ElasticNet model training process...")
    run_ml_pipeline()

if __name__ == "__main__":
    main()