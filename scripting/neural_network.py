import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scripting import ml_file_cleaner as mlfc

# Constants
RANDOM_STATE = 42
MODEL_FOLDER = os.path.join(os.getcwd(), "ml_models", "neural_net_models")
BINARY_CLASS_FOLDER = os.path.join(os.getcwd(), "Processed_Data", "3_properties_merged", "ml_processed_properties")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data(df):
    """Cleans data by removing NaN columns, low-variance features, and extreme values."""
    initial_cols = df.shape[1]

    # Drop NaN columns
    df = df.dropna(axis=1)

    # Convert all columns to numeric, coerce errors to NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Remove low-variance features
    df = df.loc[:, df.nunique() > 1]

    # Drop remaining NaNs
    df.dropna(inplace=True)

    # Standardize features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    final_cols = df.shape[1]
    logging.info(f"Data cleaned: Removed {initial_cols - final_cols} columns.")
    
    return df

def build_mlp_model(input_dim):
    """Builds an MLP model for binary classification."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=keras.regularizers.l2(0.02)),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def train_mlp_model(X_train, y_train, X_val, y_val, input_dim):
    """Trains an MLP model for classification."""
    model = build_mlp_model(input_dim)
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train, epochs=150, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model

def evaluate_and_save_model(model, X_test, y_test, gene_name):
    """Evaluates the model and saves it."""
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class (0 or 1)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"])
    logging.info(f"{gene_name} | Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Report:\n{report}")
    
    # Save model
    model_filename = os.path.join(MODEL_FOLDER, f"mlp_{gene_name}.h5")
    model.save(model_filename)
    logging.info(f"Model saved as {model_filename}")
    
    return {"gene": gene_name, "score": accuracy}

def plot_model_performance(results):
    """Plots accuracy for classification models."""
    if not results:
        logging.warning("No results available to plot.")
        return
    
    plt.figure(figsize=(10, 5))
    genes = [res["gene"] for res in results]
    accuracies = [res["score"] for res in results]
    plt.barh(genes, accuracies, color="skyblue")
    plt.xlabel("Accuracy")
    plt.ylabel("Gene")
    plt.title("MLP Classification Model Accuracy")
    plt.xlim(0, 1)  # Accuracy ranges from 0 to 1
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

def run_mlp():
    """Loads data, trains classification models, and saves them."""

    # Check if model files already exist
    existing_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith(".h5")]
    if existing_models:
        logging.info("Existing models found. Skipping model training to avoid unnecessary re-training.")
        exit()

    # Get all CSV files for classification
    classification_files = [f for f in os.listdir(BINARY_CLASS_FOLDER) if f.endswith(".csv")]

    if not classification_files:
        mlfc.MLFileCleaner.run_file_clean()
        classification_files = [f for f in os.listdir(BINARY_CLASS_FOLDER) if f.endswith(".csv")]

    results = []
    
    for csv_file in classification_files:
        gene_name = os.path.splitext(csv_file)[0]
        logging.info(f"Processing Classification: {gene_name}")
        
        df = pd.read_csv(os.path.join(BINARY_CLASS_FOLDER, csv_file))
        if "Label" not in df.columns:
            continue
        
        df = clean_data(df)
        X, y = df.iloc[:, 3:], LabelEncoder().fit_transform(df["Label"])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
        
        model = train_mlp_model(X_train, y_train, X_val, y_val, X.shape[1])
        results.append(evaluate_and_save_model(model, X_test, y_test, gene_name))
    
    return results