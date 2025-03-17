import os
import logging
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scripting.MachineLearning import ml_file_cleaner as mlfc

parent_dir = mlfc.MLFolderFinder().parent_dir
MODEL_FOLDER = os.path.join(parent_dir, "ml_models", "neural_net_models")
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def evaluate_model(model, data):

    X_test = data["X_test"]

    """Evaluates the model and saves it."""
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary class (0 or 1)
    
    return y_pred

def run_mlp(result):
    """Loads data, trains classification models, and saves them."""
    
    # Structure as per MLWorker.py function extract_data(self, processed_file)
    X_train = result["X_train"]
    X_val = result["X_val"]
    y_train = result["y_train"]
    y_val = result["y_val"]
    X = result["X"]
    
    """Trains an MLP model for classification."""
    model = build_mlp_model(X.shape[1])
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train, epochs=300, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model