import os
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Fix: Add models directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.unet import unet_model  # Import UNet model

# Define dataset paths
PROCESSED_DATA_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/data/processed_numpy/"
MODEL_PATH = "/home/debjit/Programming/ML/BrainTumorSegmentation/models/unet_brain_segmentation.h5"

# Load dataset (batch processing to prevent memory issues)
def load_data(set_name):
    files = sorted([f for f in os.listdir(PROCESSED_DATA_PATH) if f.startswith(f"X_{set_name}_")])
    
    X, y = [], []
    for file in files:
        X.append(np.load(os.path.join(PROCESSED_DATA_PATH, file)))
        y_file = file.replace("X_", "y_")
        y.append(np.load(os.path.join(PROCESSED_DATA_PATH, y_file)))

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

print("📥 Loading training data...")
X_train, y_train = load_data("train")
print(f"✅ Loaded Training Set: {X_train.shape}")

print("📥 Loading validation data...")
X_val, y_val = load_data("val")
print(f"✅ Loaded Validation Set: {X_val.shape}")

# Build UNet model
model = unet_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
EPOCHS = 20
BATCH_SIZE = 4  # Reduce batch size to avoid memory overflow

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save the trained model
model.save(MODEL_PATH)
print(f"✅ Model trained and saved at {MODEL_PATH}")
