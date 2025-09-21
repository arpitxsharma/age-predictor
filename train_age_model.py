import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Switched to the faster MobileNetV2 model ---
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- Configuration for SPEED ---
DATASET_PATH = os.path.join('data', 'UTKFace')
IMG_SIZE = 128       # Smaller image size for faster processing
BATCH_SIZE = 64      # Larger batch size can speed up training steps
EPOCHS = 20          # Fewer epochs for a faster run
MAX_IMAGES = 5000    # Limit the dataset size for a very fast prototype

# --- 1. Load and Parse a Fraction of the UTKFace Dataset ---
print(f"Loading and parsing up to {MAX_IMAGES} images...")
images = []
age_labels = []

for i, filename in enumerate(os.listdir(DATASET_PATH)):
    if i >= MAX_IMAGES:
        print(f"Reached max image limit of {MAX_IMAGES}.")
        break
    try:
        age = int(filename.split('_')[0])
        if 1 <= age <= 100:
            img_path = os.path.join(DATASET_PATH, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            age_labels.append(age)
    except Exception as e:
        pass # Silently skip files

images = np.array(images, dtype='float32')
age_labels = np.array(age_labels, dtype='float32')
print(f"Dataset loaded: {len(images)} images")

# --- 2. Preprocess Data and Split ---
images_preprocessed = preprocess_input(images)
X_train, X_test, y_train, y_test = train_test_split(
    images_preprocessed, age_labels, test_size=0.2, random_state=42
)
print(f"Training data shape: {X_train.shape}")

# --- 3. Build the Lightweight Regression Model ---
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=inputs,
    pooling='avg'
)
base_model.trainable = False

x = Dropout(0.4)(base_model.output)
predictions = Dense(1, activation='linear', name='age_output')(x)

model = Model(inputs=inputs, outputs=predictions)

# --- 4. Compile the Model ---
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mean_absolute_error',
    metrics=['mae']
)
model.summary()

# --- 5. Train the Model (Simplified) ---
callbacks = [
    EarlyStopping(monitor='val_mae', patience=5, mode='min', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=2, mode='min', verbose=1)
]

print("\n--- Training the head of the age prediction model ---")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# --- 6. Evaluate the Model ---
print("\n--- Evaluating final model ---")
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"\nFinal Test MAE (Mean Absolute Error): {test_mae:.2f} years")

# --- 7. NEW: Save the Final Model ---
MODEL_SAVE_PATH = 'age_prediction_model_fast.h5'
print(f"\nSaving final model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("✓ Model saved successfully.")