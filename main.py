import os
import cv2
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- 1. Configuration ---
# --- IMPORTANT: Set this to True for a quick test, or False for the full training ---
USE_FRACTION_OF_DATA = True 
FRACTION_TO_USE = 0.25 # Use 25% of the data if the above is True
# ------------------------------------------------------------------------------------

DATASET_PATH = os.path.join('data', 'UTKFace')
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# --- 2. Scan Dataset and Get All File Paths and Labels ---
print("Scanning dataset to get all file paths and labels...")
image_paths = []
age_labels = []

for filename in os.listdir(DATASET_PATH):
    try:
        age = int(filename.split('_')[0])
        image_paths.append(os.path.join(DATASET_PATH, filename))
        age_labels.append(age)
    except Exception as e:
        # This handles cases where a file in the folder is not a valid image
        print(f"Skipping file {filename}, cannot parse age.")

print(f"Found {len(image_paths)} total images.")

# --- 3. (Optional) Reduce Dataset Size Based on the Flag ---
if USE_FRACTION_OF_DATA:
    print("\n--- WARNING: Reducing dataset size for fast prototyping ---")
    num_samples = int(len(image_paths) * FRACTION_TO_USE)
    # Make sure we shuffle before taking a fraction to get a representative sample
    p = np.random.permutation(len(image_paths))
    image_paths = [image_paths[i] for i in p]
    age_labels = [age_labels[i] for i in p]
    image_paths = image_paths[:num_samples]
    age_labels = age_labels[:num_samples]
    print(f"--- Now using only {num_samples} images ---")


# --- 4. Split Data into Train, Validation, and Test Sets ---
print("\nSplitting data into train, validation, and test sets...")
paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
    image_paths, age_labels, test_size=0.2, random_state=42
)
paths_train, paths_val, labels_train, labels_val = train_test_split(
    paths_train_val, labels_train_val, test_size=0.2, random_state=42
)
print(f"Training samples: {len(paths_train)}, Validation: {len(paths_val)}, Test: {len(paths_test)}")


# --- 5. Custom Data Generator (Memory Efficient) ---
class AgeDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, dim, shuffle=True):
        self.image_paths = image_paths
        self.labels = np.array(labels, dtype='float32')
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[k] for k in batch_indexes]
        batch_labels = self.labels[batch_indexes]
        X = self.__data_generation(batch_paths)
        return X, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths):
        X = np.empty((len(batch_paths), *self.dim, 3), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.dim)
            X[i,] = img
        return preprocess_input(X)

# Instantiate the generators
train_generator = AgeDataGenerator(paths_train, labels_train, BATCH_SIZE, dim=(IMG_SIZE, IMG_SIZE))
val_generator = AgeDataGenerator(paths_val, labels_val, BATCH_SIZE, dim=(IMG_SIZE, IMG_SIZE))
test_generator = AgeDataGenerator(paths_test, labels_test, BATCH_SIZE, dim=(IMG_SIZE, IMG_SIZE), shuffle=False)


# --- 6. Build and Compile the Regression Model ---
print("\nBuilding MobileNetV2 model...")
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='linear', name='age_output')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_absolute_error', metrics=['mae'])
model.summary()


# --- 7. Train the Model ---
callbacks = [
    ModelCheckpoint('age_prediction_model_mobilenet.h5', monitor='val_mae', save_best_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_mae', patience=10, mode='min', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_mae', factor=0.2, patience=3, mode='min', verbose=1)
]

print("\n--- Starting Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# --- 8. Evaluate the Final Model ---
print("\n--- Evaluating final model on the test set ---")
model.load_weights('age_prediction_model_mobilenet.h5')
test_loss, test_mae = model.evaluate(test_generator, verbose=1)

print(f"\nFinal Test MAE (Mean Absolute Error): {test_mae:.2f} years")
print("This means the model's age predictions are, on average, off by about {:.2f} years.".format(test_mae))
print("\nAge prediction model saved as age_prediction_model_mobilenet.h5")