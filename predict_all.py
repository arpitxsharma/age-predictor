import cv2
import numpy as np
import tensorflow as tf

# --- CORRECTED: Import both model-specific preprocessing functions ---
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_skin
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_age

# --- 1. Configuration ---
# --- Model Paths ---
SKIN_MODEL_PATH = 'dermal_scan_model_best.h5'
AGE_MODEL_PATH = 'age_prediction_model_fast.h5'  # Using the fast model
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# --- Test Image Path ---
TEST_IMAGE_PATH = 'Arpit.jpg' # Make sure this image is in your folder

# --- Class Names ---
SKIN_CLASS_NAMES = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']

# --- 2. Load All Models ---
print("Loading all required models...")
try:
    skin_model = tf.keras.models.load_model(SKIN_MODEL_PATH)
    print("✓ Skin classification model loaded successfully.")
    
    age_model = tf.keras.models.load_model(AGE_MODEL_PATH)
    print("✓ Age prediction model loaded successfully.")

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Could not load Haar Cascade classifier.")
    print("✓ Haar Cascade face detector loaded successfully.")

except Exception as e:
    print(f"FATAL ERROR: Could not load one or more models. Details: {e}")
    exit()

# --- 3. Load and Prepare the Test Image ---
print(f"\nLoading test image from: {TEST_IMAGE_PATH}")
image = cv2.imread(TEST_IMAGE_PATH)
if image is None:
    print(f"FATAL ERROR: Could not read the image at path: {TEST_IMAGE_PATH}. Please check the path.")
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 4. Detect Faces in the Image ---
print("Detecting faces...")
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("No faces were detected in the image.")
else:
    print(f"Detected {len(faces)} face(s). Running predictions...")
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        # --- 5. CORRECTED: Preprocess separately for each model ---
        
        # a) Preprocess for the Skin Model (ResNetV2 - requires 224x224)
        resized_skin = cv2.resize(face_roi, (224, 224))
        img_array_skin = np.expand_dims(np.array(resized_skin, dtype='float32'), axis=0)
        preprocessed_img_skin = preprocess_input_skin(img_array_skin)

        # b) Preprocess for the Age Model (MobileNetV2 - requires 128x128)
        resized_age = cv2.resize(face_roi, (128, 128))
        img_array_age = np.expand_dims(np.array(resized_age, dtype='float32'), axis=0)
        preprocessed_img_age = preprocess_input_age(img_array_age)

        # --- 6. Make Predictions with Correctly Processed Data ---
        skin_predictions = skin_model.predict(preprocessed_img_skin)
        age_prediction = age_model.predict(preprocessed_img_age)

        # Process prediction results
        skin_class_index = np.argmax(skin_predictions[0])
        skin_class_name = SKIN_CLASS_NAMES[skin_class_index]
        skin_confidence = skin_predictions[0][skin_class_index] * 100
        predicted_age = int(age_prediction[0][0])

        # --- 7. Visualize the Results ---
        cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
        skin_text = f"{skin_class_name}: {skin_confidence:.1f}%"
        age_text = f"Predicted Age: {predicted_age}"
        cv2.putText(image, skin_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
        cv2.putText(image, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

# --- 8. Resize Final Image for Consistent Display ---
DISPLAY_WIDTH = 800
original_height, original_width = image.shape[:2]

if original_width > DISPLAY_WIDTH:
    ratio = DISPLAY_WIDTH / float(original_width)
    display_height = int(original_height * ratio)
    display_image = cv2.resize(image, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
    print(f"\nImage was too large, resized for display.")
else:
    display_image = image

# --- 9. Display the Final Annotated Image ---
cv2.imshow('DermalScan - Integrated Analysis', display_image)
print("\nPrediction complete. Displaying result image. Press any key to close the window.")
cv2.waitKey(0)
cv2.destroyAllWindows()