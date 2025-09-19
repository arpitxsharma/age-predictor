import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import os
import csv

# --- MODEL-SPECIFIC PREPROCESSORS ---
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_skin
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_age

# --- CONFIGURATION ---
SKIN_CLASS_NAMES = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']
LOG_FILE = 'prediction_log.csv'

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """Load all AI models and the face detector from disk."""
    try:
        skin_model = tf.keras.models.load_model('dermal_scan_model_best.h5')
        # --- CHANGE 1: Load the correct 'fast' age model file ---
        age_model = tf.keras.models.load_model('age_prediction_model_fast.h5')
        # --- END CHANGE 1 ---
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        return skin_model, age_model, face_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure all model files are in the same folder as app.py.")
        return None, None, None

skin_model, age_model, face_cascade = load_models()

# --- LOGGING FUNCTION ---
def log_prediction(filename, box, skin_class, skin_conf, age):
    """Saves prediction details to a CSV file."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'filename', 'box_x', 'box_y', 'box_w', 'box_h', 'predicted_skin_condition', 'skin_confidence', 'predicted_age'])
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        x, y, w, h = box
        writer.writerow([timestamp, filename, x, y, w, h, skin_class, f"{skin_conf:.2f}%", age])

# --- MODULARIZED INFERENCE PIPELINE ---
def run_analysis(image_cv, filename):
    """Takes an image, performs the full analysis, and returns the annotated image."""
    annotated_image = image_cv.copy()
    gray_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) == 0:
        st.warning("No faces were detected in the uploaded image.")
        return image_cv

    st.success(f"Detected {len(faces)} face(s). Processing...")
    
    for face_box in faces:
        x, y, w, h = face_box
        face_roi = annotated_image[y:y+h, x:x+w]
        
        # Preprocess for Skin Model (requires 224x224)
        resized_skin = cv2.resize(face_roi, (224, 224))
        img_skin = np.expand_dims(np.array(resized_skin, dtype='float32'), axis=0)
        preprocessed_skin = preprocess_input_skin(img_skin)
        
        # --- CHANGE 2: Preprocess for the FAST Age Model (requires 128x128) ---
        resized_age = cv2.resize(face_roi, (128, 128))
        # --- END CHANGE 2 ---
        img_age = np.expand_dims(np.array(resized_age, dtype='float32'), axis=0)
        preprocessed_age = preprocess_input_age(img_age)

        # Make predictions
        skin_predictions = skin_model.predict(preprocessed_skin)[0]
        skin_class_index = np.argmax(skin_predictions)
        skin_class_name = SKIN_CLASS_NAMES[skin_class_index]
        skin_confidence = skin_predictions[skin_class_index] * 100
        
        age_prediction = age_model.predict(preprocessed_age)[0][0]
        predicted_age = int(age_prediction)
        
        log_prediction(filename, face_box, skin_class_name, skin_confidence, predicted_age)

        # Draw annotations
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (36, 255, 12), 2)
        skin_text = f"{skin_class_name}: {skin_confidence:.1f}%"
        age_text = f"Predicted Age: ~{predicted_age}"
        cv2.putText(annotated_image, skin_text, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
        cv2.putText(annotated_image, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
        
    return annotated_image

# --- WEB UI ---
st.set_page_config(layout="wide", page_title="AI DermalScan")
st.title("🔬 AI DermalScan: Facial Skin & Age Analysis")
st.write("This application uses deep learning to analyze facial images for skin conditions and predict age. Upload an image to begin.")

uploaded_file = st.file_uploader("Choose a facial image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image', type="primary"):
        if skin_model is not None and age_model is not None and face_cascade is not None:
            with st.spinner('Performing analysis... Please wait.'):
                annotated_image = run_analysis(image_cv, uploaded_file.name)
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                st.write("---")
                st.subheader("Analysis Results")
                st.image(annotated_image_rgb, caption='Processed Image with Predictions', use_column_width=True)
                
                _, buffer = cv2.imencode('.jpg', annotated_image)
                image_bytes = buffer.tobytes()

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=image_bytes,
                        file_name=f"result_{uploaded_file.name}",
                        mime="image/jpeg"
                    )
                if os.path.exists(LOG_FILE):
                    with col2:
                        with open(LOG_FILE, "r") as f:
                            st.download_button(
                                label="📋 Download Prediction Log (CSV)",
                                data=f.read(),
                                file_name="prediction_log.csv",
                                mime="text/csv"
                            )
else:
    st.info('Please upload an image to get started.')