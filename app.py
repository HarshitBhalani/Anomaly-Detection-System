import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Title and description
st.set_page_config(page_title="Anomaly Detection System", layout="wide")
st.title('🔍 Anomaly Detection System for plate')

st.write("""
This system can detect whether a plate is **Normal** or **Defective** 
based on the uploaded image or live camera feed.
""")

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names
classes = ['Normal', 'Defective']  # Same order as in Teachable Machine

# Prediction function
def predict_image(img):
    img = img.resize((224, 224))    # Resize to match the model input
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32) / 255.0   # Normalize
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    return prediction

def predict_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    frame = np.array(frame, dtype=np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.squeeze(output_data)
    return prediction

# Sidebar options
option = st.sidebar.radio(
    "Select Input Method:",
    ('Upload an Image', 'Real-Time Camera Detection')
)

# Image Upload Interface
if option == 'Upload an Image':
    uploaded_file = st.file_uploader("Upload an image of the Plate...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        prediction = predict_image(image)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
        st.markdown(f"### 🔥 Confidence: **{confidence:.2f}%**")

# Real-Time Camera Detection
elif option == 'Real-Time Camera Detection':
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error('Failed to access camera')
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = predict_frame(frame_rgb)
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Put Prediction Text
        annotated_frame = frame_rgb.copy()
        cv2.putText(annotated_frame, f'{predicted_class} ({confidence:.2f}%)',
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        FRAME_WINDOW.image(annotated_frame)
    
    else:
        st.write('Camera stopped.')

