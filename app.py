import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image

# Load your model and labels
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Function to process frames for live camera feed
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Resize the image to match the model input size
    resized = cv2.resize(img, (224, 224))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=0)

    # Model prediction
    prediction = model.predict(reshaped)
    predicted_class = labels[np.argmax(prediction)]

    # Draw predicted label on the image
    cv2.putText(img, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI setup
st.title(" Anomaly Detection System")
st.markdown("##  Live Camera Feed  for Anomaly Detection")

# Option to upload an image
uploaded_file = st.file_uploader("Upload  an image for anomaly detection", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_array = np.array(image)
    resized = cv2.resize(img_array, (224, 224))
    normalized = resized / 255.0
    reshaped = np.expand_dims(normalized, axis=0)

    # Predict using the model
    prediction = model.predict(reshaped)
    predicted_class = labels[np.argmax(prediction)]

    st.write(f"Prediction : {predicted_class}")

# Display live webcam stream if user chooses
st.markdown("### Live Camera Stream  ")

webrtc_streamer(
    key="live-anomaly-detection",
    video_frame_callback=video_frame_callback
)

# Option to show extra features or explanations
st.sidebar.title("Options")
st.sidebar.markdown("You can choose between **live camera feed** or **uploading an image**.")
