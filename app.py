"""
app.py
------
Streamlit web application for the Deer vs Antelope binary image classifier.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on Streamlit Community Cloud
    (share.streamlit.io) by pointing it at this file.
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
MODEL_PATH = "model/deer_antelope_model.h5"
CLASS_INDICES_PATH = "model/class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Deer vs Antelope Classifier", page_icon="🦌", layout="centered")


# ---------------------------------------------------------------------
# LOAD MODEL (cached so it only loads once per session)
# ---------------------------------------------------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)

    # Map model output (0/1) back to class name
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)  # e.g. {"antelope": 0, "deer": 1}
    else:
        # Fallback if the json wasn't saved during training
        class_indices = {"antelope": 0, "deer": 1}

    # Invert so we can look up by predicted index
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🦌 Deer vs Antelope Classifier")
st.write(
    "Upload an image and this app will predict whether it shows a **Deer** "
    "or an **Antelope**, using a CNN (MobileNetV2 transfer learning model)."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Classify"):
        with st.spinner("Analyzing image..."):
            try:
                model, idx_to_class = load_model_and_labels()
                processed = preprocess_image(image)
                prediction = model.predict(processed)[0][0]  # sigmoid output, 0-1

                # prediction close to 1 -> class index 1 ; close to 0 -> class index 0
                predicted_idx = 1 if prediction >= 0.5 else 0
                predicted_label = idx_to_class[predicted_idx].capitalize()
                confidence = prediction if predicted_idx == 1 else 1 - prediction

                st.success(f"Prediction: **{predicted_label}**")
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.progress(float(confidence))

            except Exception as e:
                st.error(f"Something went wrong during prediction: {e}")
else:
    st.info("Please upload an image file to get a prediction.")

st.markdown("---")
st.caption(
    "Model: MobileNetV2 transfer learning | Binary classification: Deer vs Antelope | "
    "Built with TensorFlow/Keras and Streamlit."
)
