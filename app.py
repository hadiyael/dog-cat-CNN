import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

model = tf.keras.models.load_model('cat_dog_final_model.keras')

# Define class names
class_names = ["Cat", "Dog"]

# Streamlit UI
st.title("Cat vs. Dog Classifier 🐱🐶")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    image = image.resize((360,400))  
    image_array = np.array(image) / 255.0  
    image_array = np.expand_dims(image_array, axis=0) 

    # Predict
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display Result
    st.write(f"Prediction: **{class_names[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
