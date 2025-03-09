import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('cat_dog_final_model.keras')

# Define class names for the prediction
class_names = ["Cat", "Dog"]

# Streamlit UI
st.title("Cat vs. Dog Classifier 🐱🐶")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_resized = image.resize((360, 400))  # Resize to the expected input size
    image_array = np.array(image_resized) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class of the image
    prediction = model.predict(image_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display the prediction result and confidence
    st.write(f"Prediction: **{class_names[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}**")

