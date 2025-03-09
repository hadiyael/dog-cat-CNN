import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# Load the pre-trained VGG model from the saved .keras file
model_vgg = tf.keras.models.load_model('cat_dog_final_model.keras')

# Define class names for the prediction (modify if necessary)
class_names = ["Cat", "Dog"]

# Streamlit UI
st.title("Cat vs. Dog Classifier 🐱🐶")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image (resize to match the model's input size)
    img = image.resize((400, 360))  # Resize to the expected input size (Width=400, Height=360)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess image as per VGG16 requirements

    # Check shape and value range of the image array
    st.write(f"Image array shape: {x.shape}")
    st.write(f"Image array range: {np.min(x)} to {np.max(x)}")

    # Predict the class of the image
    preds = model_vgg.predict(x)

    # Decode predictions to get the class label and the probability
    decoded_preds = decode_predictions(preds, top=1)[0]
    predicted_class = decoded_preds[0][1]
    confidence = decoded_preds[0][2]

    # Display the prediction result and confidence
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}")
