import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('cat_dog_final_model.keras')

# Define a function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((400, 360))  # Resize to the correct shape (width, height)
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize the image (as done in training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
st.title("Cat vs Dog Classification")
st.write("Upload an image of a cat or a dog, and the model will predict which one it is!")

# Upload image
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded, display it and make predictions
if uploaded_image is not None:
    # Open the uploaded image and display it
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)
    
    # Display prediction result
    if prediction[0] > 0.5:
        st.write("Prediction: Dog 🐶")
    else:
        st.write("Prediction: Cat 🐱")

# Run the Streamlit app by saving the file and running: `streamlit run app.py`
