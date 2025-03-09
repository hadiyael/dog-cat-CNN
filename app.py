import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('cat_dog_final_model.keras')

# Define a function to preprocess the uploaded image
def preprocess_image(img):
    img = img.convert("RGB")  # Convert image to RGB format (removes alpha channel if exists)
    img = img.resize((400, 360))  # Resize to the correct shape (width, height)
    img_array = np.array(img)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize the image (as done in training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app layout
# Header with Image
st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
    }
    .header-container img {
        max-width: 100%;
        height: auto;
    }
    </style>
    <div class="header-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcnEoopczYX_eLHV5kWIGUiiow7pKpxLW-bQ&s" alt="App Header">
    </div>
    """, unsafe_allow_html=True
)
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

    # Determine the class based on the prediction probability
    if prediction[0] > 0.5:
        st.write("Prediction: Dog 🐶")
    else:
        st.write("Prediction: Cat 🐱")

    # Display the probability (confidence) of the prediction
    st.write(f"Prediction Probability: {prediction[0][0]*100:.2f}%")
