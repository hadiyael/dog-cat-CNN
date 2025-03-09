import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('cat_dog_final_model.keras')

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((400, 360))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
    }
    .header-container img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .prediction-container {
        text-align: center;
        margin-top: 20px;
    }
    .prediction-container h2 {
        color: #4CAF50;
    }
    .probability-container {
        text-align: center;
        margin-top: 10px;
        font-size: 1.2em;
    }
    .prediction-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        font-size: 1.1em;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .prediction-btn:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcnEoopczYX_eLHV5kWIGUiiow7pKpxLW-bQ&s" alt="App Header">
    </div>
    """, unsafe_allow_html=True
)

st.title("Cat vs Dog Classification")
st.write("Upload an image of a cat or a dog, and the model will predict which one it is!")

st.sidebar.title("Model Information")
st.sidebar.write("""
This model was trained using transfer learning with **VGG16** to classify images as either a **Cat** or a **Dog**. It uses pre-trained weights from the ImageNet dataset for better performance.
""")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(img)

    prediction = model.predict(img_array)

    st.markdown("<div class='prediction-container'><h2>Prediction:</h2></div>", unsafe_allow_html=True)
    
    if prediction[0] > 0.5:
        st.markdown("<div class='prediction-container'><h2>Dog 🐶</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='prediction-container'><h2>Cat 🐱</h2></div>", unsafe_allow_html=True)

    st.markdown(f"<div class='probability-container'>Prediction Probability: {prediction[0][0]*100:.2f}%</div>", unsafe_allow_html=True)

    if st.button('Upload another image', key='reset'):
        st.experimental_rerun()
