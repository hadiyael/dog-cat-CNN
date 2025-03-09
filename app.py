import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('cat_dog_final_model.keras')

st.title("Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = image.load_img(uploaded_file, target_size=(360, 400)) # Ensure target size matches training
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model_vgg.predict(x)
        predicted_class = "Dog" if preds[0][0] > 0.5 else "Cat"  # Adjust threshold if necessary

        st.image(img, caption=f"Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {predicted_class}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.write("Please upload an image.")
