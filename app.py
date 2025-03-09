import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import gdown


@st.cache_resource
def load_model():
    url = "https://drive.google.com/1mJCFadc75OkuX5YfM_dbE7lqsWHgEJ7Z"
    output = "cat_dog_final_model.keras"
    gdown.download(url, output, quiet=False)
    
    model = tf.keras.models.load_model(output)
    return model

# Charger le modèle une seule fois
model = load_model()

st.write("Modèle chargé avec succès !")
