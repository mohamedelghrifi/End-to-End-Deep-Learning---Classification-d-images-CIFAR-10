import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Charger le modèle
model = tf.keras.models.load_model("saved_model/cifar10_model.h5")
class_names = ['Avion', 'Auto', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion']

st.title("Classification d'images CIFAR-10")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Image importée', use_column_width=True)
    img_array = np.array(image) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    st.write("Classe prédite :", class_names[np.argmax(prediction)])