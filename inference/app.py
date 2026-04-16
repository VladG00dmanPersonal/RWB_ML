import time

import pandas as pd
import streamlit as st
from PIL import Image

from model_vlad import Model


@st.cache_resource
def load_model():
    return Model()



def predict(name, description, images):
    loaded_images = list(map(lambda img: Image.open(img).convert('RGB'), images))
    with st.spinner('Predicting...'):
        prediction = model.predict(name, description, loaded_images)
        output = pd.DataFrame({'image': list(map(lambda img: img.name, images)), 'relevancy': prediction})
        st.header('Prediction')
        st.dataframe(output)
        for i in range(50):
            st.balloons()
            time.sleep(0.1)

def render_form():
    st.title("Product image relevancy prediction")
    name = st.text_input("Product name")
    description = st.text_input("Product description")
    images = st.file_uploader("Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
    if st.button("Predict"):
        predict(name, description, images)


model = load_model()
render_form()