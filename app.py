import numpy as np
import streamlit as st
from keras.preprocessing import image


def classify_fun(text):
    st.write(text.name)
    # st.image(text)


# UI

st.title("Classifier")

st.subheader("Check whether it is a dog or cat")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    clicked = st.button("Predict")
    if clicked:
        classify_fun(uploaded_file)
