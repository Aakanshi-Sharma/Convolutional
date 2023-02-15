import numpy as np
import time
import pickle
import streamlit as st
import keras.utils as image

model = pickle.load(open("model.pkl", "rb"))


def classify_fun(text):
    test_image = image.load_img(text, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = "Dog"
    else:
        prediction = "Cat"
    with st.spinner('Wait for it...'):
        time.sleep(1)

    st.title(prediction)


# UI
st.title("Cats vs Dogs Classification")
st.subheader("Check whether it is a dog or cat")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.image(uploaded_file, width=300)
    clicked = st.button("Predict")
    if clicked:
        classify_fun(uploaded_file)
