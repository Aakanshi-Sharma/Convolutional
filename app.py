import numpy as np
import pickle
import streamlit as st
import keras.utils as image

model = pickle.load(open("model.pkl", "rb"))


def classify_fun(text):
    test_image = image.load_img(text, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    st.title(result)
    if result[0][0] == 1:
        st.title("Dog")
    else:
        st.title("Cat")
    # st.image(test_image)


# UI
st.title("Classifier")
st.subheader("Check whether it is a dog or cat")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    st.image(uploaded_file, width=300)
    clicked = st.button("Predict")
    if clicked:
        classify_fun(uploaded_file)
