import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from keras.utils import img_to_array


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@st.cache(allow_output_mutation=True)
def load_model():
        model = tf.keras.models.load_model("my_model.h5")
        return model


def predict(image, model):
        image = Image.open(image)
        new_image = image.resize((256, 256))
        # st.image(image)
        # image = img_to_array(image)
        new_image = np.asarray(new_image)  # Since our model accepts inputs in the form of tensors
        img_batch = np.expand_dims(new_image, 0)  # Adding fourth dim as model accepts inputs in batch
        predictions = model.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # finds the index  of max prob amongst CLASSES
        confidence = np.max(predictions[0])

        return predicted_class, confidence


def first_predictor(image):
        # status = st.text("Loading model ...")
        model = load_model()
        # status.text("Loading model ... done!")
        # image = st.file_uploader("")
        if image is not None:
                st.image(image)
                # status.text("")
                button = st.button(label="Classify", key="classify_tab1")
                if button:
                        prediction, confidence = predict(image, model)
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Class", prediction)
                        col2.metric("Confidence", confidence)


def second_predictor(image):
        model = load_model()  # Loading our classifier

        if image is not None:
                st.image(image)

                button = st.button(label="Classify", key="classify_tab2")
                if button:
                        prediction, confidence = predict(image, model)
                        col1, col2 = st.columns(2)
                        col1.metric("Predicted Class", prediction)
                        col2.metric("Confidence", confidence)


def main():
        tab1, tab2 = st.tabs(["Upload", "Use Camera"])  # Creating separate tabs for each Classifiers
        with tab1:
                tab1_image = st.file_uploader("Upload an Image")
                first_predictor(tab1_image)
        with tab2:
                # cam_button = st.button("Open Camera")
                # if cam_button:
                #         st.write("Hello world")
                tab2_image = st.camera_input("Take a picture")
                second_predictor(tab2_image)


with st.sidebar:
        st.header("About")
        st.markdown('A CNN based potato plant classifier to detect the status of a plant in terms of its health')
        st.sidebar.caption("Diagnose your potato plant")
        st.markdown("Made by [Sandeep Kashyap](https://www.linkedin.com/in/siavash-yasini/)")
        st.header("Resources")
        st.markdown('''
        - [Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
        - [Streamlit](https://docs.streamlit.io/)
        ''')

        st.sidebar.header('Deploy')
        st.sidebar.markdown(
                'You can quickly deploy Streamlit apps using [Streamlit Community Cloud](https://streamlit.io/cloud) '
                'in just a few clicks.')


if __name__ == '__main__':
        main()