import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class_names = ["Curved spine", "Dead", "Edema", "Normal", "Unhatched", "Yolk deformation"]

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    try:
        # Clear Keras session to free up resources
        tf.keras.backend.clear_session()

        # convert image to (224, 224)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

        # convert image to numpy array
        image_array = np.asarray(image)

        # normalize image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # set model input
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # make prediction
        pred = model.predict(data)
        
        # Ensure the prediction is a numpy array
        pred = np.array(pred)
        
        return pred

    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# Set TensorFlow config
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# set title
st.title('Zebrafish Malformations Classification')

# set header
st.header('Please upload a Zebrafish larvae image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
try:
    model = tf.keras.models.load_model("pythonProject/ResNet_BestSoFar.h5", compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# display image
if file is not None and model is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    pred = classify(image, model, class_names)

    if pred is not None and len(pred) > 0:
        sorted_indices = np.argsort(pred[0])[::-1]  # Sort indices in descending order based on pred[0]

        sorted_class_names = [class_names[i] for i in sorted_indices]
        sorted_pred = [pred[0][i] for i in sorted_indices]

        filtered_data = {"Class Name": [], "Prediction (%)": []}
        for class_name, prediction in zip(sorted_class_names, sorted_pred):
            if prediction > 0.5:
                filtered_data["Class Name"].append(class_name)
                filtered_data["Prediction (%)"].append(f"{prediction * 100:.2f}%")

        # Create a DataFrame to hold the filtered data
        filtered_df = pd.DataFrame(filtered_data)

        # Display the filtered DataFrame as a table with invisible borders
        st.write(filtered_df)
    else:
        st.error("Failed to classify the image.")
else:
    if model is None:
        st.error("Model is not loaded.")
