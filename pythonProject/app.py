import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf

class_names = ["Curved spine", "Dead", "Edema", "Normal", "Unhatched", "Yolk deformation"]

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
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
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        numpy.ndarray: The prediction array.
    """
    # Convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    pred = model.predict(data)

    return pred

# Set title
st.title('Zebrafish Malformations Classification')

# Set header
st.header('Please upload a Zebrafish larvae image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
try:
    model = tf.keras.models.load_model("pythonProject/ResNet_BestSoFar.h5", compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Display image and classify if a file is uploaded and model is loaded
if file is not None and model is not None:
    try:
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        # Classify image
        pred = classify(image, model, class_names)

        # Convert prediction to numpy array
        pred = np.array(pred)

        # Display prediction results
        st.write("Prediction Results:", pred)
    except Exception as e:
        st.error(f"Error during classification: {e}")
else:
    if model is None:
        st.error("Model is not loaded.")
    else:
        st.info("Please upload an image file.")
