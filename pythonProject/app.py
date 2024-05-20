import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import pandas as pd
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
        A tuple of the predicted class name and the confidence score for that prediction.
    """
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

    st.write(pred)

    return pred


# set title
st.title('Zebrafish malformations classification')

# set header
st.header('Please upload a Zebrafish larvae image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = tf.keras.models.load_model("pythonProject/ResNet_BestSoFar.h5", compile=False)

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    pred = classify(image, model, class_names)

    pred = np.array(pred)

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
