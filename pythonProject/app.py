import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

        # Get the feature map
        last_conv_layer = model.get_layer('conv5_block3_out')
        grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(data)
            loss = predictions[:, np.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        return class_names[np.argmax(pred)], np.max(pred), conv_outputs, pooled_grads
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None, None, None

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
    predicted_class, confidence, conv_outputs, pooled_grads = classify(image, model, class_names)

    if predicted_class is not None and confidence is not None:
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence}")

        # Grad-CAM
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap[0]

        # Resize heatmap to image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((image.width, image.height))

        # Apply heatmap to image
        heatmap = np.array(heatmap)
        image_array = np.array(image)
        superimposed_img = heatmap * 0.4 + image_array

        # Display Grad-CAM
        fig, ax = plt.subplots()
        ax.imshow(superimposed_img)
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.error("Classification failed.")
else:
    if model is None:
        st.error("Model is not loaded.")
