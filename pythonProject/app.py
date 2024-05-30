import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
from matplotlib import cm

# Define class names
class_names = ["Curved spine", "Dead", "Edema", "Normal", "Unhatched", "Yolk deformation"]

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.
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
    return pred, data

def get_img_array(img_path, size):
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array.astype(np.float32) / 127.5) - 1
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_indices):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        class_channel = tf.gather(preds[0], pred_indices)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmaps = []
    for i in range(len(pred_indices)):
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmaps.append(heatmap.numpy())
    return heatmaps

def save_and_display_gradcam(img_path, heatmaps, pred_indices, alpha=0.3):
    img = Image.open(img_path).convert('RGB')
    img = np.asarray(img)
    for i, heatmap in enumerate(heatmaps):
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = np.asarray(jet_heatmap)
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = Image.fromarray(np.uint8(superimposed_img))
        st.image(superimposed_img, caption=f"Grad-CAM for {class_names[pred_indices[i]]}", use_column_width=True)

# Set title
st.title('Zebrafish Malformations Classification')

# Set header
st.header('Please upload a Zebrafish larvae image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = tf.keras.models.load_model("pythonProject/ResNet_BestSoFar.h5", compile=False)

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    pred, img_array = classify(image, model, class_names)
    pred = np.array(pred)
    st.write(pred)
    sorted_indices = np.argsort(pred[0])[::-1]  # Sort indices in descending order based on pred[0]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_pred = [pred[0][i] for i in sorted_indices]

    st.write("### Prediction Results:")
    pred_indices = []
    for class_name, prediction in zip(sorted_class_names, sorted_pred):
        if prediction > 0.5:
            st.write(f"**{class_name}:** {prediction * 100:.2f}%")
            pred_indices.append(class_names.index(class_name))
