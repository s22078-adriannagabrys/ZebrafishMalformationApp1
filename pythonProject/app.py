import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import wikipedia

# Define class names
class_names = ["Curved spine", "Dead", "Edema", "Normal", "Unhatched", "Yolk deformation"]

def classify(image, model, target_size):
    """
    This function takes an image, a model, and a target size and returns the predicted class and confidence
    score of the image.
    """
    # Convert image to target size
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    # Convert image to numpy array
    image_array = np.asarray(image)
    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Set model input
    data = np.ndarray(shape=(1, *target_size, 3), dtype=np.float32)
    data[0] = normalized_image_array
    # Make prediction
    pred = model.predict(data)
    return pred

def display_predictions(pred, model_name):
    pred = np.array(pred)
    sorted_indices = np.argsort(pred[0])[::-1]  # Sort indices in descending order based on pred[0]
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_pred = [pred[0][i] for i in sorted_indices]

    st.write(f"#### Model {model_name}:")

    cols = st.columns(2)
    cols[0].write("**Klasa:**")
    cols[1].write("**Prawdopodobieństwo:**")

    for class_name, prediction in zip(sorted_class_names, sorted_pred):
        if prediction > 0.5:
            cols = st.columns(2)
            cols[0].write(f"{class_name}")
            cols[1].write(f"{prediction * 100:.2f}%")

def main():
    st.set_page_config(page_title="Klasyfikacja wad rozwojowych Danio rerio")
    overview = st.container()
    prediction = st.container()

     # Wikipedia search in sidebar
    st.sidebar.title("Wyszukiwanie w Wikipedii")
    search_query = st.sidebar.text_input("Wpisz zapytanie")

    if search_query:
        try:
            st.sidebar.write(wikipedia.summary(search_query))
        except wikipedia.exceptions.DisambiguationError as e:
            st.sidebar.error(f"Podane zapytanie jest dwuznaczne. Proszę wybrać bardziej konkretny termin.")
        except wikipedia.exceptions.PageError as e:
            st.sidebar.error(f"Nie znaleziono strony na Wikipedii dla podanego zapytania.")

    with overview:
        # Set title
        st.title('Klasyfikacja wad rozwojowych Danio rerio')

        # Upload file with a label
        file = st.file_uploader('Wgraj zdjęcie larwy Danio rerio', type=['jpeg', 'jpg', 'png'])

        # Display image and classify
        if file is not None:
            image = Image.open(file).convert('RGB')
            new_image = image.resize((400, 400))
            st.image(new_image)

            with prediction:
                if st.button("Klasyfikuj"):
                    st.divider()

                    # Load classifiers
                    modelResNet = tf.keras.models.load_model("pythonProject/ResNet_FT_BestSoFar.h5", compile=False)
                    modelInception = tf.keras.models.load_model("pythonProject/Inception_FT_BestSoFar2.h5", compile=False)
                    cols = st.columns(2)  # Add a small column in between for the vertical line

                    with cols[0]:
                        # Classify image with ResNet
                        predResNet = classify(image, modelResNet, (224, 224))
                        display_predictions(predResNet, "ResNet")

                    with cols[1]:
                        # Classify image with Inception
                        predInception = classify(image, modelInception, (299, 299))
                        display_predictions(predInception, "Inception")

if __name__ == "__main__":
    main()
