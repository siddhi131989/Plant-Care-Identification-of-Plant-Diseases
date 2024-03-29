import streamlit as st
import tensorflow as tf
import numpy as np
# Define paths to the model and label files
MODEL_PATH = "/Users/siddh/OneDrive/Desktop/4th yr project/model/keras_new_model.h5"
LABEL_PATH = "/Users/siddh/OneDrive/Desktop/4th yr project/model/labelss.txt"
# Load the Keras model and labels
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, 'r') as f:
        labels = f.read().splitlines()
    return model, labels
# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image
# Function to make predictions
def predict(image, model, labels):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_label, confidence
# Main function to run the Streamlit app
def main():
    st.title("Image Classification")

    # Load model and labels
    model, labels = load_model_and_labels()

    # Sidebar for file upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Perform prediction if file uploaded
    if uploaded_file is not None:
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        predicted_label, confidence = predict(preprocessed_image, model, labels)

        st.write("Prediction:", predicted_label)
        st.write("Confidence:", confidence)
# Run the main function
if __name__ == "__main__":
    main()
