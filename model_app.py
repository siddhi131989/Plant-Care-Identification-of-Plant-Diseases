{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9b33a5cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "671d60e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define paths to the model and label files\n",
    "MODEL_PATH = \"/Users/siddh/OneDrive/Desktop/4th yr project/model/keras_new_model.h5\"\n",
    "LABEL_PATH = \"/Users/siddh/OneDrive/Desktop/4th yr project/model/labelss.txt\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "02ec799d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the Keras model and labels\n",
    "def load_model_and_labels():\n",
    "    model = tf.keras.models.load_model(MODEL_PATH)\n",
    "    with open(LABEL_PATH, 'r') as f:\n",
    "        labels = f.read().splitlines()\n",
    "    return model, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a99ae216",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to preprocess the image\n",
    "def preprocess_image(image):\n",
    "    image = tf.image.resize(image, (224, 224))\n",
    "    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)\n",
    "    return image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "edb5394b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to make predictions\n",
    "def predict(image, model, labels):\n",
    "    image = np.expand_dims(image, axis=0)\n",
    "    prediction = model.predict(image)\n",
    "    predicted_label = labels[np.argmax(prediction)]\n",
    "    confidence = np.max(prediction)\n",
    "    return predicted_label, confidence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3a24fc0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main function to run the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Image Classification\")\n",
    "\n",
    "    # Load model and labels\n",
    "    model, labels = load_model_and_labels()\n",
    "\n",
    "    # Sidebar for file upload\n",
    "    uploaded_file = st.file_uploader(\"Choose an image...\", type=[\"jpg\", \"png\", \"jpeg\"])\n",
    "\n",
    "    # Perform prediction if file uploaded\n",
    "    if uploaded_file is not None:\n",
    "        image = tf.image.decode_image(uploaded_file.read(), channels=3)\n",
    "        st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "        # Preprocess image\n",
    "        preprocessed_image = preprocess_image(image)\n",
    "\n",
    "        # Make prediction\n",
    "        predicted_label, confidence = predict(preprocessed_image, model, labels)\n",
    "\n",
    "        st.write(\"Prediction:\", predicted_label)\n",
    "        st.write(\"Confidence:\", confidence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6fc2c74b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-03-29 17:09:22.627 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\siddh\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n"
     ]
    }
   ],
   "source": [
    "# Run the main function\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "cda45c32",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
