#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import tensorflow as tf
import numpy as np


# In[8]:


# Function to compile the loaded model
def compile_model(model):
    # Compile the model with appropriate optimizer, loss function, and metrics
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# In[9]:


# Function for Tensorflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("/Users/siddh/OneDrive/Desktop/model final/keras_model.h5")
    compile_model(model)  # Compile the model manually
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# In[14]:


# Sidebar navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header(" PLANTCARE: PLANT DISEASE DETECTION")
    image_path = "/Users/siddh/OneDrive/Desktop/model final/home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the PlantCare: Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 8K RGB images of healthy and diseased crop leaves which are categorized into 28 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (6400 images)
                2. test (49 images)
                3. validation (1551 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=4, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            # Reading Labels
            # Load labels
            labels = load_labels("/Users/siddh/OneDrive/Desktop/model final/labels.txt")
            st.success("Model is Predicting it's a {}".format(labels[result_index]))
    else:
        st.warning("Please upload an image first.")


# In[ ]:




