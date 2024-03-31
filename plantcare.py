#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# In[2]:


# Load the model
model = load_model('/Users/siddh/OneDrive/Desktop/model final/keras_model.h5', compile=False)


# In[3]:


# Load the labels
class_names = open('/Users/siddh/OneDrive/Desktop/model final/labels.txt', "r").readlines()


# In[4]:


def preprocess_image(image):
    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array


# In[5]:


def predict(image):
    # Predict the class of the image
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image(image)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score


# In[7]:


def main():
    st.title('Image Classification')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)
        class_name, confidence_score = predict(image)
        st.write(f'Class: {class_name[2:]}')
        st.write(f'Confidence Score: {confidence_score}')


# In[8]:


if __name__ == "__main__":
    main()


# In[ ]:




