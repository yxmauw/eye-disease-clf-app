import numpy as np
import tensorflow as tf
import streamlit as st

@st.cache
def load_model():  # Load the TFLite model and allocate tensors.
   with open('ENet_model.tflite','rb') as f:
        model = tf.lite.Interpreter(model_content=f)
        model.allocate_tensors()
   return model

def predict(image):
  # read image
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_image(image, channels=3)
  # Resize the image to the desired size
  img = tf.image.resize(img, [160, 160])
  
  # preprocess new image
  tf.keras.applications.efficientnet_v2.preprocess_input(img)
  
  # load model if not previously loaded
  model = load_model()
  pred = model.predict(img)
  return pred
