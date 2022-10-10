import numpy as np
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
import io

def predict(image):
        interpreter = tf.lite.Interpreter('ENet_model.tflite')
        interpreter.allocate_tensors()
        #get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
      
        # Read the image and decode to a tensor
        img = Image.open(io.BytesIO(image.read()))
        img = img.convert('RGB')
        # Resize the image to the desired size
        img = img.resize((160,160))
        img = tf.keras.preprocessing.image.img_to_array(img)
  
        #Preprocess the image to required size and cast
        input_shape = input_details[0]['shape']
        input_tensor= np.array(np.expand_dims(img,0), dtype=np.float32)
        input_tensor= tf.keras.applications.efficientnet_v2.preprocess_input(input_tensor)
        #set the tensor to point to the input data to be inferred
      
        # Invoke the model on the input data
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        #Run the inference
        interpreter.invoke()
        output_details = interpreter.get_tensor(output_details[0]['index'])
        return output_details
