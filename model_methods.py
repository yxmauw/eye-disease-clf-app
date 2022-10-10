import numpy as np
import tensorflow as tf
import streamlit as st
import cv2

def predict(image):
   with open('ENet_model.tflite','rb') as f:
        interpreter = tf.lite.Interpreter(model_content=f)
        interpreter.allocate_tensors()
        #get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # read image
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_image(image, channels=3)
        # Resize the image to the desired size
        img = tf.image.resize(img, [160, 160])
        #Preprocess the image to required size and cast
        input_shape = input_details[0]['shape']
        input_tensor= np.array(np.expand_dims(img,0), dtype=np.float32)
        input_tensor= tf.keras.applications.efficientnet_v2.preprocess_input(input_tensor)
        #set the tensor to point to the input data to be inferred
        input_index = interpreter.get_input_details()[0]["index"]
        interpreter.set_tensor(input_index, input_tensor)
        #Run the inference
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]  
   return output_details
