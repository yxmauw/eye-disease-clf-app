# reference https://stackoverflow.com/questions/69134379/how-to-make-prediction-based-on-model-tensorflow-lite
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt

@st.cache
def full_model():
        model = tf.keras.models.load_model("ENet_ep20_val0.311", 
                                            custom_objects={'f1_score': f1_score})
        return model

def predict(image): # to predict raw image input
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
        #input_shape = input_details[0]['shape']
        input_tensor= np.array(np.expand_dims(img,0), dtype=np.float32)
        input_tensor= tf.keras.applications.efficientnet_v2.preprocess_input(input_tensor)
        #set the tensor to point to the input data to be inferred
      
        # Invoke the model on the input data
        interpreter.set_tensor(input_details[0]['index'], input_tensor)

        #Run the inference
        interpreter.invoke()
        output_details = interpreter.get_tensor(output_details[0]['index'])
        return output_details

def orig_img(image):   
    img = Image.open(io.BytesIO(image.read()))
    img = img.convert('RGB')
    # Resize the image to the desired size
    img = img.resize((160,160))
    img = tf.keras.preprocessing.image.img_to_array(img)
  
    #Preprocess the image to required size and cast
    #input_shape = input_details[0]['shape']
    input_array= np.array(np.expand_dims(img,0), dtype=np.float32)
    input_array= tf.keras.applications.efficientnet_v2.preprocess_input(input_array)

    input_tensor = tf.convert_to_tensor(input_array)
    return input_tensor # output tensor format of image

def normalize_image(img): #normalise image
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm
# see this for cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plot_maps(img1, img2,vmin=0.3,vmax=0.7, mix_val=2): 
    fig, ax = plt.subplots()
    ax.imshow(img1*mix_val+img2/mix_val, cmap = "terrain" )
    plt.axis("off");
    st.pyplot(fig)
    #st.caption('Saliency Map')

# load full Saved model for Saliency and activation maps, unable to use tf lite model for these unless previously specified upon model construct
import keras.backend as K # F1 score metric custom object
def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def plot_gradient_maps(input_im): # plot_maps() and predict() function embedded        
    with tf.GradientTape() as tape:
        tape.watch(input_im)   
        result_img = full_model()(input_im)
        max_idx = tf.argmax(result_img,axis = 1)
        max_score = tf.math.reduce_max(result_img[0,max_idx[0]]) # tensor max probability
    grads = tape.gradient(max_score, input_im)
    plot_maps(normalize_image(grads[0]), normalize_image(input_im[0]))
