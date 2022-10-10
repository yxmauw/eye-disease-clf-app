import numpy as np
import streamlit as st
import tensorflow as tf
from model_methods import predict

# configuration of the page
st.set_page_config(
    layout='centered',
    page_icon='👁️',
    page_title='Eye Disease Classifier',
    initial_sidebar_state='auto'
)

st.title('Eye Disease classifier')
st.info('Only classifies Cataract, Diabetic retinopathy, Glaucoma or Normal, unable to give more than 1 classification at a time')

new_img = st.file_uploader('PLease upload your retinal image')

def predict_upload(new_img):
  result = predict(new_img) # result is a probabilities array
  classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
  pred_prob = np.max(result, axis=-1) # max probability
  pred_class = classes[np.argmax(result, axis=-1)] # string
  st.success(f'There is a {pred_prob:.4f} that this retinal image shows {pred_class}')
  
# instantiate submit button
if st.button('Submit'):
   with st.sidebar:
       try: 
            predict_upload()
