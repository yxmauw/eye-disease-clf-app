import numpy as np
import streamlit as st
import tensorflow as tf
from model_methods import predict

# configuration of the page
st.set_page_config(
    layout='wide',
    page_icon='👁️',
    page_title='Eye Disease Classifier',
    initial_sidebar_state='auto'
)

st.title('👁️ Eye Disease classifier')
st.info('Only classifies Cataract, Diabetic retinopathy, Glaucoma or Normal. \n\n Model is restricted to giving 1 class at a time')

new_img = st.file_uploader('PLease upload your retinal image')

def predict_upload():
  result = predict(new_img) # result is a probabilities array 
  classes = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
  max_result = (np.max(result, axis=-1)) * 100 # max probability
  pred_prob = np.format_float_positional(max_result, precision=2) # format probability
  pred_class = classes[(np.argmax(result, axis=-1)[0])] # string
  st.success(f'There is a **{pred_prob}** % probability that this retinal image shows **{pred_class}**')
  
# instantiate submit button
if st.button('Classify'):
   with st.sidebar:
       try: 
            predict_upload()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(new_img)
       except:
            st.warning('''
            Unable to detect image. 
            Please upload retinal image for classification. 
            \n\n Thank you 🙏
            ''')
            
    
