
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Skin lesion Image Classifier")
st.text("Provide URL of skin Image for image classification")

@st.cache(allow_output_mutation=True)
def load_model():
   model = tf.keras.models.load_model('"C:\Users\swethareddy\Downloads\Skin lesion.ipynb"')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['melanocytic nevi','melanoma','benign keratosis-like lesions',' basal cell carcinoma',' pyogenic granulomas and hemorrhage','Actinic keratoses and intraepithelial carcinomae','dermatofibroma']

def scale(image):
  image = tf.cast(image, tf.float32)
  image /= 255.0

  return tf.image.resize(image,[224,224])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=7)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ','"C:\Users\swethareddy\OneDrive\Desktop\MP5\HAM10000_images_part_1\ISIC_0024306.jpg"')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying skin lesion Image', use_column_width=True) 
