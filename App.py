import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('Model.h5')
    return model

model = load_model()

# Map predicted class indices to class labels
class_labels = ['glaucoma', 'normal', 'diabetic_retinopathy', 'cataract']

# Create a function to make predictions
def predict(image):
    # Preprocess the image
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = tf.image.resize(image, [60, 60])
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    prediction_class = np.argmax(prediction, axis=1)
    return prediction_class

# Streamlit app
def main():
    st.title('Eye Disease Classifier')

    # Input image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        if st.button('Predict'):
            prediction_class = predict(image)
            predicted_label = class_labels[prediction_class[0]]
            st.write('Predicted Class:', predicted_label)

if __name__ == '__main__':
    main()
