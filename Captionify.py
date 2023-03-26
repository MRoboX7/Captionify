import streamlit as st
import numpy as np
import cv2
import pickle
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model,load_model
from keras.utils import pad_sequences
from keras.utils.image_utils import load_img, img_to_array

# Configure Streamlit app settings
st.set_page_config(page_title='Captionify', page_icon=':page:', layout='wide')

# Define Footer
def footer():
    # Add footer with contact information
    st.markdown(
        """
        <style>
        footer {
            visibility: visible;
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #2D3748;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <footer>
        <p style="color:white;">Contact us at info@captionify.com</p>
        </footer>
        """,
        unsafe_allow_html=True
    )

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text

# Define app theme
def set_theme():
    primaryColor = "#36a3f3"
    backgroundColor = "#f2f2f2"
    secondaryBackgroundColor = "#f9f9f9"
    textColor = "#212529"
    font = "sans-serif"
    css = f"""
        body {{
            color: {textColor};
            background-color: {backgroundColor};
            font-family: {font};
        }}

        .stButton {{
            background-color: {primaryColor};
            color: {backgroundColor};
        }}

        .stTextInput, .stTextArea {{
            background-color: {secondaryBackgroundColor};
        }}

        .stPlotlyChart {{
            background-color: {backgroundColor};
        }}
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Image Feature Extraction
@st.cache_resource()
def load_extraction_model():
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    return vgg_model

# Model
@st.cache_resource()
def load_prediction_model():
    model = load_model('model.h5',compile=False)
    model.compile = True

    with open("vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)

    return model, vocab

def generete_captions(image,extraction_model,model,tokenizer,max_length):
    image = load_img(image, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = extraction_model.predict(image, verbose=0)
    return predict_caption(model, feature, tokenizer, max_length)

def main():
    set_theme()

    st.title('Captionify')
    st.write('Generete descriptive captions for your images :)')

    st.sidebar.title('Captionify')
    
    # Img Relay
    img_file_buffer = st.file_uploader('Upload Image', type=['jpg','png','jpeg'])
    input_img_container = st.empty()

    if img_file_buffer:
        input_img_container.image(img_file_buffer.read())

    predict_button = st.button('Generete Captions',use_container_width=True)

    if predict_caption:
        if img_file_buffer:
            extraction_model = load_extraction_model()
            model, vocab = load_prediction_model()
            caption = generete_captions(img_file_buffer,extraction_model,model,vocab,35)
            caption = caption.strip('startseq').strip('endseq').title()
            st.success('Caption : ' + caption)
        else : st.warning("Please Upload A File")

    footer()

if __name__ == '__main__':
    main()