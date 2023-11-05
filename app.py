import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import os

def cleanText(text: str) -> str:
    """
    This function cleans the text by removing all the unnecessary characters.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

models_directory = "./models"
models = os.listdir(models_directory)

model_mapping = {
    "spam": None,
    "sentiment": None,
    "stress": None,
    "hate": None,
    "sarcasm": None
}

for keyword, model_var in model_mapping.items():
    for model_name in models:
        if keyword in model_name.lower():
            model_mapping[keyword] = load_model(os.path.join(models_directory, model_name))
            break  

spam_model = model_mapping["spam"]
sentiment_model = model_mapping["sentiment"]
stress_model = model_mapping["stress"]
hate_model = model_mapping["hate"]
sarcasm_model = model_mapping["sarcasm"]

tokenizer = Tokenizer(num_words=10000, split=' ')

def predict(
        text : str,
        tokenizer : object,
        model : object,
        ) -> int:
    """
    This function predicts the class of the text.

    Args:
        text (str): The text to be predicted.
        tokenizer (object): The tokenizer object.
        model (object): The model object.

    Returns:
        int: The predicted class.
    """
    text = cleanText(text)
    input_seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(input_seq, maxlen=100, padding='post')
    pred = model.predict(pad_seq)
    pred = np.round(pred).astype(int)
    return pred

rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"])

if rad=="Home":
    st.title("Complete Text Analysis App")
    st.image("Complete Text Analysis Home Page.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")

if rad=="Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predict(text,tokenizer,spam_model)
            if pred==1:
                st.write("The Text Is Spam")
            else:
                st.write("The Text Is Ham")

if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of A Text??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predict(text,tokenizer,sentiment_model)
            if pred==0:
                st.write("The Text Is Negative")
            else:
                st.write("The Text Is Positive")

if rad=="Stress Detection":
    st.header("Detect Whether A Text Is Stressed Or Not??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predict(text,tokenizer,stress_model)
            if pred==0:
                st.write("The Text Is Not Stressed")
            else:
                st.write("The Text Is Stressed")

if rad=="Hate and Offensive Content Detection":
    st.header("Detect Whether A Text Is Hate and Offensive Content Or Not??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predict(text,tokenizer,hate_model)
            if pred==0:
                st.write("The Text Is Highly Offensive")
            elif pred==1:
                st.write("The Text Is Offensive")
            else:
                st.write("The Text Is Not Offensive")

if rad=="Sarcasm Detection":
    st.header("Detect Whether A Text Is Sarcasm Or Not??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predict(text,tokenizer,sarcasm_model)
            if pred==0:
                st.write("The Text Is Not Sarcasm")
            else:
                st.write("The Text Is Sarcasm")
