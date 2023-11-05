import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import os
from sklearn.feature_extraction.text import CountVectorizer

import joblib


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



def predictSpamHam(text):
    cv = joblib.load("./models/cv.pkl")
    text = cleanText(text)
    model = load_model("./models/spamDetection.h5")
    text = cv.transform([text])  
    pred = model.predict(text)
    pred = np.round(pred).astype(int)
    return pred[0][0]


def predictSentiment(text):
    cv = joblib.load("./models/cvsentiment.pkl")
    text = cleanText(text)
    model = load_model("./models/sentimentDetection.h5")
    text = cv.transform([text])
    pred = model.predict(text)
    pred = np.round(pred).astype(int)
    return pred[0][0]
    
def predictStress(text):
    cv = joblib.load("./models/cvstress.pkl")
    text = cleanText(text)
    model = load_model("./models/stressDetection.h5")
    text = cv.transform([text])
    pred = model.predict(text)
    pred = np.round(pred).astype(int)
    return pred[0][0]

def predictHate(text):
    cv = joblib.load("./models/cvhate.pkl")
    text = cleanText(text)
    model = load_model("./models/hateDetection.h5")
    text = cv.transform([text])
    pred = model.predict(text)
    pred = np.argmax(pred)
    return pred

def predictSarcasm(text):
    cv = joblib.load("./models/cvsarcasm.pkl")
    text = cleanText(text)
    pred = model.predict(pad_seq)
    pred = np.round(pred).astype(int)
    return pred[0][0]


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
            pred=predictSpamHam(text)
            if pred==0:
                st.write("The Text Is Ham")
            else:
                st.write("The Text Is Spam")


if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of A Text??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predictSentiment(text)
            if pred==0:
                st.write("The Sentiment Of The Text Is Negative")
            else:
                st.write("The Sentiment Of The Text Is Positive")

if rad=="Stress Detection":
    st.header("Detect Whether A Text Is Stressed Or Not??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predictStress(text)
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
            pred=predictHate(text)
            if pred==0:
                st.write("The Text Is Hate and Offensive Content")
            elif pred==1:
                st.write("The Text Is hatefull")
            else:
                st.write("The Text Is not hatefull and offensive")

if rad=="Sarcasm Detection":
    st.header("Detect Whether A Text Is Sarcasm Or Not??")

    text=st.text_area("Enter Text Here")
    if st.button("Predict"):
        if text=="":
            st.write("Please Enter Some Text")
        elif len(text) < 50:
            st.warning("Please Enter A Text Of Atleast 50 Characters")

        else:
            pred=predictSarcasm(text)
            if pred==0:
                st.write("The Text Is Not Sarcasm")
            else:
                st.write("The Text Is Sarcasm")
