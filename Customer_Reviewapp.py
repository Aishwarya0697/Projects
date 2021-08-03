import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from preprocessing import preprocess
import Setiment_analysis
import models

#read in saved classification model
load_clf = pickle.load(open('sentiment_clf.pkl','rb'))
load_cv = pickle.load(open("CVectorizor.pkl",'rb'))
st.title("Positive or negative Review Prediction App")
st.subheader("Natural Language Processing On the Go..")
st.markdown("""
        	#### Description
        	+ This is a Natural Language Processing(NLP) Based App useful for basic NLP task
        	Sentiment analysis 
        	""")
# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['tsv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, error_bad_lines=False)
# Displays the user input features
st.subheader('User Input features')
if uploaded_file:
    st.write(input_df)
#Sentiment Analysis

st.subheader("Analyse Your Text")
# Apply model on the prediction
st.subheader('Prediction')
message = st.text_area("Enter your text here", "type here")
if st.button("Analyse"):
    data = [message]
    vectorizer = load_cv.transform(data).toarray()
    pred = load_clf.predict(vectorizer)
    prediction_proba = load_clf.predict_proba(vectorizer)
#output results
    if pred == 0:
        st.write('Negative Review')
        st.write(f'Confidence: {prediction_proba * 100}')

    else:
        st.write('Positive Review')
        st.write(f'Confidence: {prediction_proba * 100}')
