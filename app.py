import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re

nltk.download('stopswords')
stop_words = stopwords.words('english')

file = ".\\model\\support_vector_classifier.joblib"
model = joblib.load(file)

vectorizer = TfidfVectorizer(max_features = 3000)
stemmer = PorterStemmer()

st.title('Spam Detection')

st.header('Enter Text')


message = st.text_area("Enter a message", "Type your message here...")
if st.button("Classify"):
    if not message.strip():
        st.error("Please enter a non-empty message.")
    elif all(word in stop_words for word in message.lower().split()):
        st.error("Message contains only stop words. Please include meaningful words.")
    else:
        # Process the message

        st.write("Processing:", message)