import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import re

nltk.download('stopwords')
stop_words = stopwords.words('english')

model_file = ".\\model\\RFC.joblib"
model = joblib.load(model_file)

vectorizer_file = ".\\model\\vectorizer.joblib"
vectorizer = joblib.load(vectorizer_file)

# vectorizer = TfidfVectorizer(max_features = 3000)
stemmer = PorterStemmer()

def processed_text(text):
    cleaned_text = re.sub('[^a-zA-Z]',' ',text).lower().split()

    cleaned_text = [words for words in cleaned_text if words not in stop_words]

    stemmed_text = [stemmer.stem(words) for words in cleaned_text]

    return ' '.join(stemmed_text)

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
        processed_message = processed_text(message)
        vect_message = vectorizer.transform([processed_message]).toarray()
        if not processed_message.strip():
            st.error('Processed Message is empty. Please Enter the meaning full text')
        else:
            try:
                st.write('input shape', vect_message.shape)
                prediction = model.predict(vect_message)
                st.write('prediciton value',prediction)
                if prediction == 1:
                    st.success('It is Spam message')
                else:
                    st.success('It is a Ham Message')

            except Exception as e:
                st.error(f"An error occurred: {e}")
        # st.write("Processing:", message)