import streamlit as st# type: ignore
import joblib# type: ignore
import re
from nltk.corpus import stopwords # type: ignore
from nltk.stem import PorterStemmer # type: ignore
import nltk # type: ignore

# Download NLTK stopwords with error handling
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Failed to download NLTK stopwords: {e}")
    st.stop()

# Load models and resources
try:
    vectorizer = joblib.load('model/vectorizer.joblib')
    model = joblib.load('model/MNB.joblib')
    encoder = joblib.load('model/label_encoder.joblib')
    stemmer = PorterStemmer()
except Exception as e:
    st.error(f"Failed to load model or resources: {e}")
    st.stop()

# Preprocess function (same as your notebook)
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ''
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]
    return ' '.join(text)

# Streamlit app
st.title("Spam Detection App")
user_input = st.text_area("Enter a message to check if it's spam or ham:")

if st.button("Predict"):
    if user_input:
        # Preprocess and predict
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        prediction = model.predict(vectorized_text)[0]
        label = encoder.inverse_transform([prediction])[0]
        st.write(f"Prediction: **{label}**")
    else:
        st.write("Please enter a message.")