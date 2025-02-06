import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊", layout="wide")

@st.cache_resource
def load_model_and_tfidf():
    dirname = os.path.dirname(os.path.abspath(__file__))  
    svc_model_path = os.path.join(dirname, "svc_model.pkl")
    tfidf_path = os.path.join(dirname, "tfidf_vectorizer.pkl")

    with open(svc_model_path, "rb") as f:
        svc_model = pickle.load(f)
    
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)

    return svc_model, tfidf

svc_model, tfidf = load_model_and_tfidf()

stop_words = set(stopwords.words('english')) - {'not', 'no'}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text).strip()

def preprocess_text(text):
    text = clean_text(text)
    return tfidf.transform([text])  

st.title("📊 Sentiment Analysis with SVC and TF-IDF")
user_text = st.text_area("✍️ Enter your text here:")

if st.button('🔍 Analyze Sentiment'):
    if user_text:
        processed_text = preprocess_text(user_text)
        prediction = svc_model.predict(processed_text)
        
        if prediction[0] == 1:
            st.success("😊 The text expresses **positive** sentiment! 👍")
        else:
            st.error("😞 The text expresses **negative** sentiment! 👎")

        st.markdown("### 📈 Visual Result:")
        fig, ax = plt.subplots(figsize=(6, 3))
        prediction_value = float(svc_model.decision_function(processed_text))
        ax.bar(["Positive 😊","Negative 😞"], [prediction_value, 1 - prediction_value], color=["green", "red"])
        ax.set_ylabel('Value')
        ax.set_title('Sentiment Analysis')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter some text to analyze.")
