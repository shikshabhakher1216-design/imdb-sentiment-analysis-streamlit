import streamlit as st
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# load model & TF-IDF
model = joblib.load('model.joblib')

tfidf = joblib.load('tfidf.joblib')

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

# text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# prediction
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    #label mapping
    return "Positive" if prediction == 1 else "Negative"

# user interface
st.title("Sentiment Analyzer📉🎥📈")

user_input = st.text_input("Enter your review:")

if st.button("Predict"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text")


# sidebar panel

st.sidebar.header("🎯 Model Summery")




shape = st.sidebar.subheader("There are 5000 rows and 3 columns")

analyse = st.sidebar.subheader("Here we show sentiments according to the IMBD reviews")



# reset button
if st.button("Reset"):
    st.session_state.clear()
    st.rerun()

st.divider()
