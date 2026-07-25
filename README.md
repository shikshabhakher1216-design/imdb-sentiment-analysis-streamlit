# imdb-sentiment-analysis-streamlit
"An end-to-end NLP sentiment analysis web application trained on 50,000 IMDB movie reviews using TF-IDF vectorization, Multinomial Naive Bayes, and Streamlit."


# 🎬 IMDB Sentiment Analyzer (NLP & Streamlit Web App)

An end-to-end Natural Language Processing (NLP) sentiment classification pipeline and interactive **Streamlit** web application trained on the **IMDB 50k Movie Reviews Dataset**.

---

## 📌 Project Architecture & Workflow

1. **Text Preprocessing & Normalization (`nltk` & `re`):**
   * Normalizes raw reviews by converting text to lowercase and stripping special characters/punctuation using Regular Expressions (`re.sub('[^a-z]', ' ', text)`).
   * Tokenizes text and filters out English **Stopwords**.
   * Applies **Porter Stemming** (`PorterStemmer`) to reduce words to root forms.

2. **Feature Extraction & Machine Learning:**
   * Extracts top $5,000$ textual features using `TfidfVectorizer(max_features=5000)`.
   * Trains a **Multinomial Naive Bayes Classifier (`MultinomialNB`)** on an 80/20 train-test split.
   * Exports trained assets (`model.joblib` and `tfidf.joblib`) using `joblib`.

3. **Interactive Web App (`app.py`):**
   * Serves a user interface using **Streamlit** for real-time sentiment prediction on custom movie reviews.
   * Includes dynamic input clearing and model metrics summary panel.

---

## 🧰 Tech Stack

* **Language:** Python
* **NLP & ML:** NLTK, Scikit-Learn
* **Web Framework:** Streamlit
* **Asset Serialization:** Joblib
* **Data Processing:** Pandas, NumPy

---
