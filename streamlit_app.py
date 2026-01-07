import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))
sentiment_important = {
    "not", "no", "never", "none", "without", "below",
    "hardly", "barely", "scarcely", "poor"
}
stop = stop - sentiment_important

model = joblib.load("sentiment_model_retrained.pkl")
tfidf = joblib.load("tfidf_vectorizer_retrained.pkl")

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"[^a-zA-Z ]", "", t)
    t = " ".join([w for w in t.split() if w not in stop])
    return t

def soft_positive_bias(original_text, cleaned_text):
    strong_neg = {
        "bad","terrible","awful","horrible","worst","poor",
        "sucks","disappoint","disappointed","waste","wasted",
        "useless","hate","broken","issue","problem","angry",
        "scam","fraud","pathetic"
    }
    positive_words = {
        "good","great","nice","fine","okay","ok","love","amazing","awesome",
        "pleasant","decent","worth","comfortable","enjoy","liked"
    }

    lower = original_text.lower()
    for neg in strong_neg:
        if neg in lower:
            return None

    pattern = r"\b(not|never|no|none|without|hardly|barely|scarcely)\b\s+\b(" + "|".join(re.escape(w) for w in positive_words) + r")\b"
    if re.search(pattern, lower):
        return 0

    if len(cleaned_text.split()) <= 3:
        return 1

    cleaned_words = set(cleaned_text.split())
    if len(cleaned_words & strong_neg) == 0:
        return 1

    return None

st.markdown(
    """
    <style>
    .centered { text-align: center; }
    .stButton button { margin: 0 auto; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="centered">Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="centered">Enter a sentence and I’ll predict whether it\'s <b>Positive</b> or <b>Negative</b>.</p>', unsafe_allow_html=True)

user_text = st.text_area("Enter text:", key="text_area")

if st.button("Predict"):
    cleaned = clean_text(user_text)
    bias = soft_positive_bias(user_text, cleaned)

    if bias is not None:
        pred = bias
    else:
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]

    if pred == 1:
        st.success("Positive 😀")
    else:
        st.error("Negative 😠")
