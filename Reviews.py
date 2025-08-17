# Emotion Detection in Reviews with in-app BERT training
# ------------------------------------------------------
# 1) Automated emotion classification of food reviews using pre-trained BERT models
# 2) Traditional ML model training (Decision Tree & SVM) for comparison
# 3) Interactive data visualization and analysis
# 4) User interface for submitting new reviews with real-time emotion detection
# 5) Word cloud generation to visualize emotion-specific vocabulary patterns

# ===================== IMPORTS AND INITIAL SETUP =====================
import os
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"  # allow large uploads

import re
from datetime import datetime
from pathlib import Path
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------- NLTK BOOTSTRAP (Cloud-safe) --------------------
import nltk
NLTK_DATA_DIR = Path.home() / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
nltk.data.path.append(str(NLTK_DATA_DIR))

def _ensure_nltk_resource(resource: str, subdir: str):
    try:
        nltk.data.find(f"{subdir}/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)

_ensure_nltk_resource("punkt", "tokenizers")
_ensure_nltk_resource("stopwords", "corpora")

# Try to import Transformers early and show a friendly error if missing
try:
    from transformers import pipeline
    _HF_OK = True
except Exception as _e:
    _HF_OK = False
    _HF_ERR = _e

# NLP helpers (fallbacks handled later)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# -------------------- STREAMLIT PAGE CONFIG --------------------
st.set_page_config(page_title="Amazon Fine Food Reviews", layout="wide", page_icon="⭐⭐⭐⭐⭐")

# ===================== FILE PATHS =====================
BASE = Path(__file__).parent
IMAGE_FILES = [BASE / "Food.jpeg", BASE / "Food1.jpeg", BASE / "Food2.jpeg"]
DATA_PATH = BASE / "Reviews_7k.csv"   # unified path

# -------------------- UTILITIES ----------------------
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

@st.cache_data(show_spinner=False)
def load_csv_auto(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)

# ===================== HEADER IMAGE =====================
if "img_index" not in st.session_state:
    st.session_state.img_index = 0
current_image = IMAGE_FILES[st.session_state.img_index]

if current_image.exists():
    try:
        image_base64 = get_base64_image(current_image)
        st.markdown(
            f"""
            <div style="width:100%; text-align:center;">
                <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Header Image">
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        st.image(str(current_image), use_container_width=True)
else:
    st.warning(f"Image not found: {current_image.name}")

st.session_state.img_index = (st.session_state.img_index + 1) % len(IMAGE_FILES)

# ===================== APP INTRO =====================
st.markdown("""### 🤗😤 Emotion-based Analysis on Reviews""")

# ===================== DATA LOADING AND PREPROCESSING =====================
@st.cache_data(show_spinner=True)
def load_data(path: Path, sample_cap: int = 5000, seed: int = 42) -> pd.DataFrame:
    df = load_csv_auto(path)
    sample_size = min(sample_cap, len(df))
    return df.sample(sample_size, random_state=seed).reset_index(drop=True)

df_reviews = load_data(DATA_PATH)

# --------------------- BERT LABELING ---------------------
@st.cache_resource(show_spinner=True)
def label_emotions(data: pd.DataFrame) -> pd.DataFrame:
    if not _HF_OK:
        st.error("Failed to import Hugging Face Transformers. Check requirements/runtime.")
        st.exception(_HF_ERR)
        st.stop()

    clf = pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        top_k=None,
        device=-1  # CPU
    )

    texts = [str(t) for t in data['Text'].tolist()]
    results = clf(texts, truncation=True, max_length=512, batch_size=16)

    emotions = [max(r, key=lambda x: x['score'])['label'] if r else "neutral" for r in results]
    data = data.copy()
    data['emotion'] = emotions
    return data

df_labeled = label_emotions(df_reviews)

# ===================== TEXT PREPROCESSING =====================
def clean_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'<.*?>', " ", text)      # remove HTML
        text = re.sub(r'[^\w\s]', "", text)     # remove punctuation
        text = re.sub(r'\d+', " ", text)        # remove digits
        text = re.sub(r'\s+', " ", text).strip()
        cleaned.append(text)
    return cleaned

df_labeled['clean_text'] = clean_text(df_labeled['Text'])

# ===================== TOKENIZATION & STOPWORDS (with fallbacks) =====================
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOP_WORDS = set(ENGLISH_STOP_WORDS)

try:
    words = word_tokenize(' '.join(df_labeled['clean_text']))
except LookupError:
    words = ' '.join(df_labeled['clean_text']).split()

tok_text = pd.DataFrame({'Tokens': words})
filtered_text = [w for w in words if w.lower() not in STOP_WORDS]

# ===================== WORD CLOUD =====================
def plot_wordcloud(emotion):
    text = " ".join(df_labeled[df_labeled['emotion'] == emotion]['clean_text'])
    if not text.strip():
        st.info("No text found for this emotion.")
        return
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig = plt.figure()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)
    plt.close(fig)

# ===================== MODEL TRAINING (TF-IDF + DT/SVM) =====================
def _align_proba_to_dummies(y_true_dummies: pd.DataFrame, proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return np.column_stack([
        proba[:, class_to_idx[c]] if c in class_to_idx else np.zeros(proba.shape[0])
        for c in y_true_dummies.columns
    ])

def train_models():
    X = df_labeled['clean_text']
    y = df_labeled['emotion']

    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_proba = dt.predict_proba(X_test)

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_proba = svm.predict_proba(X_test)

    y_true_dummies = pd.get_dummies(y_test)
    dt_proba_aligned  = _align_proba_to_dummies(y_true_dummies, dt_proba,  dt.classes_)
    svm_proba_aligned = _align_proba_to_dummies(y_true_dummies, svm_proba, svm.classes_)

    metrics = {
        'Model': ['Decision Tree', 'SVM'],
        'Precision': [
            precision_score(y_test, dt_pred, average='weighted', zero_division=0),
            precision_score(y_test, svm_pred, average='weighted', zero_division=0)
        ],
        'Recall': [
            recall_score(y_test, dt_pred, average='weighted', zero_division=0),
            recall_score(y_test, svm_pred, average='weighted', zero_division=0)
        ],
        'F1-score': [
            f1_score(y_test, dt_pred, average='weighted', zero_division=0),
            f1_score(y_test, svm_pred, average='weighted', zero_division=0)
        ],
        'ROC-AUC': [
            roc_auc_score(y_true_dummies.values, dt_proba_aligned, average='weighted', multi_class='ovr'),
            roc_auc_score(y_true_dummies.values, svm_proba_aligned, average='weighted', multi_class='ovr')
        ]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, dt, svm, tfidf

metrics_df, dt_model, svm_model, tfidf_vectorizer = train_models()

# ===================== STREAMLIT USER INTERFACE =====================
st.markdown("""
    <style>
    div[data-baseweb="tab-list"] { gap: 40px; }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Dataset", "😊 Emotion Detection", "☁ Word Clouds", "✅ Conclusion", "User Interface", "Submitted Review"]
)

# TAB 1: DATASET EXPLORATION
with tab1:
    st.subheader("📊 Dataset content")

    if st.checkbox("Show Sample Data"):
        st.write(df_labeled.head())
    st.write("Emotion distribution in the sample:")
    st.bar_chart(df_labeled['emotion'].value_counts())

# TAB 2: EMOTION DETECTION AND MODEL PERFORMANCE
with tab2:
    st.subheader("😊 Emotion Detection")
    st.write("Model performance metrics:")
    st.dataframe(metrics_df)
    st.bar_chart(metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-score', 'ROC-AUC']])

    user_input = st.text_area("Enter text for emotion prediction:")
    if st.button("Predict Emotion"):
        if user_input.strip():
            cleaned_input = clean_text([user_input])
            input_tfidf = tfidf_vectorizer.transform(cleaned_input)
            pred_dt = dt_model.predict(input_tfidf)[0]
            pred_svm = svm_model.predict(input_tfidf)[0]
            st.write(f"*Decision Tree prediction:* {pred_dt}")
            st.write(f"*SVM prediction:* {pred_svm}")

# TAB 3: WORD CLOUD VISUALIZATION
with tab3:
    st.subheader("☁ Word clouds content")
    emotions = df_labeled['emotion'].unique()
    emotion_choice = st.selectbox("Select emotion:", emotions)
    plot_wordcloud(emotion_choice)

# TAB 4: CONCLUSIONS AND INSIGHTS
with tab4:
    st.subheader("Conclusion")
    st.write("""
    - We used TF-IDF with Decision Tree and SVM classifiers for emotion detection.
    - SVM generally performs better in text classification tasks.
    - Word clouds give insights into common words per emotion.
    - Auto-labeling was done using a pre-trained DistilBERT emotion model.
    """)

# TAB 5: USER REVIEW SUBMISSION INTERFACE
with tab5:
    @st.cache_resource(show_spinner=True)
    def load_emotion_model():
        if not _HF_OK:
            st.error("Failed to import Hugging Face Transformers. Check requirements/runtime.")
            st.exception(_HF_ERR)
            st.stop()
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None, device=-1)

    emotion_classifier = load_emotion_model()

    st.title("🍽 Food Review & Feedback System")
    st.write("Share your experience with our dishes!")

    CSV_FILE = BASE / "food_reviews.csv"
    if CSV_FILE.exists():
        df_reviews_file = pd.read_csv(CSV_FILE)
    else:
        df_reviews_file = pd.DataFrame(columns=["Customer", "Food Item", "Rating", "Review", "Emotion", "Date"])

    with st.form("review_form", clear_on_submit=True):
        customer_name = st.text_input("👤 Your Name")
        food_item = st.text_input("🍔 Food Item")
        rating = st.slider("⭐ Rating", 1, 5, 5)
        review_text = st.text_area("📝 Write your review:", height=150)
        submitted = st.form_submit_button("💾 Submit Review")

        if submitted:
            if customer_name.strip() and food_item.strip() and review_text.strip():
                emotion_result = emotion_classifier(review_text, truncation=True, max_length=512)[0]["label"]
                new_review = {
                    "Customer": customer_name,
                    "Food Item": food_item,
                    "Rating": rating,
                    "Review": review_text,
                    "Emotion": emotion_result,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                df_reviews_file = pd.concat([df_reviews_file, pd.DataFrame([new_review])], ignore_index=True)
                df_reviews_file.to_csv(CSV_FILE, index=False)
                st.success(f"✅ Review saved with detected emotion: **{emotion_result}**")
            else:
                st.warning("⚠ Please fill in all fields.")

# TAB 6: REVIEW ANALYTICS AND MANAGEMENT
with tab6:
    st.subheader("📊 Reviews Summary")

    CSV_FILE = BASE / "food_reviews.csv"
    if CSV_FILE.exists():
        df_reviews_file = pd.read_csv(CSV_FILE)

        csv_data = df_reviews_file.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Reviews CSV",
            data=csv_data,
            file_name="food_reviews.csv",
            mime="text/csv"
        )

        st.write(f"**Total Reviews:** {len(df_reviews_file)}")
        st.write(f"**Average Rating:** {df_reviews_file['Rating'].mean():.2f} ⭐")

        if not df_reviews_file['Food Item'].empty:
            most_food = df_reviews_file["Food Item"].mode()[0]
            st.write(f"**Most Ordered Dish:** {most_food}")

        if "Emotion" in df_reviews_file.columns:
            emotion_counts = df_reviews_file["Emotion"].value_counts()
            st.markdown("### 📊 Distribution of Emotion")
            st.bar_chart(emotion_counts)

        st.markdown("### 📊 Distribution of Ratings")
        st.bar_chart(df_reviews_file["Rating"].value_counts().sort_index())
    else:
        st.info("No reviews available yet.")
