# Emotion Detection in Reviews with in-app BERT training
# ------------------------------------------------------
# This is a comprehensive Streamlit web application that provides:
# 1. Automated emotion classification of food reviews using pre-trained BERT models
# 2. Traditional ML model training (Decision Tree & SVM) for comparison
# 3. Interactive data visualization and analysis
# 4. User interface for submitting new reviews with real-time emotion detection
# 5. Word cloud generation to visualize emotion-specific vocabulary patterns

# ===================== IMPORTS AND INITIAL SETUP =====================
# Import required libraries for data handling, ML, NLP, visualization, and the web app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Set Streamlit's maximum upload size to 1GB to handle large CSV files
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"
import streamlit as st
import re
# Natural Language Processing libraries
import nltk
from PIL import Image
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# Machine Learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from transformers import pipeline
# Download required NLTK data packages
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# File handling and encoding utilities
import base64
from pathlib import Path
# Database connectivity (MySQL)
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import plotly.express as px


# -------------------- STREAMLIT PAGE CONFIG --------------------
# Configure the Streamlit page (title, icon, layout)
st.set_page_config(page_title="Amazon Fine Food Reviews", layout="wide", page_icon="⭐⭐⭐⭐⭐")

# ===================== FILE PATHS =====================
# Set up file locations and basic settings for the app
BASE = Path(__file__).parent
IMAGE_FILES = [
    BASE / "Food.jpeg",
    BASE / "Food1.jpeg",
    BASE / "Food2.jpeg"
    ]
# Upload CSV file
DATA_PATH = BASE / "Reviews.csv"

# -------------------- UTILITIES ----------------------
#Convert an image file to base64 encoding for embedding in HTML.
def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

 # Cache results to avoid reloading same file
@st.cache_data(show_spinner=False)
def load_csv_auto(path: Path) -> pd.DataFrame:
    """Robust CSV loader that automatically detects encoding and delimiter."""
    # Try different encodings in order of preference
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path, low_memory=False)  # last attempt with defaults to surface a clear error

# ===================== HEADER IMAGE =====================
# Initialize index in session state
if "img_index" not in st.session_state:
    st.session_state.img_index = 0

# Get current image
current_image = IMAGE_FILES[st.session_state.img_index]

# Display current header image
if current_image.exists():
    try:
        # Convert image to base64 and embed in HTML for display
        image_base64 = get_base64_image(current_image)
        st.markdown(
            f"""
            <div style="width:100%; text-align:center;">
                <img src="{image_base64}" style="width:100%; max-height:250px; object-fit:cover;" alt="Header Image">
            </div>
            """,
            unsafe_allow_html=True  # Allow HTML rendering in Streamlit
        )
    except Exception as e:
        st.image(str(current_image), use_container_width=True)
else:
    st.warning(f"Image not found: {current_image.name}")

# Move to the next image for the next page refresh
st.session_state.img_index = (st.session_state.img_index + 1) % len(IMAGE_FILES)

# =====================APPLICATION INTRO =====================
st.markdown("""
###  🤗😤 Emotion-based on Review

""")

# ===================== DATA LOADING AND PREPROCESSING =====================
#Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Reviews_7k.csv")
        
    # Sample up to 5000 rows, or all rows if fewer than 5000
    sample_size = min(5000, len(df))
    df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return df_sample

# Load the sample dataset
df_reviews = load_data()

# Automatically label emotions in review texts using a pre-trained BERT model.
@st.cache_resource
def label_emotions(data):
    # Load pre-trained emotion classification pipeline
    classifier = pipeline("text-classification",
                          model="bhadresh-savani/distilbert-base-uncased-emotion",
                          top_k=None)
    # Process each review text
    emotions = []
    for text in data['Text']:
        try:
            result = classifier(str(text))[0]
        # Get emotion with highest score
            top_emotion = max(result, key=lambda x: x['score'])['label']
            emotions.append(top_emotion)
        except:
            emotions.append("neutral")
    # Add emotion labels to the dataframe
    data['emotion'] = emotions
    return data

# Apply emotion labeling to the review dataset
df_labeled = label_emotions(df_reviews)

# ===================== TEXT PREPROCESSING =====================
# Clean Text
def clean_text(texts):
    cleaned = []
    for text in texts:
        text = str(text).lower()  # Convert to string and lowercase
        text = re.sub(r'<.*?>', " ", text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', "", text)  # Remove punctuation
        text = re.sub(r'\d+', " ", text)  # Remove digits
        text = re.sub(r'\s+', " ", text).strip()  #replace multiple spaces with single space
        cleaned.append(text)
    return cleaned

# Apply text cleaning to the review texts
df_labeled['clean_text'] = clean_text(df_labeled['Text'])

# ===================== TEXT TOKENIZATION AND ANALYSIS =====================
# Tokenize all cleaned text into individual words
words = word_tokenize(' '.join(df_labeled['clean_text']))
tok_text = pd.DataFrame({'Tokens': words})

# Display stop words from the nltk
stop_words_disp = set(stopwords.words('english'))

# Removal of the stopwords
filtered_text = [word for word in words if word.lower() not in stop_words_disp]

# Generate and display a word cloud for a specific emotion category.
def plot_wordcloud(emotion):
    text = " ".join(df_labeled[df_labeled['emotion'] == emotion]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Train traditional machine learning models for emotion classification.
def train_models():
    # Prepare features (X) and target labels (y)
    X = df_labeled['clean_text']
    y = df_labeled['emotion']

    # Convert text to numerical features using TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X)

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Decision Tree Model
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # SVM Model
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    # Model Evaluation
    metrics = {
        'Model': ['Decision Tree', 'SVM'],
        'Precision': [
            precision_score(y_test, dt_pred, average='weighted'),
            precision_score(y_test, svm_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_test, dt_pred, average='weighted'),
            recall_score(y_test, svm_pred, average='weighted')
        ],
        'F1-score': [
            f1_score(y_test, dt_pred, average='weighted'),
            f1_score(y_test, svm_pred, average='weighted')
        ],
        'ROC-AUC': [
            roc_auc_score(pd.get_dummies(y_test), dt.predict_proba(X_test), average='weighted', multi_class='ovr'),
            roc_auc_score(pd.get_dummies(y_test), svm.predict_proba(X_test), average='weighted', multi_class='ovr')
        ]
    }

    # Convert metrics to DataFrame for easy display and comparison
    metrics_df = pd.DataFrame(metrics)
    return metrics_df, dt, svm, tfidf

# Train models and store results
metrics_df, dt_model, svm_model, tfidf_vectorizer = train_models()

# ===================== STREAMLIT USER INTERFACE =====================
# Custom CSS for spacing tabs
st.markdown("""
    <style>
    div[data-baseweb="tab-list"] {
        gap: 40px; /* space between tabs */
    }
    </style>
    """, unsafe_allow_html=True)

# Create tabbed interface for different app sections
tab1, tab2, tab3, tab4,tab5, tab6 = st.tabs(["📊 Dataset", "😊 Emotion Detection", "☁ Word Clouds", "✅ Conclusion","User Interface","Submitted Review"])

#Streamlit Pages
#TAB 1: DATASET EXPLORATION
with tab1:
    st.subheader("📊 Dataset content")

    if st.checkbox("Show Sample Data"):
        st.write(df_labeled.head())
    st.write("Emotion distribution in the sample:")
    st.bar_chart(df_labeled['emotion'].value_counts())

#TAB 2: EMOTION DETECTION AND MODEL PERFORMANCE
with tab2:

    st.subheader("😊 Emotion Detection")
    st.write("Model performance metrics:")
    st.dataframe(metrics_df)
    st.bar_chart(metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-score', 'ROC-AUC']])

    # Allow users to input text and get real-time emotion predictions
    user_input = st.text_area("Enter text for emotion prediction:")
    if st.button("Predict Emotion"):
        if user_input.strip():
            cleaned_input = clean_text([user_input])    # Apply same text cleaning as used in training
            input_tfidf = tfidf_vectorizer.transform(cleaned_input)  # Convert text to TF-IDF features using the trained vectorizer
            # Get predictions from both trained models
            pred_dt = dt_model.predict(input_tfidf)[0]
            pred_svm = svm_model.predict(input_tfidf)[0]
            st.write(f"*Decision Tree prediction:* {pred_dt}")
            st.write(f"*SVM prediction:* {pred_svm}")

#TAB 3: WORD CLOUD VISUALIZATION
with tab3:
    st.subheader("☁ Word clouds content")
    emotions = df_labeled['emotion'].unique()
    emotion_choice = st.selectbox("Select emotion:", emotions)
    plot_wordcloud(emotion_choice)

#TAB 4: CONCLUSIONS AND INSIGHTS
with tab4:
    st.write("✅ Conclusion content here")
    st.subheader("Conclusion")
    st.write("""
    - We used TF-IDF with Decision Tree and SVM classifiers for emotion detection.
    - SVM generally performs better in text classification tasks.
    - Word clouds give insights into common words per emotion.
    - Auto-labeling was done using a pre-trained RoBERTa model.
    """)

#TAB 5: USER REVIEW SUBMISSION INTERFACE
with tab5:
    # Load the emotion classification model for real-time prediction.
    @st.cache_resource
    def load_emotion_model():
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")


    # Load the emotion classification model
    emotion_classifier = load_emotion_model()

    # Set page configuration for this section
    st.set_page_config(page_title="🍽 Food Review App", page_icon="🍔", layout="centered")
    st.title("🍽 Food Review & Feedback System")
    st.write("Share your experience with our dishes!")

    # File to store submitted reviews
    CSV_FILE = "food_reviews.csv"

    # Load existing reviews if the file exists
    if os.path.exists(CSV_FILE):
        df_reviews = pd.read_csv(CSV_FILE)
    else:
        # Create empty DataFrame with required columns if file doesn't exist
        df_reviews = pd.DataFrame(columns=["Customer", "Food Item", "Rating", "Review", "Emotion", "Date"])

# ===================== REVIEW SUBMISSION FORM  =====================
    # Create a form that clears after submission
    with st.form("review_form", clear_on_submit=True):
        customer_name = st.text_input("👤 Your Name")
        food_item = st.text_input("🍔 Food Item")
        rating = st.slider("⭐ Rating", 1, 5, 5)
        review_text = st.text_area("📝 Write your review:", height=150)
        submitted = st.form_submit_button("💾 Submit Review")  # Form submission button

        # Process form submission
        if submitted:
            if customer_name.strip() and food_item.strip() and review_text.strip():
                emotion_result = emotion_classifier(review_text)[0]["label"]

                # Create new review record
                new_review = {
                    "Customer": customer_name,
                    "Food Item": food_item,
                    "Rating": rating,
                    "Review": review_text,
                    "Emotion": emotion_result,
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Add new review to existing dataframe
                df_reviews = pd.concat([df_reviews, pd.DataFrame([new_review])], ignore_index=True)

                # Save updated dataframe to CSV file
                df_reviews.to_csv(CSV_FILE, index=False)

                st.success(f"✅ Review saved with detected emotion: **{emotion_result}**")
            else:
                st.warning("⚠ Please fill in all fields.")

#TAB 6: REVIEW ANALYTICS AND MANAGEMENT
with tab6:
    st.subheader("📊 Reviews Summary")

    # CSV file path
    CSV_FILE = "food_reviews.csv"

    if os.path.exists(CSV_FILE):
        df_reviews = pd.read_csv(CSV_FILE)

        # Create downloadable CSV file
        csv_data = df_reviews.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Reviews CSV",
            data=csv_data,
            file_name="food_reviews.csv",
            mime="text/csv"
        )

        # Summary stats
        st.write(f"**Total Reviews:** {len(df_reviews)}")
        st.write(f"**Average Rating:** {df_reviews['Rating'].mean():.2f} ⭐")

        # Most ordered food
        if not df_reviews['Food Item'].empty:
            most_food = df_reviews["Food Item"].mode()[0]
            st.write(f"**Most Ordered Dish:** {most_food}")

        # Emotion distribution
        if "Emotion" in df_reviews.columns:
            emotion_counts = df_reviews["Emotion"].value_counts()
            st.markdown("### 📊 Distribution of Emotion")
            st.bar_chart(emotion_counts)

        # Ratings distribution
        st.markdown("### 📊 Distribution of Ratings")
        st.bar_chart(df_reviews["Rating"].value_counts().sort_index())

    else:
        st.info("No reviews available yet.")
