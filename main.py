import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# -------------------------------
# 1. Page configuration (ONCE)
# -------------------------------
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="🐦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# 2. Combined CSS – removes all Streamlit chrome & white bars
# -------------------------------
st.markdown("""
    <style>
    /* Remove Streamlit default header, footer, menu, deploy button */
    header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
    .stApp > header { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Remove padding/margins from main content areas */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    .stApp { margin-top: -50px; }

    /* Remove extra top margins from any legacy containers */
    .css-1d391kg, .css-18e3th9, .css-1v3fvcr, .view-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Your custom UI styles */
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1DA1F2, #0D8BD9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #657786;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 0rem;          /* ← FIXED: removes inner white gap */
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 0rem;    /* ← FIXED: removes outer white gap */
    }
    .stButton > button {
        background-color: #1DA1F2;
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0D8BD9;
    }
    .sentiment-box {
        font-size: 1.2rem;
        font-weight: 500;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 0.5rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    .footer {
        text-align: center;
        margin-top: 1rem;
        color: #8899A6;
        font-size: 0.75rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 3. Load model and tokenizer
# -------------------------------
@st.cache_resource
def load_resources():
    try:
        model = load_model('model.h5')
    except Exception as e:
        st.error(f"Failed to load model.h5: {e}")
        st.stop()
    try:
        with open('tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
    except Exception as e:
        st.error(f"Failed to load tokenizer.pkl: {e}")
        st.stop()
    return model, tokenizer

model, tokenizer = load_resources()

# -------------------------------
# 4. UI Components
# -------------------------------
st.markdown('<div class="title">🐦 Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a tweet and get real‑time sentiment prediction</div>', unsafe_allow_html=True)

with st.expander("📝 See example tweets"):
    st.markdown("- **Positive**: *I love this new update! It's amazing.*")
    st.markdown("- **Neutral**: *The weather is okay today.*")
    st.markdown("- **Negative**: *This product is terrible, worst purchase ever.*")

st.markdown('<div class="card">', unsafe_allow_html=True)
tweet = st.text_area(
    "✍️ Your tweet",
    height=120,
    placeholder="Type or paste a tweet here..."
)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if predict_button:
    if not tweet.strip():
        st.warning("Please enter a tweet before analyzing.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sequences = tokenizer.texts_to_sequences([tweet])
            sequences = pad_sequences(sequences, maxlen=99)
            prediction = model.predict(sequences)
            predicted_class = np.argmax(prediction, axis=1)[0]

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map[predicted_class]
        confidence = float(np.max(prediction) * 100)

        if sentiment == 'Positive':
            emoji = "😊"
            css_class = "sentiment-positive"
        elif sentiment == 'Neutral':
            emoji = "😐"
            css_class = "sentiment-neutral"
        else:
            emoji = "😞"
            css_class = "sentiment-negative"

        st.markdown(
            f"""
            <div class="sentiment-box {css_class}">
                {emoji} <strong>Sentiment:</strong> {sentiment}<br>
                <span style="font-size: 0.9rem;">Confidence: {confidence:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("🔎 Show detailed probabilities"):
            prob_data = {
                "Negative": float(prediction[0][0]),
                "Neutral": float(prediction[0][1]),
                "Positive": float(prediction[0][2])
            }
            st.bar_chart(prob_data)

st.markdown('<div class="footer">Built with Streamlit & TensorFlow | Twitter Sentiment Analysis RNN Model</div>', unsafe_allow_html=True)