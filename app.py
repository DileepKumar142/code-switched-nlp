import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "./model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model only once
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# -----------------------------
# Prediction
# -----------------------------
def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]

    sentiment = labels[predicted_class]

    confidence = (
        probabilities[0][predicted_class].item()
        * 100
    )

    return sentiment, confidence


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌐 Code-Switched Sentiment Analyzer")

st.write(
    "Analyze sentiment in Hindi-English code-switched text."
)

text = st.text_area(
    "Enter your sentence:",
    placeholder="Example: Yaar movie chala bagundi"
)

if st.button("Analyze Sentiment"):

    if text.strip():

        sentiment, confidence = predict_sentiment(text)

        st.subheader("Result")

        st.write(
            f"**Sentiment:** {sentiment}"
        )

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )

    else:

        st.warning(
            "Please enter a sentence."
        )