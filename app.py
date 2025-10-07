import streamlit as st
from src.predict import predict_top_k

def update_suggestions():
    text = st.session_state.user_text
    # Call prediction on the current text every time Enter is pressed
    st.session_state.top_words = predict_top_k(text.strip())

st.title("Next Word Predictor ✍️")

# Initialize session state
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "top_words" not in st.session_state:
    st.session_state.top_words = []

# Input box with on_change callback
st.text_input(
    "Type here:",
    key="user_text",
    on_change=update_suggestions
)

# Display suggestions
if st.session_state.top_words:
    st.write("### Suggestions:")
    st.write(", ".join(st.session_state.top_words))
