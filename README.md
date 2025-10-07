# Next Word Predictor 📝

A deep learning project that predicts the next word in a sentence using an LSTM model built with PyTorch. The project includes data preprocessing, vocabulary building, training sequences, and a Streamlit interface for interactive word prediction.

---

## Features

- Train an LSTM model on your text corpus.
- Predict the next word based on input text.
- Interactive Streamlit app to visualize predictions.
- Handles unknown words with `<UNK>` token.
- Supports variable-length input sequences with padding.

---

## Project Structure

```bash
lstm-next-word-predictor/
│
├─ data/
│ └─ document.txt # Text corpus for training
│
├─ src/
│ ├─ dataset.py # Preprocessing and dataset creation
│ ├─ model.py # LSTM model definition
│ ├─ predict.py # Prediction functions
│ └─ train.py # Training script
│
├─ app.py # Streamlit interface
└─ README.md
