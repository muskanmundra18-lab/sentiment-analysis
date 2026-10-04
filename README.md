# IMDB Sentiment Analysis using an RNN

A natural-language processing project that uses an **RNN built with PyTorch** to classify movie reviews from the IMDB dataset.

## Pipeline

```text
IMDB Reviews
     ↓
Text Cleaning
     ↓
Tokenization / Stopword Removal
     ↓
Stemming
     ↓
TF-IDF Vectorization
     ↓
RNN
     ↓
Sigmoid Output
     ↓
Sentiment Prediction
```

## Data Processing

The notebook performs:

- Duplicate removal
- Lowercasing
- URL removal
- Punctuation removal
- HTML-tag removal
- Stopword removal
- Porter stemming
- Label encoding
- TF-IDF vectorization

The TF-IDF representation is limited to **5,000 features**.

## Model

The project implements a PyTorch RNN with:

- Hidden size: 128
- One recurrent layer
- Fully connected output layer
- Adam optimizer
- Binary cross-entropy loss
- 10 training epochs

The current notebook reports approximately **85.74% test accuracy**.

## Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Jupyter Notebook

## Getting Started

```bash
git clone https://github.com/muskanmundra18-lab/sentiment-analysis.git
cd sentiment-analysis
```

Install dependencies:

```bash
pip install pandas numpy torch scikit-learn nltk
```

Open:

```text
RNN_imdb.ipynb
```

Ensure the expected IMDB dataset is available before running the notebook.

## Learning Outcomes

- NLP preprocessing
- TF-IDF feature engineering
- Binary sentiment classification
- RNN implementation in PyTorch
- Training and evaluating sequence models

## Future Improvements

- Use learned word embeddings instead of TF-IDF vectors
- Compare RNN, LSTM, and GRU architectures
- Add precision, recall, F1-score and confusion matrix
- Add a prediction interface
- Compare against transformer-based models such as BERT

## Author

**Muskan Mundra**
