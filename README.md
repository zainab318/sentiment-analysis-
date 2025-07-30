# IMDB Sentiment Analysis

A comprehensive Natural Language Processing (NLP) project that performs sentiment analysis on IMDB movie reviews using machine learning techniques.

## 📋 Overview

This project implements a complete sentiment analysis pipeline that can classify movie reviews as either positive or negative. The implementation uses advanced NLP techniques including:

- **Text Preprocessing**: HTML tag removal, lowercasing, punctuation removal, tokenization, stop word removal, and lemmatization
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Machine Learning**: Logistic Regression classifier
- **Model Evaluation**: Accuracy metrics, classification reports, and confusion matrix visualization

## 🎯 Features

- **Complete NLP Pipeline**: From raw text to sentiment prediction
- **Advanced Text Preprocessing**: Using NLTK and spaCy for robust text cleaning
- **TF-IDF Vectorization**: Efficient feature extraction from text data
- **Machine Learning Model**: Logistic Regression with optimized parameters
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Easy-to-Use Interface**: Simple functions for prediction and analysis

## 📊 Dataset

The project uses the IMDB dataset containing 50,000 movie reviews:
- **Size**: 50,000 reviews
- **Balance**: 25,000 positive and 25,000 negative reviews
- **Format**: CSV with 'review' and 'sentiment' columns
- **Source**: IMDB movie reviews

### Getting the Dataset

Due to GitHub's file size limitations, the dataset is not included in this repository. You can obtain the IMDB dataset from:

1. **Kaggle**: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. **Direct Download**: Download the CSV file and place it in the project directory as `IMDB_dataset.csv`

The dataset should have two columns:
- `review`: The movie review text
- `sentiment`: Either 'positive' or 'negative'

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone or download the project files**:
   ```bash
   # Make sure you have these files in your project directory:
   # - imdb_sentiment_analysis.py
   # - IMDB_dataset.csv
   # - requirements.txt
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model** (if not already downloaded):
   ```bash
   python -c "import spacy; spacy.cli.download('en_core_web_sm')"
   ```

## 🚀 Usage

### Running the Complete Pipeline

To run the entire sentiment analysis pipeline:

```bash
python imdb_sentiment_analysis.py
```

This will:
1. Load and explore the dataset
2. Preprocess the text data
3. Train the machine learning model
4. Evaluate model performance
5. Test predictions on sample reviews

### Using Individual Functions

You can also use the functions individually:

```python
import pandas as pd
from imdb_sentiment_analysis import *

# Load data
df = load_and_explore_data('IMDB_dataset.csv')

# Preprocess text
cleaned_text = preprocess_text("This movie was fantastic!")

# Prepare data for training
X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_data(df)

# Train model
model = train_model(X_train_tfidf, y_train)

# Make predictions
sentiment = predict_sentiment("I loved this movie!", model, tfidf)
print(sentiment)  # Output: Positive
```

## 📈 Model Performance

The trained model achieves:
- **Accuracy**: ~89.3%
- **Precision**: 89% for both positive and negative classes
- **Recall**: 90% for positive, 89% for negative
- **F1-Score**: 89% for both classes

## 🔧 Project Structure

```
imdb-sentiment-analysis/
├── imdb_sentiment_analysis.py    # Main implementation script
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── NLP.ipynb                    # Original Jupyter notebook
├── .gitignore                   # Git ignore rules
└── venv/                        # Virtual environment (created during setup)
```

## 📝 Code Structure

The main script (`imdb_sentiment_analysis.py`) is organized into several sections:

### 1. Imports and Setup
- Required libraries and NLTK/spaCy model downloads

### 2. Data Loading and Exploration
- `load_and_explore_data()`: Loads dataset and displays basic information

### 3. Text Preprocessing
- `get_stop_words()`: Retrieves English stop words
- `preprocess_text()`: Comprehensive text cleaning function
- `preprocess_dataset()`: Processes entire dataset

### 4. Model Training
- `prepare_data()`: Splits data and creates TF-IDF features
- `train_model()`: Trains Logistic Regression model

### 5. Prediction
- `predict_sentiment()`: Makes sentiment predictions on new text

### 6. Model Evaluation
- `evaluate_model()`: Calculates metrics and creates visualizations

## 🧪 Text Preprocessing Pipeline

The text preprocessing includes several steps:

1. **HTML Tag Removal**: Removes `<br>`, `<p>`, etc.
2. **Lowercasing**: Converts all text to lowercase
3. **Punctuation Removal**: Removes special characters and numbers
4. **Tokenization**: Splits text into individual words
5. **Stop Word Removal**: Removes common words like "the", "is", "at"
6. **Lemmatization**: Reduces words to their base form (e.g., "running" → "run")

## 🤖 Machine Learning Approach

### Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **Max Features**: 5,000 most common words
- **Vocabulary**: Built from training data only

### Model Selection
- **Algorithm**: Logistic Regression
- **Solver**: 'liblinear' (optimized for binary classification)
- **Regularization**: L2 regularization (default)

### Training Strategy
- **Train/Test Split**: 80%/20%
- **Stratification**: Maintains class balance in splits
- **Random State**: 42 (for reproducibility)

## 📊 Evaluation Metrics

The model evaluation includes:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions vs actual

## 🔍 Example Predictions

```python
# Positive review
text = "This movie was absolutely fantastic! The acting was superb and the plot was gripping."
sentiment = predict_sentiment(text, model, tfidf)
# Output: Positive

# Negative review
text = "I was so bored throughout the entire film. It was a complete waste of time and money."
sentiment = predict_sentiment(text, model, tfidf)
# Output: Negative
```

## 🛠️ Customization

### Modifying Preprocessing
You can customize the text preprocessing by modifying the `preprocess_text()` function:

```python
def preprocess_text(text):
    # Add your custom preprocessing steps here
    # ...
    return processed_text
```

### Changing Model Parameters
Modify the model training in `train_model()`:

```python
# Try different algorithms
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

# Or different TF-IDF parameters
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
```

### Adjusting Dataset Size
For faster testing, use a smaller sample:

```python
df_processed = preprocess_dataset(df, sample_size=1000)
```

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data not downloaded**:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

3. **Memory issues with large dataset**:
   - Use `sample_size` parameter in `preprocess_dataset()`
   - Reduce `max_features` in TF-IDF vectorizer

4. **File not found error**:
   - Ensure `IMDB_dataset.csv` is in the same directory as the script

## 📚 Dependencies

- **nltk**: Natural language processing toolkit
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **spacy**: Advanced NLP library
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **regex**: Regular expression operations

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the preprocessing pipeline
- Testing different machine learning algorithms
- Adding new evaluation metrics
- Optimizing performance
- Adding new features


---

**Note**: This project is designed for educational and research purposes. The model achieves good performance on the IMDB dataset but may need adjustments for other domains or datasets.
