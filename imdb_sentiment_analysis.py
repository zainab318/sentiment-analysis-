#!/usr/bin/env python3
"""
IMDB Sentiment Analysis - Complete Implementation

This script implements a complete sentiment analysis pipeline for IMDB movie reviews
using Natural Language Processing (NLP) and Machine Learning techniques.

Author: [Your Name]
Date: [Current Date]
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

import nltk
import pandas as pd
import re
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Download and load spaCy model
print("Downloading spaCy model...")
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# =============================================================================
# DATA LOADING AND EXPLORATION
# =============================================================================

def load_and_explore_data(file_path='IMDB_dataset.csv'):
    """
    Load the IMDB dataset and display basic information.
    
    Args:
        file_path (str): Path to the CSV file containing the dataset
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    # Set pandas to display full text for better inspection
    pd.set_option('display.max_colwidth', None)
    
    # Load the dataset from the CSV file
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found. Make sure it's in the same folder as this script.")
        return None
    
    # Display the first 5 rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Display dataset information
    print("\nDataset Information:")
    print(df.info())
    
    # Display sentiment distribution
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    
    return df

# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def get_stop_words():
    """
    Get the list of English stop words.
    
    Returns:
        set: Set of English stop words
    """
    return set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text string.
    
    Steps:
    1. Removes HTML tags
    2. Lowercases text
    3. Removes punctuation and numbers
    4. Tokenizes text
    5. Removes stop words
    6. Lemmatizes words
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = get_stop_words()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization using spaCy
    text = ' '.join(tokens)
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    return ' '.join(lemmatized_tokens)

def preprocess_dataset(df, sample_size=None):
    """
    Preprocess the entire dataset.
    
    Args:
        df (pandas.DataFrame): Original dataset
        sample_size (int, optional): Number of samples to process (for testing)
        
    Returns:
        pandas.DataFrame: Dataset with cleaned reviews
    """
    if sample_size:
        df = df.sample(sample_size).copy()
        print(f"Preprocessing {sample_size} reviews...")
    else:
        print("Preprocessing all reviews...")
    
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    print("Preprocessing complete!")
    
    # Show comparison of original and cleaned reviews
    print("\nComparing original and cleaned reviews:")
    print(df[['review', 'cleaned_review']].head())
    
    return df

# =============================================================================
# MODEL TRAINING
# =============================================================================

def prepare_data(df):
    """
    Prepare data for model training.
    
    Args:
        df (pandas.DataFrame): Dataset with reviews and sentiments
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, tfidf_vectorizer)
    """
    # Map sentiment labels to 0 and 1
    df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Define features (X) and target (y)
    X = df['review']
    y = df['sentiment_numeric']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Initialize and fit TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"Shape of the training TF-IDF matrix: {X_train_tfidf.shape}")
    print(f"Shape of the testing TF-IDF matrix: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf

def train_model(X_train_tfidf, y_train):
    """
    Train the Logistic Regression model.
    
    Args:
        X_train_tfidf: TF-IDF features for training
        y_train: Target labels for training
        
    Returns:
        LogisticRegression: Trained model
    """
    # Initialize and train the model
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_tfidf, y_train)
    
    print("Model training complete!")
    return model

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================

def predict_sentiment(text, model, tfidf_vectorizer):
    """
    Takes a raw text string and predicts its sentiment using the trained model.
    
    Args:
        text (str): Raw text to analyze
        model: Trained model
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        
    Returns:
        str: 'Positive' or 'Negative'
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Vectorize the text using the fitted TF-IDF vectorizer
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    
    # Predict using the trained model
    prediction = model.predict(vectorized_text)
    
    # Return the human-readable result
    return 'Positive' if prediction[0] == 1 else 'Negative'

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the trained model and display results.
    
    Args:
        model: Trained model
        X_test_tfidf: TF-IDF features for testing
        y_test: True labels for testing
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Print the detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete sentiment analysis pipeline.
    """
    print("=" * 60)
    print("IMDB SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and explore data
    print("\n1. Loading and exploring data...")
    df = load_and_explore_data()
    
    if df is None:
        return
    
    # Step 2: Preprocess data (using a sample for demonstration)
    print("\n2. Preprocessing data...")
    df_processed = preprocess_dataset(df, sample_size=500)
    
    # Step 3: Prepare data for training
    print("\n3. Preparing data for training...")
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = prepare_data(df)
    
    # Step 4: Train model
    print("\n4. Training model...")
    model = train_model(X_train_tfidf, y_train)
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model...")
    evaluate_model(model, X_test_tfidf, y_test)
    
    # Step 6: Test predictions
    print("\n6. Testing predictions...")
    test_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was gripping.",
        "I was so bored throughout the entire film. It was a complete waste of time and money.",
        "The film was okay, not great but not terrible either. Some parts were good."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        sentiment = predict_sentiment(review, model, tfidf)
        print(f"Review {i}: '{review}'")
        print(f"Predicted Sentiment: {sentiment}\n")
    
    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main() 