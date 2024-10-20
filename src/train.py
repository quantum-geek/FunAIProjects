import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.preprocess import load_data, preprocess_reviews, convert_ratings_to_labels, vectorize_text


def balance_data(data):
    """
    Balances the dataset by oversampling the minority classes (neutral and negative).
    """
    # Separate the data by class
    positive = data[data['Sentiment'] == 'positive']
    neutral = data[data['Sentiment'] == 'neutral']
    negative = data[data['Sentiment'] == 'negative']
    

    # Oversample the neutral and negative classes
    neutral_upsampled = resample(neutral, replace=True, n_samples=len(positive), random_state=42)
    negative_upsampled = resample(negative, replace=True, n_samples=len(positive), random_state=42)

    # Combine the oversampled data with the positive class
    balanced_data = pd.concat([positive, neutral_upsampled, negative_upsampled])

    return balanced_data

def train_model(file_path):
    """
    Train a logistic regression model on the balanced data.
    """
    # Load and preprocess the dataset
    data = load_data(file_path)
    data = preprocess_reviews(data)
    data = convert_ratings_to_labels(data)

    # Balance the dataset
    balanced_data = balance_data(data)

    # Split the data into features (X) and target (y)
    X = balanced_data['CleanedText']
    y = balanced_data['Sentiment']

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=2000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    return model, vectorizer
