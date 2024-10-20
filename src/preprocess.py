import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


'''
This code performs basic preprocessing tasks like:

Lowercasing the text.
Removing punctuation.
Removing stopwords (common words like "and", "the", "is" that don’t add much meaning).
Tokenizing (breaking the text into individual words).

'''

nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_text(text):
    """
    Clean the text data by:
    - Lowercasing
    - Removing punctuation
    - Removing stopwords
    - Tokenizing the text
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return ' '.join(words)

def preprocess_reviews(data):
    """
    Preprocess the 'Text' column in the dataset.
    """
    data['CleanedText'] = data['Text'].apply(preprocess_text)
    return data

'''
This function will map the Score column to sentiment labels (negative, neutral, positive).
'''
def convert_ratings_to_labels(data):
    """
    Convert numerical ratings to sentiment labels:
    - 1, 2 -> Negative
    - 3 -> Neutral
    - 4, 5 -> Positive
    """
    def label_rating(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    data['Sentiment'] = data['Score'].apply(label_rating)
    return data
