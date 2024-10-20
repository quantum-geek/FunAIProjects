from src.preprocess import load_data, preprocess_reviews, convert_ratings_to_labels
from src.train import train_model

# File path to the dataset
file_path = 'data/Reviews.csv'

# Load the dataset
data = load_data(file_path)

# Preprocess the text data
preprocessed_data = preprocess_reviews(data)

# Convert ratings to sentiment labels
preprocessed_data = convert_ratings_to_labels(preprocessed_data)


# Show the first few rows of the preprocessed data
#print(preprocessed_data[['Text', 'CleanedText']].head())


# Show the first few rows with the new 'Sentiment' column
print(preprocessed_data[['Score', 'Sentiment', 'CleanedText']].head())

# Train the model
model, vectorizer = train_model(preprocessed_data)