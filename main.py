from src.train import train_model

# File path to the dataset
file_path = 'data/Reviews.csv'

# Train the model (preprocessing and balancing are handled inside train_model)
model, vectorizer = train_model(file_path)

# Optional: You can print some information about the model or save it for later use
print("Model training completed.")
