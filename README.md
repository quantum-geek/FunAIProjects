# Sentiment Analysis for Product Reviews

This project aims to classify the sentiment of product reviews as **positive**, **neutral**, or **negative** using a machine learning model. We use text preprocessing techniques and a logistic regression classifier for sentiment analysis.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [How to Run](#how-to-run)
5. [Results](#results)
6. [Future Improvements](#future-improvements)

## Project Overview
The goal of this project is to build a machine learning model that can classify product reviews into positive, neutral, or negative categories based on their content. The dataset used is the **Amazon Fine Food Reviews** dataset.

The workflow consists of:
- **Data Preprocessing**: Cleaning and preparing text data (removing stop words, tokenizing, etc.).
- **Sentiment Labeling**: Converting review ratings into sentiment labels.
- **Model Training**: Training a logistic regression model on the processed text data.
- **Evaluation**: Evaluating the model’s performance using precision, recall, F1-score, and accuracy.

sentiment-analysis/
│
├── data/                         # Folder for storing dataset
│   └── Reviews.csv               # The Amazon Fine Food Reviews dataset
│
├── notebooks/                    # Jupyter notebooks for exploration
│   └── data_exploration.ipynb    # (Optional) Notebook for EDA and exploration
│
├── models/                       # Folder for storing saved models
│   └── sentiment_model.pkl       # Saved logistic regression model
│
├── src/                          # Folder for all source code
│   ├── __init__.py               # Makes src a package
│   ├── preprocess.py             # Code for preprocessing the dataset
│   ├── train.py                  # Code for training the model
│   └── evaluate.py               # Code for evaluating the model (optional)
│
├── outputs/                      # Outputs like evaluation reports
│   └── model_performance.txt     # Evaluation results of the trained model
│
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
└── main.py                       # Main script to run the project

## Setup Instructions
### 1. Clone the repository
git clone <repository-url>
cd sentiment-analysis
2. Install the required dependencies
pip install -r requirements.txt
3. Download the dataset
Download the Amazon Fine Food Reviews dataset from Kaggle here and place the Reviews.csv file inside the data/ folder.
4. Run the Project
You can run the project using the following command:
python main.py

Results
After running the model on 10,000 reviews, we achieved the following results:

Accuracy: 81%
Precision, Recall, F1-Score for Positive, Neutral, and Negative classes.
yaml
Copy code
              precision    recall  f1-score   support

    negative       0.71      0.42      0.53       321
     neutral       0.55      0.10      0.17       171
    positive       0.83      0.98      0.90      1508

    accuracy                           0.81      2000
   macro avg       0.69      0.50      0.53      2000
weighted avg       0.78      0.81      0.78      2000


Future Improvements
Handle Class Imbalance: Implement techniques such as oversampling, undersampling, or class weighting to improve performance on neutral and negative reviews.
Model Tuning: Experiment with hyperparameter tuning, n-grams in the TF-IDF vectorizer, or different machine learning models.
Expand Dataset: Use the full dataset and implement batch processing for scalability.
Incorporate Deep Learning: Try neural network models to improve sentiment classification accuracy.
python
Copy code

---

