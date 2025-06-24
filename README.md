# Machine Learning Model

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: Gattadi Manikanta

*INTERN ID*: CT04DM1361

*DOMAIN*: Python Programming

*DURATION*: 4 Weeks

*MENTOR*: NEELA SANTOSH

# Predictive Model using Scikit-learn

## Overview

In this project, we have created a predictive machine learning model using the scikit-learn library. The goal is to classify or predict outcomes based on a dataset, showcasing model implementation and evaluation within a Jupyter Notebook.

## Objective

The main objective of this task is to demonstrate the ability to build and evaluate a machine learning classification model. We selected a spam email detection problem as the use case, where the model determines whether a given email message is spam or not based on the features in the dataset.

## Tools and Technologies Used

- **Python**: Programming language used for building the model.
- **Jupyter Notebook**: Interactive environment for coding and visualization.
- **Scikit-learn**: Machine learning library used for model implementation.
- **Pandas**: For data handling and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib / Seaborn** *(optional)*: For visualizations.
- **Multinomial Naive Bayes**: Chosen classification algorithm for spam detection.

## Dataset Description

The dataset used consists of text-based email data labeled as "spam" or "ham" (not spam). Each record contains a message and a label. Text preprocessing and transformation are applied to convert the raw text into numerical form suitable for machine learning models.

## Data Preprocessing

Before feeding the data into the model, preprocessing is essential:
- **Label Encoding**: Converts text labels ("spam", "ham") into numerical values.
- **Train-Test Split**: Splits the dataset into 80% for training and 20% for testing.
- **Text Vectorization**: Using `TfidfVectorizer` to convert text into feature vectors.

## Model Building

For this classification task, the **Multinomial Naive Bayes** algorithm was used. It is a popular choice for text classification problems due to its simplicity and efficiency. The model is trained on the training set and then tested on the unseen test data to evaluate its performance.

python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)

## How to Run

Follow these steps to run the project locally:

1. **Clone the Repository**:
   git clone https://github.com/manikanta9356/MACHINE-LEARNING-MODEL-IMPLEMETATION

2. **Install Required Libraries**:
   pip install numpy pandas scikit-learn matplotlib seaborn

3. **Launch Jupyter Notebook**:
   jupyter notebook
   Then open the Predictive Model.ipynb file.

4. **Run the Notebook**:
   - Run each cell in order.
   - The notebook will preprocess the data, train the model, and display evaluation results.
