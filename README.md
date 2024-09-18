# Spam Detection Using Random Forest Classifier

This project implements a spam detection system using a Random Forest Classifier. The model is trained on a dataset of emails labeled as spam or ham (not spam) and can classify new emails based on their content.

## Features

- Preprocesses email text by removing punctuation, converting to lowercase, and stemming words.
- Uses a Count Vectorizer to convert text data into numerical features.
- Trains a Random Forest Classifier to classify emails as spam or ham.
- Evaluates the model's performance on a test set.
- Predicts the label of a new email.

## Requirements

- Python 3.x
- `numpy` (`pip install numpy`)
- `pandas` (`pip install pandas`)
- `nltk` (`pip install nltk`)
- `scikit-learn` (`pip install scikit-learn`)

## Usage

1. **Clone the repository** (if applicable) or download the script file.

2. **Install the required modules**:
   ```bash
   pip install numpy pandas nltk scikit-learn
   ```

3. **Download the NLTK stopwords**:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Prepare the dataset**:
   - Ensure you have the `spam_ham_dataset.csv` file in the same directory as the script.

5. **Run the script**:
   ```bash
   python script_name.py
   ```

## How It Works

1. **Data Preprocessing**:
   - The script reads the dataset and replaces newline characters in the email text.
   - It converts the text to lowercase, removes punctuation, and stems the words using the Porter Stemmer.
   - Stopwords are removed from the text.

2. **Feature Extraction**:
   - The `CountVectorizer` converts the preprocessed text into numerical features.

3. **Model Training**:
   - The dataset is split into training and testing sets.
   - A `RandomForestClassifier` is trained on the training set.

4. **Model Evaluation**:
   - The model's performance is evaluated on the test set using the `score` method.

5. **Email Classification**:
   - The script demonstrates how to classify a new email by preprocessing the text and using the trained model to predict its label.

## Notes

- Ensure that the `spam_ham_dataset.csv` file is correctly formatted and accessible.
- You can modify the script to classify different emails by changing the `email_to_classify` variable.
