# Spam Detection

## Overview
This project classifies text messages as **spam** or **not spam** (ham) using machine learning. It preprocesses text (cleaning, stemming), extracts TF-IDF features, and trains multiple models. The best model, Support Vector Classifier (SVC), achieves an accuracy of ~98.26% and precision of ~97.62%.

## Dataset
- Source: [Email Spam Detection Dataset](https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification/data) (Kaggle)
- License: Check Kaggle dataset page for details
- Description: 5,572 messages (5,169 after deduplication), with labels (`ham` or `spam`) and text content. Features: `v1` (label), `v2` (text).

## Installation
### Pip
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost joblib
```

### NLTK Data
```bash
python -m nltk.downloader stopwords
```

## Usage
1. Place `spam.csv` in `data/raw/`.
2. Run `spam_detection.ipynb` in Jupyter Notebook to preprocess data, train models, and save them to `spam_detection/model/`.
3. View model performance (accuracy, precision) in the notebook output.

## Models
- Support Vector Classifier (SVC): Best performer (accuracy: 98.26%, precision: 97.62%)
- Others: Logistic Regression, Random Forest, Multinomial Naive Bayes, XGBoost, etc.

## Notes
- Fix typo in notebook: `nltk.download('stopwords')` (not `stopswords`).
- Stopword removal is incomplete; consider adding to preprocessing.
- Models saved as `.joblib` files for reuse.