# 🕵️ Yelp Spam Detection using Temporal and Linguistic Features

This project implements a spam detection system for Yelp reviews by leveraging both temporal and linguistic features. It supports fetching real Yelp data from SQL databases and training multiple machine learning models to classify reviews as fake or genuine.

## 📁 Project Structure

- **Notebook**: `Yelp_Spam.ipynb` – main notebook for data extraction, processing, and model training.
- **Cache**: Data fetched from the databases can be optionally cached for reuse using pickle.
- **Logs**: Logging is configured to trace training duration and performance metrics.

## 🚀 Features

- Connects to remote MySQL databases and retrieves tables from:
  - `yelp_res` (Yelp restaurant data)
  - `yelp_hotel` (Yelp hotel data)
- Caches the dataset locally to avoid repeated fetching.
- Preprocesses review text using:
  - TF-IDF vectorization
  - Stopword removal
  - POS tagging (if used)
- Trains and evaluates multiple classifiers:
  - Logistic Regression
  - Naive Bayes
  - SVM (LinearSVC)
  - Decision Tree
  - Random Forest
  - XGBoost
  - StackingClassifier (ensemble)
- Reports precision, recall, F1-score, accuracy, and training time for each model.

## 🧠 Requirements

- Python 3.8+
- Packages:
  ```bash
  pip install pandas numpy scikit-learn xgboost nltk sqlalchemy pymysql
  ```
- NLTK data:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

## 🔐 Database Access

Make sure you have access to the following:
- Host: `database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com`
- Databases: `yelp_res`, `yelp_hotel`
- Credentials are requested securely during runtime using `getpass`.

## ⚙️ Usage

1. Open `Yelp_Spam.ipynb` in Jupyter or Colab.
2. Run the cells step by step.
3. Input your MySQL username and password when prompted.
4. The notebook will:
   - Fetch and cache data
   - Preprocess and vectorize reviews
   - Train multiple models
   - Output evaluation metrics

## 📊 Output

Each model outputs a table like:
| Model | Precision | Recall | F1 | Accuracy | Time (s) |
|-------|-----------|--------|----|----------|----------|

## 📌 Notes

- You can force a data refresh by passing `force_refresh=True` to `fetch_data_from_databases()`.
- The ensemble (StackingClassifier) generally gives the best performance but is slower to train.
