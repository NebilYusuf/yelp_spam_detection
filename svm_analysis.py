from database_utils import get_db_credentials, create_db_engine
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import scipy.sparse as sp
from tqdm import tqdm
from joblib import Parallel, delayed, Memory
import multiprocessing
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import time
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.sparse import hstack
from sqlalchemy.exc import OperationalError
import os

# Caching setup
cachedir = './cachedir'
memory = Memory(location=cachedir, verbose=0)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading SpaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def process_fold(fold, X, y, train_idx, test_idx):
    if sp.issparse(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        scaler = StandardScaler(with_mean=False)
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        scaler = StandardScaler()

    y_train, y_test = y[train_idx], y[test_idx]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = LinearSVC(dual=False, max_iter=1000, class_weight='balanced')
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)

    return {
        'P': precision_score(y_test, y_pred, pos_label=1),
        'R': recall_score(y_test, y_pred, pos_label=1),
        'F1': f1_score(y_test, y_pred, pos_label=1),
        'A': accuracy_score(y_test, y_pred)
    }

@memory.cache
def extract_pos_sequences(review_text, batch_size):
    pos_sequences = []
    num_cores = max(2, multiprocessing.cpu_count() // 2)  # Avoid overwhelming the system
    spacy_batch_size = 32

    for i in tqdm(range(0, len(review_text), batch_size), desc="Extracting POS tags"):
        batch = review_text.iloc[i:i+batch_size].tolist()
        for doc in nlp.pipe(batch, disable=["ner", "parser"], batch_size=spacy_batch_size, n_process=num_cores):
            pos_sequence = " ".join([token.pos_ for token in doc])
            pos_sequences.append(pos_sequence)

    return pos_sequences

@memory.cache
def cached_get_review_data(engine_url, feature_type, batch_size=20000):
    engine = create_db_engine_from_url(engine_url)
    return get_review_data(engine, feature_type, batch_size)

def create_db_engine_from_url(url):
    from sqlalchemy import create_engine
    return create_engine(url)

def get_review_data(engine, feature_type, batch_size=20000):
    try:
        print("\nFetching reviews from database...")
        start_time = time.time()

        count_query = """
        SELECT COUNT(*) as total
        FROM review r
        WHERE r.reviewID IS NOT NULL
        """
        total_rows = pd.read_sql(count_query, engine).iloc[0]['total']

        base_query = """
        SELECT 
            r.reviewID,
            r.reviewContent,
            CASE 
                WHEN r.flagged IN ('Y', 'YR') THEN 1
                ELSE 0
            END as is_fake,
            rf.Review_Body
        FROM review r
        LEFT JOIN review_features rf ON r.reviewID = rf.Review_ID
        WHERE r.reviewID IS NOT NULL
        """

        chunks = []
        with tqdm(total=total_rows, desc="Loading reviews") as pbar:
            for chunk in pd.read_sql(base_query, engine, chunksize=batch_size):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks)

        print(f"Retrieved {len(df)} reviews in {time.time() - start_time:.2f} seconds")
        print(f"Fake reviews: {df['is_fake'].sum()}, Non-fake reviews: {len(df) - df['is_fake'].sum()}")

        review_text = df['Review_Body'].fillna(df['reviewContent'])

        print("\nExtracting features...")
        start_time = time.time()

        if feature_type == "WU+POS":
            print("Using combined Word Unigrams + POS features")
            wu_vectorizer = CountVectorizer(min_df=5, max_features=10000, binary=True, dtype=np.float32, lowercase=True, stop_words='english')
            wu_features = wu_vectorizer.fit_transform(review_text)

            pos_sequences = extract_pos_sequences(review_text, batch_size)
            pos_vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=5000, binary=True, dtype=np.float32)
            X_raw = pos_vectorizer.fit_transform(pos_sequences)
            k = min(1000, X_raw.shape[1])
            selector = SelectKBest(mutual_info_classif, k=k)
            pos_features = selector.fit_transform(X_raw, df['is_fake'].values)

            features = hstack([wu_features, pos_features])
            print(f"Combined features: {features.shape[1]} total")

        elif feature_type == "WU":
            print("Using Word Unigrams feature set")
            vectorizer = CountVectorizer(min_df=5, max_features=10000, binary=True, dtype=np.float32, lowercase=True, stop_words='english')
            features = vectorizer.fit_transform(review_text)
            print(f"Generated {features.shape[1]} features")

        elif feature_type == "POS":
            print("Using POS tag feature set (SpaCy + POS bigrams)")
            pos_sequences = extract_pos_sequences(review_text, batch_size)
            vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=5000, binary=True, dtype=np.float32)
            X_raw = vectorizer.fit_transform(pos_sequences)
            k = min(1000, X_raw.shape[1])
            selector = SelectKBest(mutual_info_classif, k=k)
            features = selector.fit_transform(X_raw, df['is_fake'].values)
            print(f"Selected top {features.shape[1]} POS features")

        else:
            raise ValueError(f"Feature type {feature_type} not implemented for testing")

        print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
        return features, df['is_fake'].values
    except OperationalError as e:
        print(f"Database error: {str(e)}\nRecreating engine...")
        engine.dispose()
        raise

def evaluate_model(X, y):
    print("Performing 5-fold cross-validation...")
    start_time = time.time()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    fold_indices = list(enumerate(skf.split(X, y), 1))

    with tqdm(total=len(fold_indices), desc="Processing folds") as pbar:
        fold_results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(process_fold)(fold, X, y, train_idx, test_idx)
            for fold, (train_idx, test_idx) in fold_indices
        )
        for _ in fold_results:
            pbar.update(1)

    metrics = {'P': [], 'R': [], 'F1': [], 'A': []}
    for result in fold_results:
        for k, v in result.items():
            metrics[k].append(v)

    print(f"Cross-validation completed in {time.time() - start_time:.2f} seconds")
    return {k: np.mean(v) * 100 for k, v in metrics.items()}

def evaluate_boosting_models(X, y):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   subsample=0.8, colsample_bytree=0.8, force_col_wise=True)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'P': precision_score(y_test, y_pred, pos_label=1) * 100,
            'R': recall_score(y_test, y_pred, pos_label=1) * 100,
            'F1': f1_score(y_test, y_pred, pos_label=1) * 100,
            'A': accuracy_score(y_test, y_pred) * 100
        }

    return results

def analyze_features():
    total_start_time = time.time()
    host = "database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com"

    print("Please enter your database credentials:")
    user, password = get_db_credentials()
    print("Connecting to databases...")

    hotel_url = f"mysql+pymysql://{user}:{password}@{host}/yelp_hotel"
    rest_url = f"mysql+pymysql://{user}:{password}@{host}/yelp_res"

    feature_sets = [
        ("Word unigrams (WU)", "WU"),
        ("POS tags (POS)", "POS"),
        ("WU + POS", "WU+POS")
    ]

    results = []
    for feature_name, feature_type in feature_sets:
        try:
            print(f"Processing {feature_name} for hotels...")
            hotel_features, hotel_labels = cached_get_review_data(hotel_url, feature_type)
            hotel_svm = evaluate_model(hotel_features, hotel_labels)
            hotel_boost = evaluate_boosting_models(hotel_features, hotel_labels)

            print(f"Processing {feature_name} for restaurants...")
            rest_features, rest_labels = cached_get_review_data(rest_url, feature_type)
            rest_svm = evaluate_model(rest_features, rest_labels)
            rest_boost = evaluate_boosting_models(rest_features, rest_labels)

            results.append({
                'feature_name': feature_name,
                'hotel_metrics': {
                    'SVM': hotel_svm,
                    'XGBoost': hotel_boost['XGBoost'],
                    'LightGBM': hotel_boost['LightGBM']
                },
                'rest_metrics': {
                    'SVM': rest_svm,
                    'XGBoost': rest_boost['XGBoost'],
                    'LightGBM': rest_boost['LightGBM']
                }
            })

        except Exception as e:
            print(f"Error processing {feature_name}: {str(e)}")
            continue

    print("Table 2: Model Comparison (SVM vs Boosting) ")
    print("-" * 120)
    print(f"{'Features':<20} {'Model':<10} {'P':>6} {'R':>6} {'F1':>6} {'A':>6} | {'P':>6} {'R':>6} {'F1':>6} {'A':>6}")
    print(" " * 30 + "(a): Hotel" + " " * 20 + "(b): Restaurant")
    print("-" * 120)

    for result in results:
        for model_name in ['SVM', 'XGBoost', 'LightGBM']:
            h = result['hotel_metrics'][model_name]
            r = result['rest_metrics'][model_name]
            print(f"{result['feature_name']:<20} {model_name:<10} "
                  f"{h['P']:>6.1f} {h['R']:>6.1f} {h['F1']:>6.1f} {h['A']:>6.1f} | "
                  f"{r['P']:>6.1f} {r['R']:>6.1f} {r['F1']:>6.1f} {r['A']:>6.1f}")
    print("-" * 120)

    total_time = time.time() - total_start_time
    print(f"Total analysis completed in {total_time/60:.1f} minutes")

if __name__ == "__main__":
    analyze_features()
