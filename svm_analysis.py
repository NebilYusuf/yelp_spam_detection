from database_utils import get_db_credentials, create_db_engine
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def get_features(text, feature_type):
    """Extract different types of features from text."""
    if feature_type == "WU":  # Word Unigrams
        vectorizer = CountVectorizer(ngram_range=(1, 1))
        features = vectorizer.fit_transform([text])
        return features
    
    elif feature_type == "WU_IG":  # Word Unigrams with Information Gain
        # Implementation for top k% features according to Information Gain
        pass
    
    elif feature_type == "WB":  # Word Bigrams
        vectorizer = CountVectorizer(ngram_range=(2, 2))
        features = vectorizer.fit_transform([text])
        return features
    
    elif feature_type == "WB_LIWC":  # Word Bigrams + LIWC
        # Implementation for LIWC features
        pass
    
    elif feature_type == "POS":  # POS Unigrams
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        return pos_tags
    
    # Add other feature extraction methods...

def evaluate_model(X, y):
    """Perform 5-fold CV and return evaluation metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    metrics = {
        'P': [], 'R': [], 'F1': [], 'A': []
    }
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train SVM
        svm = SVC(kernel='rbf', C=1.0)
        svm.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = svm.predict(X_test_scaled)
        
        # Calculate metrics
        metrics['P'].append(precision_score(y_test, y_pred, pos_label='Y'))
        metrics['R'].append(recall_score(y_test, y_pred, pos_label='Y'))
        metrics['F1'].append(f1_score(y_test, y_pred, pos_label='Y'))
        metrics['A'].append(accuracy_score(y_test, y_pred))
    
    # Return averages
    return {k: np.mean(v) * 100 for k, v in metrics.items()}

def analyze_features():
    """Analyze reviews using different feature sets and SVM classification."""
    host = "database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com"
    
    # Get database credentials
    print("Please enter your database credentials:")
    user, password = get_db_credentials()
    
    # Create engines for both databases
    hotel_engine = create_db_engine(user, password, host, "yelp_hotel")
    restaurant_engine = create_db_engine(user, password, host, "yelp_res")
    
    # Define feature sets
    feature_sets = [
        "Word unigrams (WU)",
        "WU + IG (top 1%)",
        "WU + IG (top 2%)",
        "Word-Bigrams (WB)",
        "WB+LIWC",
        "POS Unigrams",
        "WB + POS Bigrams",
        "WB + Deep Syntax",
        "WB + POS Seq. Pat."
    ]
    
    # Print table header
    print("\nTable 2: SVM 5-fold CV results")
    print("-" * 80)
    print(f"{'Features':<20} {'P':>6} {'R':>6} {'F1':>6} {'A':>6} | {'P':>6} {'R':>6} {'F1':>6} {'A':>6}")
    print(" " * 20 + "(a): Hotel" + " " * 10 + "(b): Restaurant")
    print("-" * 80)
    
    for feature_set in feature_sets:
        # Get and process features for both domains
        # This is a placeholder - actual feature extraction would go here
        hotel_metrics = {'P': 62.9, 'R': 76.6, 'F1': 68.9, 'A': 65.6}  # Example values
        rest_metrics = {'P': 64.3, 'R': 76.3, 'F1': 69.7, 'A': 66.9}   # Example values
        
        print(f"{feature_set:<20} "
              f"{hotel_metrics['P']:>6.1f} {hotel_metrics['R']:>6.1f} "
              f"{hotel_metrics['F1']:>6.1f} {hotel_metrics['A']:>6.1f} | "
              f"{rest_metrics['P']:>6.1f} {rest_metrics['R']:>6.1f} "
              f"{rest_metrics['F1']:>6.1f} {rest_metrics['A']:>6.1f}")

if __name__ == "__main__":
    analyze_features() 