from database_utils import get_db_credentials, create_db_engine
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC  # Changed to LinearSVC for faster training
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import scipy.sparse as sp
from tqdm import tqdm
import time
from joblib import Parallel, delayed
import multiprocessing
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def process_fold(fold, X, y, train_idx, test_idx):
    """Process a single cross-validation fold."""
    if sp.issparse(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        scaler = StandardScaler(with_mean=False)
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        scaler = StandardScaler()
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM (using LinearSVC for speed)
    svm = LinearSVC(dual=False, max_iter=1000)
    svm.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate metrics
    return {
        'P': precision_score(y_test, y_pred, pos_label=1),
        'R': recall_score(y_test, y_pred, pos_label=1),
        'F1': f1_score(y_test, y_pred, pos_label=1),
        'A': accuracy_score(y_test, y_pred)
    }

def get_review_data(engine, feature_type, batch_size=1000):
    """Fetch review data and features from database."""
    print("\nFetching reviews from database...")
    start_time = time.time()
    
    # First, count total rows for progress bar
    count_query = """
    SELECT COUNT(*) as total
    FROM review r
    WHERE r.reviewID IS NOT NULL
    """
    total_rows = pd.read_sql(count_query, engine).iloc[0]['total']
    
    # Base query to get reviews and their labels
    base_query = """
    SELECT 
        r.reviewID,
        r.reviewContent,
        CASE 
            WHEN r.flagged IN ('Y', 'YR') THEN 1
            ELSE 0
        END as is_fake,
        rf.Review_Body,
        rf.EXT as word_unigrams,
        rf.DEV as word_bigrams,
        rf.Filtered
    FROM review r
    LEFT JOIN review_features rf ON r.reviewID = rf.Review_ID
    WHERE r.reviewID IS NOT NULL
    """
    
    # Read data in chunks with progress bar
    chunks = []
    with tqdm(total=total_rows, desc="Loading reviews") as pbar:
        for chunk in pd.read_sql(base_query, engine, chunksize=batch_size):
            chunks.append(chunk)
            pbar.update(len(chunk))
    df = pd.concat(chunks)
    
    print(f"Retrieved {len(df)} reviews in {time.time() - start_time:.2f} seconds")
    print(f"Fake reviews: {df['is_fake'].sum()}, Non-fake reviews: {len(df) - df['is_fake'].sum()}")
    
    # Use Review_Body if available, otherwise use reviewContent
    review_text = df['Review_Body'].fillna(df['reviewContent'])
    
    print("\nExtracting features...")
    start_time = time.time()
    
    if feature_type == "WU":
        # Word Unigrams with optimized parameters
        print("Using Word Unigrams feature set")
        vectorizer = CountVectorizer(
            min_df=5,
            max_features=10000,  # Limit features for speed
            binary=True,  # Use binary features instead of counts
            dtype=np.float32  # Use float32 for memory efficiency
        )
        
        # Create progress bar for tokenization
        tokens = []
        with tqdm(total=len(review_text), desc="Tokenizing reviews") as pbar:
            for text in review_text:
                tokens.append(text)
                pbar.update(1)
        
        print("Vectorizing features...")
        features = vectorizer.fit_transform(tokens)
        print(f"Generated {features.shape[1]} features")
        
    else:
        raise ValueError(f"Feature type {feature_type} not implemented for testing")
    
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    return features, df['is_fake'].values

def evaluate_model(X, y):
    """Perform 5-fold CV and return evaluation metrics using parallel processing."""
    print("\nPerforming 5-fold cross-validation...")
    start_time = time.time()
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use all CPUs except one
    
    # Get all fold indices first
    fold_indices = list(enumerate(skf.split(X, y), 1))
    
    # Create progress bar for folds
    with tqdm(total=len(fold_indices), desc="Processing folds") as pbar:
        # Run folds in parallel with callback for progress
        def update_progress(*args):
            pbar.update(1)
        
        fold_results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(process_fold)(fold, X, y, train_idx, test_idx)
            for fold, (train_idx, test_idx) in fold_indices
        )
    
    # Aggregate results
    metrics = {
        'P': [], 'R': [], 'F1': [], 'A': []
    }
    for result in fold_results:
        for k, v in result.items():
            metrics[k].append(v)
    
    print(f"\nCross-validation completed in {time.time() - start_time:.2f} seconds")
    return {k: np.mean(v) * 100 for k, v in metrics.items()}

def analyze_features():
    """Analyze reviews using different feature sets and SVM classification."""
    total_start_time = time.time()
    
    host = "database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com"
    
    # Get database credentials
    print("Please enter your database credentials:")
    user, password = get_db_credentials()
    
    # Create engines for both databases
    print("\nConnecting to databases...")
    hotel_engine = create_db_engine(user, password, host, "yelp_hotel")
    restaurant_engine = create_db_engine(user, password, host, "yelp_res")
    
    # Test only Word Unigrams (WU) feature set
    feature_sets = [
        ("Word unigrams (WU)", "WU")
    ]
    
    # Store results for final table
    results = []
    
    for feature_name, feature_type in feature_sets:
        try:
            # Process hotel domain
            print(f"\nProcessing {feature_name} for hotels...")
            hotel_features, hotel_labels = get_review_data(hotel_engine, feature_type)
            hotel_metrics = evaluate_model(hotel_features, hotel_labels)
            
            # Process restaurant domain
            print(f"\nProcessing {feature_name} for restaurants...")
            rest_features, rest_labels = get_review_data(restaurant_engine, feature_type)
            rest_metrics = evaluate_model(rest_features, rest_labels)
            
            # Store results
            results.append({
                'feature_name': feature_name,
                'hotel_metrics': hotel_metrics,
                'rest_metrics': rest_metrics
            })
            
        except Exception as e:
            print(f"Error processing {feature_name}: {str(e)}")
            continue
    
    # Print complete table at the end
    print("\nTable 2: SVM 5-fold CV results")
    print("-" * 80)
    print(f"{'Features':<20} {'P':>6} {'R':>6} {'F1':>6} {'A':>6} | {'P':>6} {'R':>6} {'F1':>6} {'A':>6}")
    print(" " * 20 + "(a): Hotel" + " " * 10 + "(b): Restaurant")
    print("-" * 80)
    
    for result in results:
        print(f"{result['feature_name']:<20} "
              f"{result['hotel_metrics']['P']:>6.1f} {result['hotel_metrics']['R']:>6.1f} "
              f"{result['hotel_metrics']['F1']:>6.1f} {result['hotel_metrics']['A']:>6.1f} | "
              f"{result['rest_metrics']['P']:>6.1f} {result['rest_metrics']['R']:>6.1f} "
              f"{result['rest_metrics']['F1']:>6.1f} {result['rest_metrics']['A']:>6.1f}")
    print("-" * 80)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal analysis completed in {total_time/60:.1f} minutes")

if __name__ == "__main__":
    analyze_features() 