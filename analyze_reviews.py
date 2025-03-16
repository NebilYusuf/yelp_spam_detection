from database_utils import get_db_credentials, create_db_engine
import pandas as pd

def analyze_review_reliability():
    """Analyze fake and non-fake reviews statistics as shown in Table 1."""
    host = "database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com"
    
    # Get database credentials
    print("Please enter your database credentials:")
    user, password = get_db_credentials()
    
    # Create engines for both databases
    hotel_engine = create_db_engine(user, password, host, "yelp_hotel")
    restaurant_engine = create_db_engine(user, password, host, "yelp_res")
    
    # Query to get statistics matching Table 1
    stats_query = """
    SELECT 
        SUM(CASE WHEN flagged IN ('Y', 'YR') THEN 1 ELSE 0 END) as fake,
        SUM(CASE WHEN flagged IN ('N', 'NR') THEN 1 ELSE 0 END) as non_fake,
        ROUND(100.0 * SUM(CASE WHEN flagged IN ('Y', 'YR') THEN 1 ELSE 0 END) / COUNT(*), 1) as percent_fake,
        COUNT(*) as total_reviews,
        COUNT(DISTINCT reviewerID) as num_reviewers
    FROM review;
    """
    
    # Create results table matching the paper's format
    print("\nTable 1: Dataset statistics")
    print("-" * 70)
    print(f"{'Domain':<12} {'fake':>6} {'non-fake':>10} {'% fake':>8} {'total # reviews':>16} {'# reviewers':>12}")
    print("-" * 70)
    
    # Get and print hotel statistics
    hotel_stats = pd.read_sql(stats_query, hotel_engine).iloc[0]
    print(f"{'Hotel':<12} {hotel_stats['fake']:>6} {hotel_stats['non_fake']:>10} {hotel_stats['percent_fake']:>7.1f}% {hotel_stats['total_reviews']:>15} {hotel_stats['num_reviewers']:>12}")
    
    # Get and print restaurant statistics
    restaurant_stats = pd.read_sql(stats_query, restaurant_engine).iloc[0]
    print(f"{'Restaurant':<12} {restaurant_stats['fake']:>6} {restaurant_stats['non_fake']:>10} {restaurant_stats['percent_fake']:>7.1f}% {restaurant_stats['total_reviews']:>15} {restaurant_stats['num_reviewers']:>12}")
    print("-" * 70)

if __name__ == "__main__":
    analyze_review_reliability() 