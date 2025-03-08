from database_utils import get_db_credentials, fetch_data_from_databases

def main():
    """Main function to orchestrate database connection and data fetching."""
    host = "database-ml2025.cvk6uwuwc2j4.us-east-2.rds.amazonaws.com"
    databases = ["yelp_hotel", "yelp_res"]  # List of databases

    user, password = get_db_credentials()
    db_data = fetch_data_from_databases(user, password, host, databases)

    # Example: Access a table's DataFrame
    # print(db_data["yelp_hotel"]["some_table"].head())

if __name__ == "__main__":
    main()
