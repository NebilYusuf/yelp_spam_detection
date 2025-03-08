from sqlalchemy import create_engine
import pandas as pd
import getpass  # Securely get user input

def get_db_credentials():
    """Prompt user for MySQL credentials securely."""
    user = input("Enter MySQL username: ")
    password = getpass.getpass("Enter MySQL password: ")  # Hides password input
    return user, password

def create_db_engine(user, password, host, db_name):
    """Create a SQLAlchemy engine for a specific database."""
    return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")

def get_table_names(engine):
    """Fetch table names from the connected database."""
    query = "SHOW TABLES;"
    tables = pd.read_sql(query, engine)
    return tables.iloc[:, 0].tolist()  # Extract table names as a list

def fetch_table_data(engine, table_name):
    """Fetch data from a specific table."""
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql(query, engine)

def fetch_data_from_databases(user, password, host, databases):
    """Fetch data from multiple databases and store in a nested dictionary."""
    db_data = {}

    for db in databases:
        print(f"\n🔗 Connecting to {db}...")
        engine = create_db_engine(user, password, host, db)

        tables = get_table_names(engine)
        print(f"📌 Tables in {db}: {tables}")

        db_data[db] = {table: fetch_table_data(engine, table) for table in tables}
    
    print("\n✅ Data fetching complete!")
    return db_data 
