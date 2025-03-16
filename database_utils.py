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
        print(f"\nConnecting to {db}...")

        # Create SQLAlchemy engine for the database
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")

        # Get all table names from the database
        query_tables = "SHOW TABLES;"
        tables = pd.read_sql(query_tables, engine)

        # Convert to a list of table names
        table_list = tables.iloc[:, 0].tolist()  # First column has table names
        print(f"Tables in {db}: {table_list}")

        # Store data for each table in a dictionary
        db_data[db] = {}  # Create a sub-dictionary for each database

        # Loop through each table and load data
        for table in table_list:
            print(f"Fetching data from {table}...")
            query = f"SELECT * FROM {table};"
            df = pd.read_sql(query, engine)

            # Store the DataFrame in the dictionary
            db_data[db][table] = df
    print("\n✅ Data fetching complete!")
    return db_data 

def get_database_schema(user, password, host, databases):
    """
    Get schema information for multiple databases including tables and columns.
    
    Args:
        user (str): MySQL username
        password (str): MySQL password
        host (str): Database host
        databases (list): List of database names
        
    Returns:
        dict: Nested dictionary containing schema information for each database
    """
    schema_info = {}
    
    for db in databases:
        print(f"\nAnalyzing schema for database: {db}")
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db}")
        
        # Get all tables in the database
        tables = pd.read_sql("SHOW TABLES;", engine)
        table_list = tables.iloc[:, 0].tolist()
        
        schema_info[db] = {}
        
        # Get column information for each table
        for table in table_list:
            print(f"Getting column info for table: {table}")
            # SHOW COLUMNS provides detailed column information
            columns_df = pd.read_sql(f"SHOW COLUMNS FROM {table};", engine)
            
            # Store column information as a dictionary
            columns_info = columns_df.set_index('Field')[['Type', 'Null', 'Key', 'Default']].to_dict('index')
            schema_info[db][table] = columns_info
            
    print("\n✅ Schema analysis complete!")
    return schema_info

def print_schema_info(schema_info):
    """
    Print the schema information in a readable format.
    
    Args:
        schema_info (dict): Schema information from get_database_schema
    """
    for db_name, db_info in schema_info.items():
        print(f"\n📁 Database: {db_name}")
        
        for table_name, table_info in db_info.items():
            print(f"\n  📋 Table: {table_name}")
            
            print("    Columns:")
            for column_name, column_info in table_info.items():
                null_status = "NULL" if column_info['Null'] == 'YES' else "NOT NULL"
                key_info = f", {column_info['Key']}" if column_info['Key'] else ""
                default = f", Default: {column_info['Default']}" if column_info['Default'] else ""
                
                print(f"    - {column_name}: {column_info['Type']} {null_status}{key_info}{default}") 
