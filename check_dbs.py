import sqlite3
import pandas as pd
import os

def check_db(db_path):
    if not os.path.exists(db_path):
        print(f"Database {db_path} does not exist.")
        return

    print(f"--- Checking {db_path} ---")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {tables}")
        
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(df.tail(5))
                print(f"Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading table {table_name}: {e}")
        conn.close()
    except Exception as e:
        print(f"Error connecting to {db_path}: {e}")

check_db('data/orders.db')
check_db('data/llm_trades.db')
