import pandas as pd
import sys

def inspect_parquet(file_path):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return

    print("--- DataFrame Info ---")
    print(df.info())
    print("\n--- Time Range ---")
    
    # Check if 'date' is a column or the index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index
    else:
        print("No 'date' column or DatetimeIndex found. Columns:", df.columns)
        print("Head:", df.head())
        return

    print(f"Start: {df['date'].min()}")
    print(f"End:   {df['date'].max()}")
    print(f"Duration: {df['date'].max() - df['date'].min()}")

    print("\n--- Head ---")
    print(df.head())
    print("\n--- Tail ---")
    print(df.tail())
    
    print("\n--- Row Count by Month (approx) ---")
    df['month'] = df['date'].dt.to_period('M')
    print(df.groupby('month').size())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_parquet(sys.argv[1])
    else:
        print("Usage: python inspect_parquet.py <path_to_parquet>")
