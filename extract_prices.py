import re
import pandas as pd
import matplotlib.pyplot as plt

log_file = 'logs/live_trading.log'

prices = []
# Pattern for price logs
# 2026-01-16 12:50:06 CST | ... Price: 6988.25
# 2026-01-12 07:20:01 CST | ... Price: 6968.75
price_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) CST.*Price: ([\d\.]+)')

with open(log_file, 'r') as f:
    for line in f:
        match = price_pattern.search(line)
        if match:
            ts_str, price_str = match.groups()
            prices.append({'timestamp': ts_str, 'price': float(price_str)})

df_prices = pd.DataFrame(prices)
if not df_prices.empty:
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp']) # These are CST
    # The trades in DB are UTC. 
    # CST is UTC-6. So 12:50 CST is 18:50 UTC.
    # I should convert price timestamps to UTC for consistency.
    # Assuming standard time (Jan), it is CST (UTC-6).
    df_prices['timestamp'] = df_prices['timestamp'] + pd.Timedelta(hours=6)
    
    df_prices = df_prices.sort_values('timestamp')
    print(df_prices.tail())
    print(f"Price range: {df_prices['price'].min()} - {df_prices['price'].max()}")
    
    # Save for plotting or analysis
    df_prices.to_csv('extracted_prices.csv', index=False)
else:
    print("No prices extracted.")
