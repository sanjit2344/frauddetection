import os
import requests
import pandas as pd

def fetch_alpha_vantage_data(api_key, symbol, outputsize='compact'):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&outputsize={outputsize}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (1min)']).T
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # Ensure the new data has the required columns
    required_columns = ['Time','V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 
                        'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
                        'V26', 'V27', 'V28', 'Amount']
    
    # Placeholder for the required columns with some dummy data or logic to generate them
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # You can replace this with appropriate logic to generate the necessary data

    # Ensure columns are in the correct order
    df = df[required_columns]
    
    return df

api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
symbol = 'AAPL'
data = fetch_alpha_vantage_data(api_key, symbol)

# Ensure the directory exists
output_dir = 'data/external'
os.makedirs(output_dir, exist_ok=True)

data.to_csv(os.path.join(output_dir, 'alpha_vantage_data.csv'), index=False)
