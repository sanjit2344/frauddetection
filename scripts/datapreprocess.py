import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath):
    """
    Load the dataset from the specified filepath.
    """
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values and scaling features.
    """
    # Example data cleaning step: Drop rows with missing values
    df = df.dropna()

    # Separate features and target variable
    X = df.drop('Class', axis=1)  # Assuming 'Class' is the target variable
    y = df['Class']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == '__main__':
    # Load the raw data
    raw_data_path = 'data/raw/creditcard.csv'
    df = load_data(raw_data_path)

    # Preprocess the data
    X_scaled, y, scaler = preprocess_data(df)

    # Save the scaled data and target variable
    processed_data_path = 'data/processed/scaled_data.csv'
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed_df = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    processed_df['Class'] = y.values
    processed_df.to_csv(processed_data_path, index=False)

    # Save the scaler for future use
    scaler_path = 'models/scaler.pkl'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
