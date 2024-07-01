import pandas as pd
import joblib

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_new_data(df, scaler, feature_names):
    # Ensure the new data has the same features as the training data
    df = df[feature_names]
    df = df.dropna()
    X_scaled = scaler.transform(df)
    # Convert back to DataFrame to set feature names
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    return X_scaled_df

if __name__ == '__main__':
    # Load the new data
    new_data_path = 'data/external/alpha_vantage_data.csv'
    df = load_data(new_data_path)
    
    # Load the scaler and model
    scaler_path = 'models/scaler.pkl'
    scaler = joblib.load(scaler_path)
    
    model_path = 'models/model.pkl'
    model = joblib.load(model_path)
    
    # Define the feature names used during training
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                     'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 
                     'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
                     'V26', 'V27', 'V28', 'Amount']
    
    # Preprocess the new data
    X_scaled_df = preprocess_new_data(df, scaler, feature_names)
    
    # Make predictions
    predictions = model.predict(X_scaled_df)
    print(predictions)
