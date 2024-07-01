import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(filepath):
    """
    Load the preprocessed data from the specified filepath.
    """
    return pd.read_csv(filepath)

def train_model(X, y):
    """
    Train a RandomForest model on the provided features and target.
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

if __name__ == '__main__':
    # Load the preprocessed data
    processed_data_path = 'data/processed/scaled_data.csv'
    df = load_data(processed_data_path)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Save the trained model
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
