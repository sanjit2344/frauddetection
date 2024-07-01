from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return "Fraud Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    df = pd.DataFrame(data)  # Convert data to DataFrame
    X_scaled = scaler.transform(df)  # Scale the data
    predictions = model.predict(X_scaled)  # Predict using the model
    return jsonify({'predictions': predictions.tolist()})  # Return the predictions as JSON

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
