# Install necessary packages using:
# pip install pandas scikit-learn matplotlib flask

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(name)

# Sample data loading
def load_data(file_path):
    return pd.read_csv(file_path)

# Data preprocessing
def preprocess_data(df):
    # Simple preprocessing: remove rows with missing values
    df = df.dropna()
    return df

# Train and test predictive model
def train_model(df):
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target variable
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f'Model Accuracy: {accuracy:.2f}')
    return model

# Visualize data
def plot_data(df):
    df.plot(kind='scatter', x='feature1', y='target')  # Replace 'feature1' with an actual feature
    plt.show()

# Flask routes
@app.route('/')
def home():
    return "Welcome to the Smart City AI Solution!"

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    features = [content['feature1'], content['feature2']]  # Adjust based on actual features
    prediction = model.predict([features])
    return jsonify({'prediction': int(prediction[0])})

if name == "main":
    # Load and preprocess data
    data = load_data('data/data.csv')  # Ensure 'data.csv' exists and has appropriate columns
    clean_data = preprocess_data(data)
    
    # Train model
    model = train_model(clean_data)
    
    # Visualize data
    plot_data(clean_data)
    
    # Run Flask app
    app.run(debug=True)
