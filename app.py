import webbrowser
from flask import Flask, render_template, request, jsonify
from src.movie_rating_predictor import MovieRatingPredictor
import os
import threading
import time

app = Flask(__name__)
predictor = MovieRatingPredictor()

def open_browser():
    """Function to open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

# Load or train model when the app starts
print("\nLoading or training model...")
model = predictor.load_model()

if model is None:
    print("\nModel not found. Training new model...")
    # Load data
    df = predictor.load_data('data/IMDb Movies India.csv')
    # Prepare features
    X, y = predictor.prepare_features(df)
    # Train and save model
    model = predictor.train_model(X, y)
    predictor.save_model(model)
    print("\nModel training completed and saved")
else:
    print("Model loaded successfully")

# Load or train model when the app starts
try:
    print("\nLoading model...")
    predictor.load_model()
    print("Model loaded successfully")
except FileNotFoundError:
    print("\nModel not found. Training new model...")
    # Load data
    df = predictor.load_data('data/IMDb Movies India.csv')
    # Prepare features
    X, y = predictor.prepare_features(df)
    # Train and save model
    model = predictor.train_model(X, y)
    predictor.save_model(model)
    print("\nModel training completed and saved")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        # Prepare movie data
        movie_data = {
            'Year': int(data.get('year', 2023)),
            'Duration': int(data.get('duration', 120)),
            'Genre': data.get('genre', 'Unknown'),
            'Director': data.get('director', 'Unknown'),
            'Actor 1': data.get('actor1', 'Unknown'),
            'Actor 2': data.get('actor2', 'Unknown'),
            'Actor 3': data.get('actor3', 'Unknown')
        }
        
        # Make prediction
        prediction = predictor.predict(movie_data)
        
        return jsonify({
            'prediction': round(prediction, 2),
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

if __name__ == '__main__':
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
