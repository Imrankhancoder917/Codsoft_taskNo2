from src.movie_rating_predictor import MovieRatingPredictor
import pandas as pd
import numpy as np

# Create predictor instance
predictor = MovieRatingPredictor()

# Load and prepare data
print("Loading and preparing data...")
df = predictor.load_data('data/IMDb Movies India.csv')
X, y = predictor.prepare_features(df)

# Train model
print("\nTraining model...")
model = predictor.train_model(X, y)

# Save the model
print("\nSaving model...")
predictor.save_model(model)

# Example prediction
print("\nMaking example prediction...")
example_movie = {
    'Genre': 'Action',
    'Director': 'Christopher Nolan',
    'Actor 1': 'Leonardo DiCaprio',
    'Actor 2': 'Tom Hardy',
    'Actor 3': 'Joseph Gordon-Levitt',
    'Duration': 120,
    'Year': 2023
}

# Make prediction
predicted_rating = predictor.predict(example_movie)
print(f"\nPredicted Rating: {predicted_rating:.2f}")

# Get feature importance
if predictor.pipeline is not None:
    print("\nFeature Importances:")
    # Get feature names from the OneHotEncoder
    feature_names = predictor.pipeline.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
    # Get numerical feature names
    numerical_features = predictor.pipeline.named_steps['preprocessor'].transformers_[1][2]
    feature_names = np.concatenate([feature_names, numerical_features])
    
    # Get feature importances from the RandomForest model
    importances = predictor.pipeline.named_steps['regressor'].feature_importances_
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Print top 10 features
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10))
