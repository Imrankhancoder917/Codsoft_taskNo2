from src.movie_rating_predictor import MovieRatingPredictor
import pandas as pd

# Create predictor instance
predictor = MovieRatingPredictor()

# Load data with detailed checks
print("\nReading file with detailed checks...")
try:
    # Read the file with ISO-8859-1 encoding
    df = pd.read_csv('data/IMDb Movies India.csv', encoding='ISO-8859-1', on_bad_lines='warn')
    
    print("\nOriginal data shape:", df.shape)
    print("\nOriginal data types:")
    print(df.dtypes)
    
    # Check for empty or duplicate rows
    print("\nEmpty rows:")
    print(df[df.isnull().all(axis=1)])
    
    print("\nDuplicate rows:")
    print(df[df.duplicated()])
    
    # Check for problematic values in numeric columns
    print("\nFirst few rows of numeric columns:")
    print(df[['Year', 'Duration', 'Rating']].head())
    
    # Check for problematic values in string columns
    print("\nFirst few rows of string columns:")
    print(df[['Name', 'Genre', 'Director', 'Actor 1']].head())
    
except Exception as e:
    print(f"Error reading file: {str(e)}")
