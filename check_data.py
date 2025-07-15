from src.movie_rating_predictor import MovieRatingPredictor

# Create predictor instance
predictor = MovieRatingPredictor()

# Load and prepare data
df = predictor.load_data('data/IMDb Movies India.csv')
print("\nDataset information:")
print(f"Number of rows: {len(df)}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
