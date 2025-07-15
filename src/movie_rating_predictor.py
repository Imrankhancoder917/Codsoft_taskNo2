import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import joblib

class MovieRatingPredictor:
    def __init__(self):
        """Initialize the movie rating predictor."""
        self.pipeline = None
        self.feature_names = None
        
        # Initialize the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                # Numerical features
                ('num', StandardScaler(), ['Year', 'Duration', 'Votes', 'Budget', 'BoxOffice']),
                # Categorical features
                ('cat', OneHotEncoder(handle_unknown='ignore'), [
                    'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3',
                    'Country', 'Language', 'Production', 'Certificate'
                ])
            ],
            remainder='passthrough'
        )
        self.models_dir = 'models'
        
    def load_data(self, filepath):
        """Load and preprocess the IMDb Movies India dataset."""
        print("Loading dataset...")
        
        try:
            # Try reading with ISO-8859-1 encoding
            print("\nReading file with ISO-8859-1 encoding...")
            df = pd.read_csv(filepath, encoding='ISO-8859-1', on_bad_lines='warn')
            
            # Print basic information about the loaded data
            print("\nAvailable columns:")
            print(df.columns.tolist())
            print("\nFirst few rows:")
            print(df.head())
            
            # Basic cleaning
            df = df.dropna(how='all')  # Drop completely empty rows
            df = df.drop_duplicates()   # Remove duplicates
            
            # Select relevant columns
            try:
                df = df[['Name', 'Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]
            except KeyError:
                print("\nColumn names don't match expected format. Available columns:")
                print(df.columns.tolist())
                # Try to rename columns if they don't match exactly
                column_map = {}
                for col in df.columns:
                    if 'name' in col.lower():
                        column_map[col] = 'Name'
                    elif 'year' in col.lower():
                        column_map[col] = 'Year'
                    elif 'duration' in col.lower():
                        column_map[col] = 'Duration'
                    elif 'genre' in col.lower():
                        column_map[col] = 'Genre'
                    elif 'director' in col.lower():
                        column_map[col] = 'Director'
                    elif 'actor' in col.lower():
                        if '1' in col:
                            column_map[col] = 'Actor 1'
                        elif '2' in col:
                            column_map[col] = 'Actor 2'
                        elif '3' in col:
                            column_map[col] = 'Actor 3'
                    elif 'rating' in col.lower():
                        column_map[col] = 'Rating'
                
                df = df.rename(columns=column_map)
                
                # Try again with renamed columns
                df = df[['Name', 'Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]
            
            # Clean columns
            df['Name'] = df['Name'].astype(str).str.strip()
            df['Genre'] = df['Genre'].astype(str).str.strip()
            df['Director'] = df['Director'].astype(str).str.strip()
            df['Actor 1'] = df['Actor 1'].astype(str).str.strip()
            df['Actor 2'] = df['Actor 2'].astype(str).str.strip()
            df['Actor 3'] = df['Actor 3'].astype(str).str.strip()
            
            # Convert numeric columns with special handling
            # For Year: Remove parentheses and convert to numeric
            df['Year'] = df['Year'].astype(str).str.replace('(', '').str.replace(')', '')
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            
            # For Duration: Remove ' min' and commas, convert to numeric
            df['Duration'] = df['Duration'].astype(str).str.replace(' min', '').str.replace(',', '')
            df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
            
            # For Rating: Convert to numeric
            df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
            
            # Drop rows with missing ratings
            df = df.dropna(subset=['Rating'])
            
            # Handle missing values in other columns
            df = df.fillna({
                'Genre': 'Unknown',
                'Director': 'Unknown',
                'Actor 1': 'Unknown',
                'Actor 2': 'Unknown',
                'Actor 3': 'Unknown'
            })
            
            # Drop rows with any remaining NaN values
            df = df.dropna()
            
            print(f"\nSuccessfully loaded {len(df)} rows of data")
            print("\nFinal data preview:")
            print(df.head())
            return df
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise Exception("Could not read file with ISO-8859-1 encoding")
    
    def prepare_features(self, df):
        """Prepare features for model training."""
        # Add default values for new features
        df['Votes'] = df.get('Votes', 0)
        df['Budget'] = df.get('Budget', 0)
        df['BoxOffice'] = df.get('BoxOffice', 0)
        df['Country'] = df.get('Country', 'Unknown')
        df['Language'] = df.get('Language', 'Unknown')
        df['Production'] = df.get('Production', 'Unknown')
        df['Certificate'] = df.get('Certificate', 'Unknown')
        
        # Clean and convert numerical features
        df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)
        df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce').fillna(0)
        df['BoxOffice'] = pd.to_numeric(df['BoxOffice'], errors='coerce').fillna(0)
        
        # Keep only the features we need
        features = ['Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3',
                   'Votes', 'Budget', 'BoxOffice', 'Country', 'Language',
                   'Production', 'Certificate']
        X = df[features]
        y = df['Rating']
        
        # Handle missing values
        X = X.fillna('Unknown')
        
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y
    
    def train_model(self, X, y):
        """Train the model with improved features."""
        print("\nTraining model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with feature selection and model
        self.model = Pipeline([
            ('feature_selection', SelectKBest(score_func=f_regression, k='all')),
            ('regressor', RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        self.model.fit(X_train, y_train)
        self.save_model(self.model)
        
        # Get feature importances and feature names
        feature_importances = pd.DataFrame(
            {'feature': self.preprocessor.get_feature_names_out(),
             'importance': self.model.named_steps['regressor'].feature_importances_}
        )
        
        print("\nTop 15 Important Features:")
        print(feature_importances.sort_values('importance', ascending=False).head(15))
        
        # Make example prediction
        example_movie = {
            'Year': 2023,
            'Duration': 120,
            'Genre': 'Action',
            'Director': 'Unknown',
            'Actor 1': 'Unknown',
            'Actor 2': 'Unknown',
            'Actor 3': 'Unknown',
            'Votes': 10000,
            'Budget': 10000000,
            'BoxOffice': 50000000,
            'Country': 'India',
            'Language': 'Hindi',
            'Production': 'Unknown',
            'Certificate': 'U/A'
        }
        
        example_df = pd.DataFrame([example_movie])
        X_example = self.preprocessor.transform(example_df)
        prediction = self.model.predict(X_example)
        print(f"\nPredicted rating for example movie: {prediction[0]:.2f}")
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"\nModel Performance:")
        print(f"Test RMSE: {rmse:.3f}")
        print(f"Test R² Score: {r2:.3f}")
        
        return self.model
    
    def save_model(self, model):
        """Save the trained model."""
        print("\nSaving model...")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Save both the model and preprocessor
        model_path = os.path.join(self.models_dir, 'movie_rating_model.pkl')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        print(f"Model saved to: {model_path}")
        print(f"Preprocessor saved to: {preprocessor_path}")
    
    def load_model(self):
        """Load the trained model."""
        print("\nLoading model...")
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        model_path = os.path.join(self.models_dir, 'movie_rating_model.pkl')
        preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            print("Model files not found. Creating empty model...")
            self.model = None
            self.preprocessor = None
            return None
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            print(f"Model loaded from: {model_path}")
            print(f"Preprocessor loaded from: {preprocessor_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Creating empty model...")
            self.model = None
            self.preprocessor = None
            return None

    def predict(self, movie_data):
        """Predict movie rating based on input data."""
        # If model is not loaded, return error
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        try:
            # Convert input to DataFrame
            if isinstance(movie_data, dict):
                movie_df = pd.DataFrame([movie_data])
            else:
                movie_df = pd.DataFrame(movie_data)
            
            # Transform input data using preprocessor
            X_processed = self.preprocessor.transform(movie_df)
            
            # Make prediction using the model
            prediction = self.model.predict(X_processed)
            return prediction[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create predictor instance
    predictor = MovieRatingPredictor()
    
    # Load and prepare data (you'll need to provide your own dataset path)
    # df = predictor.load_data('data/movies_dataset.csv')
    # X, y, preprocessor = predictor.prepare_features(df)
    
    # Train and save model
    # predictor.train_model(X, y)
    # predictor.save_model()
    
    # Make predictions
    # example_movie = {
    #     'genre': 'Action',
    #     'director': 'Christopher Nolan',
    #     'actor_1': 'Leonardo DiCaprio',
    #     'actor_2': 'Tom Hardy',
    #     'actor_3': 'Joseph Gordon-Levitt',
    #     'budget': 160000000,
    #     'runtime': 164,
    #     'year': 2010
    # }
    # predicted_rating = predictor.predict(example_movie)
    # print(f"Predicted Rating: {predicted_rating[0]:.2f}")
