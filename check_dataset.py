import pandas as pd

# Try different encodings
encodings = ['utf-8', 'cp1252', 'latin1', 'utf-16']

for encoding in encodings:
    try:
        df = pd.read_csv('data/IMDb Movies India.csv', encoding=encoding)
        print(f"Successfully read file with encoding: {encoding}")
        print("\nDataset columns:")
        print(df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        break
    except Exception as e:
        print(f"Failed with encoding {encoding}: {str(e)}")
