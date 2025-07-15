import chardet

# Read the file in binary mode to detect encoding
with open('data/IMDb Movies India.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print("Detected encoding:", result)

# Try reading the first few lines to see the content
with open('data/IMDb Movies India.csv', 'r', encoding='cp1252', errors='replace') as f:
    print("\nFirst few lines:")
    for i, line in enumerate(f):
        if i > 5:  # Print first 5 lines
            break
        print(line.strip())
