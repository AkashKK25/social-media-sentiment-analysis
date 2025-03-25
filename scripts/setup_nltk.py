import nltk
import os
import sys

def download_nltk_data():
    """
    Download required NLTK data packages
    """
    print("Setting up NLTK data...")
    
    required_packages = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon'
    ]
    
    for package in required_packages:
        try:
            print(f"Checking for {package}...")
            nltk.data.find(f'tokenizers/{package}')
            print(f"- {package} already downloaded")
        except LookupError:
            print(f"- Downloading {package}...")
            nltk.download(package, quiet=False)
    
    print("NLTK setup complete!")

if __name__ == "__main__":
    download_nltk_data()