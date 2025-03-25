import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

# Load spaCy model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """
    Clean and normalize text data
    
    Args:
        text: String of text to clean
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text):
    """
    Tokenize text into words
    
    Args:
        text: String of text to tokenize
        
    Returns:
        List of tokens (words)
    """
    try:
        # First try standard tokenization
        return word_tokenize(text)
    except LookupError as e:
        # If punkt_tab specifically is missing, try downloading punkt
        if 'punkt_tab' in str(e):
            print("Downloading punkt tokenizer data...")
            nltk.download('punkt', quiet=False)
            try:
                return word_tokenize(text)
            except:
                # If still failing, use simple split
                return text.split()
        else:
            # For other lookup errors, try downloading the specific resource
            resource_name = str(e).split("Resource ")[1].split(" not found")[0]
            print(f"Downloading missing resource: {resource_name}")
            nltk.download(resource_name, quiet=False)
            try:
                return word_tokenize(text)
            except:
                # Fallback to simple splitting
                return text.split()
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        # Fallback to simple space-based tokenization
        return text.split()
    
def remove_stopwords(tokens, custom_stopwords=None):
    """
    Remove stopwords from list of tokens
    
    Args:
        tokens: List of word tokens
        custom_stopwords: Optional list of additional stopwords
        
    Returns:
        List of tokens with stopwords removed
    """
    try:
        stop_words = set(stopwords.words('english'))
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        # Fallback to a minimal set of common English stopwords
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                     'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
                     'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
                     'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by'}
    
    # Add custom stopwords if provided
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their root form
    
    Args:
        tokens: List of word tokens
        
    Returns:
        List of lemmatized tokens
    """
    try:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except Exception as e:
        print(f"Error lemmatizing tokens: {e}")
        # Return original tokens if lemmatization fails
        return tokens

def extract_entities(text):
    """
    Extract named entities from text using spaCy
    
    Args:
        text: String of text
        
    Returns:
        Dictionary of entities by type
    """
    try:
        doc = nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            
            entities[ent.label_].append(ent.text)
        
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return {}

def preprocess_text(text, remove_stops=True, lemmatize=True, custom_stopwords=None):
    """
    Full preprocessing pipeline
    
    Args:
        text: String of text
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize tokens
        custom_stopwords: Optional list of additional stopwords
        
    Returns:
        Preprocessed text as string
    """
    # Check if text is empty or not a string
    if not text or not isinstance(text, str):
        return ""
        
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = tokenize_text(cleaned_text)
    
    # Remove stopwords if requested
    if remove_stops:
        tokens = remove_stopwords(tokens, custom_stopwords)
    
    # Lemmatize if requested
    if lemmatize:
        tokens = lemmatize_tokens(tokens)
    
    # Join tokens back to string
    return ' '.join(tokens)