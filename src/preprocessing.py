
import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing special characters (keeping basic punctuation if needed, but removing others)
    - Removing newlines
    
    Args:
        text (str): Input raw text.
        
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    
    # Remove non-printable characters or excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # (Optional) Remove specific special characters if desired, 
    # but BERT tokenizer handles most well. 
    # Here we keep it simple as requested: "lowercase, special chars, basic cleaning"
    # Removing characters that are not alphanumeric or standard punctuation
    # text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', text) 
    
    return text
