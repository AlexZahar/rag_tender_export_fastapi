import re
import pandas as pd
from llama_index.core import Document
from src.config.settings import load_config

config=load_config()

def clean_text(text):
    # Replace </br> and _x000D_ with space
    text = text.replace('</br>', ' ').replace('_x000D_', ' ')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([,.:;])\s*', r'\1 ', text)
    
    # Remove space before closing parenthesis and after opening parenthesis
    text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\(\s+', '(', text)
    
    # Fix split numbers (e.g., "12, 5" to "12,5")
    text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
    
    # Fix common abbreviations
    text = re.sub(r'(\w)\.\s+V\.', r'\1.V.', text)
    text = re.sub(r'z\.\s+B\.', r'z.B.', text)
    text = re.sub(r'd\.\s+h\.', r'd.h.', text)
    
    # Remove spaces before % and °
    text = re.sub(r'\s+(%|°)', r'\1', text)
    
    # Ensure single space after punctuation, except for numbers
    text = re.sub(r'([.!?])(?!\d)\s*', r'\1 ', text)
    
    # Fix ellipsis
    text = re.sub(r'\.\s*\.\s*\.', '...', text)
    
    # Remove space before measurement units
    text = re.sub(r'(\d+)\s+(mm|m|dB|W|K)', r'\1\2', text)
    
    # Fix "e. V." abbreviation
    text = re.sub(r'e\.\s+V\.', 'e.V.', text)
    
    text = re.sub(r'(\d+)\s+kV', r'\1kV', text)

    # Option 1: Remove space after 'x'
    text = re.sub(r'(\d+x)\s+(\d+)', r'\1\2', text)

    # Remove space before asterisk
    text = re.sub(r'\s+\*', '*', text)
    
    # Fix system name at the end
    text = re.sub(r'(\w+)\.\s+de$', r'\1.de', text)
    
    # Remove spaces around slashes
    text = re.sub(r'\s*/\s*', '/', text)

    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def create_documents():
    df = pd.read_csv(config["data_path"])
    
    documents = [
        Document(
            text=f"Knauf System ID: {row['name']}, Eigenschaften: {clean_text(row['long_tender_text'])}",
            metadata={"name": row['name']}
        ) for i, row in df.iterrows()
    ]
    
    return documents

if __name__ == "__main__":
    # This block will only execute if the script is run directly
    df = pd.read_csv(config["data_path"])
    
    # Example usage
    example_text = df['long_tender_text'].iloc[3222]  # Get a specific tender text
    cleaned_example = clean_text(example_text)
    print("Original:")
    print(example_text)
    print("\nCleaned:")
    print(cleaned_example)

    # Create and print information about the documents
    documents = create_documents()
    print(f"\nNumber of documents created: {len(documents)}")
    print(f"First document:")
    print(documents[0].text)  # Print the entire text of the first document

    # Optionally, print metadata of the first document
    print("\nMetadata of the first document:")
    print(documents[0].text)