from fetcher import Fetcher
# Assuming these imports work correctly in your environment,
# as they provide the functionality for caching and text cleaning.
from utils.files import book_is_cached, get_cached_book

import random
import numpy as np
import string
from collections import Counter
import os # Added os for path management

# Define the file path for the saved transition matrix
TRANS_MATRIX_FILENAME = "full_english_transmat.npy"

# --- Core Bigram Calculation Logic ---

def calculate_english_bigrams_matrix(corpus_text: str) -> np.ndarray:
    """
    Calculates the 26x26 normalized transition matrix (A) from the corpus text.
    
    The text is assumed to be already cleaned (lowercase letters only, no spaces) 
    by format_text. This function applies Add-1 (Laplace) smoothing for robustness.
    """
    ALPHABET = string.ascii_lowercase
    ALPHABET_SIZE = len(ALPHABET)
    
    # 1. Initialize Count Matrix (Add-1 Smoothing)
    # The '1' ensures every possible bigram has a non-zero probability.
    bigram_counts = np.ones((ALPHABET_SIZE, ALPHABET_SIZE), dtype=int)
    
    letter_to_index = {letter: i for i, letter in enumerate(ALPHABET)}

    # 2. Count Bigrams
    # Note: Text cleaning (removing spaces/punctuation) is expected to have 
    # already been done by the format_text utility before this function is called.
    for i in range(len(corpus_text) - 1):
        try:
            char1 = corpus_text[i]
            char2 = corpus_text[i+1]
            
            # Since the corpus_text is already clean, we just map the chars.
            idx1 = letter_to_index[char1]
            idx2 = letter_to_index[char2]
            
            bigram_counts[idx1, idx2] += 1
        except KeyError:
             # Should not happen with correctly formatted text, but safe skip
             continue
        
    # 3. Normalize to Probabilities (Transition Matrix A)
    # Each row (P(Next Letter | Current Letter)) sums to 1.
    row_sums = bigram_counts.sum(axis=1, keepdims=True)
    trans_matrix = bigram_counts / row_sums
    
    return trans_matrix

# --- Corpus Generator Class ---

class HMMCorpusGenerator:
    def __init__(self):
        # Assumes Fetcher is imported and available
        self.fetcher = Fetcher() 
        
    def generate_and_calculate_bigrams(self, num_books: int = 10) -> np.ndarray:
        """
        Fetches text from multiple books, concatenates them, and calculates 
        the smoothed English transition matrix.
        
        It first checks if the matrix is already saved and loads it if available.
        """
        if os.path.exists(TRANS_MATRIX_FILENAME):
            print(f"Loading existing transition matrix from '{TRANS_MATRIX_FILENAME}'...")
            return np.load(TRANS_MATRIX_FILENAME)
            
        full_corpus_text = ""
        
        # Select a random subset of book IDs for diversity
        book_ids_to_use = random.sample(self.fetcher.BOOK_IDS, k=num_books)
        
        print(f"Fetching text from {num_books} books: {book_ids_to_use}...")
        
        for book_id in book_ids_to_use:
            try:
                # Set the fetcher to load the specific book ID
                self.fetcher.book_id = book_id
                
                # Check the cached status for the specific ID we are fetching
                self.fetcher.is_cached = book_is_cached(book_id) 
                
                if self.fetcher.is_cached:
                    text = get_cached_book(book_id)
                else:
                    # fetch_random_book_text internally calls format_text and save_book
                    text = self.fetcher.fetch_random_book_text()
                    
                full_corpus_text += text
                print(f"✅ Added book {book_id}. Total text length: {len(full_corpus_text):,} characters.")
                
            except Exception as e:
                # Catch errors during fetch or IO
                print(f"❌ Skipping book {book_id} due to error: {e}")
                
        if len(full_corpus_text) < 100000:
             print("🚨 WARNING: Corpus is very small (less than 100,000 chars). Consider increasing num_books.")

        if not full_corpus_text:
            raise RuntimeError("Failed to build a corpus from any book.")
            
        print(f"\nTotal clean characters for bigram analysis: {len(full_corpus_text):,}")
        print("Calculating 26x26 transition probabilities (A matrix) with Laplace smoothing...")
        
        # Calculate the matrix using the collected and cleaned corpus
        trans_matrix = calculate_english_bigrams_matrix(full_corpus_text)
        
        # Save the calculated matrix before returning
        np.save(TRANS_MATRIX_FILENAME, trans_matrix)
        print(f"\nMatrix saved successfully to '{TRANS_MATRIX_FILENAME}' for future use.")
        
        return trans_matrix

# --- Main Execution Block (Example Usage) ---
if __name__ == "__main__":
    try:
        generator = HMMCorpusGenerator()
        
        # Use 15 books for a significantly robust model
        new_trans_matrix = generator.generate_and_calculate_bigrams(num_books=15)
        
        print("\n--- Transition Matrix (A) ---")
        print("Shape:", new_trans_matrix.shape)
        
        # Display the normalized matrix (probabilities sum to 1.0)
        print("First 5 rows:")
        with np.printoptions(precision=5, suppress=True):
            print(new_trans_matrix[:5, :5]) # Displaying a 5x5 corner
        
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure your 'fetcher.py' and 'utils' module are correctly set up and accessible.")
