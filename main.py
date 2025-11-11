"""
Basic HMM Cryptanalysis for Substitution Ciphers
Run: python main.py
"""

from hmm_cryptanalysis import HMMCryptanalysis
from bigrams import HMMCorpusGenerator
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cipher",
        help="Specify the name of the cipher file (e.g., 'cipher1') located in the 'ciphers' directory",
        type=str,  # Changed from store_true to accept a string value
        default=None # Default to None if no cipher is provided
    )
    parser.add_argument(
        "-r",
        "--random-restarts",
        help="Specify number of random restarts",
        type=int,  # Changed from store_true to accept an integer
        default=200000 # Added default value
    )
    parser.add_argument(
        "-b",
        "--batches",
        help="Specify number of batches (batch size)",
        type=int,  # Changed from store_true to accept an integer
        default=1000 # Added default value
    )
    args = parser.parse_args()
    
    if not os.path.exists('full_english_transmat.npy'):
        generator = HMMCorpusGenerator()
        new_trans_matrix = generator.generate_and_calculate_bigrams(num_books=15)
    
    # Construct the cipher file path if a name is provided
    cipher_file_path = None
    if args.cipher:
        # Prepends '/ciphers' (as a relative dir) and appends '.json'
        cipher_file_path = os.path.join("ciphers", f"{args.cipher}.json")
        print(f"Attempting to load cipher from: {cipher_file_path}")
        if not os.path.exists(cipher_file_path):
            print(f"Warning: Cipher file not found at {cipher_file_path}")
            # Depending on HMMCryptanalysis logic, it might handle None or non-existent file
            # For this example, we'll still pass the path and let the class handle it.
            # Or, you could exit:
            # sys.exit(f"Error: Cipher file not found at {cipher_file_path}")
    
    # Pass the cipher path to the constructor (assuming it accepts it)
    analyzer = HMMCryptanalysis(cipher_file_path=cipher_file_path)
    
    # Use the restart and batch size arguments from args
    print(f"Running analysis with {args.random_restarts} total restarts in batches of {args.batches}...")
    results = analyzer.run_analysis(
        total_restarts=args.random_restarts, 
        batch_size=args.batches
    )

    # You'll likely want to do something with the results here
    print("Analysis complete.")
    print(results)