"""
Basic HMM Cryptanalysis for Substitution Ciphers
Run: python main.py
"""

from hmm_cryptanalysis import main
from bigrams import HMMCorpusGenerator
import os

if __name__ == "__main__":
    if not os.path.exists('full_english_transmat.npy'):
        generator = HMMCorpusGenerator()
        new_trans_matrix = generator.generate_and_calculate_bigrams(num_books=15)
    main()
