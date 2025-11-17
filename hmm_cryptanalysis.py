import numpy as np
import json
import os
from hmmlearn import hmm # RE-INSTATED THE ACTUAL hmmlearn MODEL
from collections import Counter, defaultdict
import string
from typing import Dict, List, Tuple, Optional

# New imports for parallel processing
from joblib import Parallel, delayed
import multiprocessing
import logging
import warnings # Re-added for RuntimeWarning suppression

np.random.seed(42)

# --- CONSTANTS ---
TRANS_MATRIX_FILENAME = "full_english_transmat.npy"
CHECKPOINT_FILE = "output/hmm_checkpoint.json"
# Global constant for the base value used in B matrix initialization
BASE_VALUE = 0.1 
# Standard English letter frequencies
ENGLISH_FREQUENCIES = {
    'e': 0.1270, 't': 0.0906, 'a': 0.0817, 'o': 0.0751, 'i': 0.0697,
    'n': 0.0675, 's': 0.0633, 'h': 0.0609, 'r': 0.0599, 'd': 0.0425,
    'l': 0.0403, 'c': 0.0278, 'u': 0.0276, 'm': 0.0241, 'w': 0.0236,
    'f': 0.0223, 'g': 0.0202, 'y': 0.0197, 'p': 0.0193, 'b': 0.0149,
    'v': 0.0098, 'k': 0.0077, 'j': 0.0015, 'x': 0.0015, 'q': 0.0010,
    'z': 0.0007
}

class HMMCryptanalysis:
    def __init__(self, cipher_file_path: Optional[str] = None):
        """Initialize the HMM cryptanalysis system."""
        if cipher_file_path is None:
            self.cipher_file = 'ciphers/z408.json'
        else:
            self.cipher_file = cipher_file_path
        
        self.en_alphabet = [chr(i + ord('a')) for i in range(26)]
        self.alphabet_size = len(self.en_alphabet)
        
        # English letter frequencies are kept for Pi and B matrix initialization
        self.english_frequencies = ENGLISH_FREQUENCIES
        
        self.load_cipher_data()
        self.prepare_observations()
        
        # 💡 Integration Point: Load the full transition matrix immediately
        self.transition_matrix = self._load_transition_matrix()
        
    def load_cipher_data(self):
        """Load and parse cipher data from JSON file."""
        # Note: Added check for file existence
        if not os.path.exists(self.cipher_file):
            raise FileNotFoundError(f"Cipher file not found: {self.cipher_file}")

        with open(self.cipher_file, 'r') as file:
            cipher = json.load(file)
        
        self.ciphertext = cipher["ciphertext"]
        self.plaintext = cipher.get("plaintext", "")
        
        print(f"Loaded cipher with {len(self.ciphertext.split())} symbols")
        if self.plaintext:
            print(f"Plaintext available for validation ({len(self.plaintext)} characters)")
    
    def prepare_observations(self):
        """Convert ciphertext to observation sequence."""
        # Parse ciphertext into integer array
        self.cipher_observations = np.array([int(x) for x in self.ciphertext.split()])
        self.n_observations = len(self.cipher_observations)
        
        # Get unique symbols and create mapping
        self.unique_symbols = np.unique(self.cipher_observations)
        self.n_unique_symbols = len(self.unique_symbols)
        
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(self.unique_symbols)}
        self.index_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_index.items()}
        
        # Convert to observation indices
        self.observations = np.array([self.symbol_to_index[symbol] 
                                    for symbol in self.cipher_observations])
        
        print(f"Observations: {self.n_observations}")
        print(f"Unique symbols: {self.n_unique_symbols}")
        print(f"Symbol range: {min(self.unique_symbols)} - {max(self.unique_symbols)}")
        
    def _load_transition_matrix(self) -> np.ndarray:
        """
        Loads the pre-calculated 26x26 matrix or creates a simple placeholder 
        if the file is not found.
        """
        if os.path.exists(TRANS_MATRIX_FILENAME):
            return np.load(TRANS_MATRIX_FILENAME)
        else:
            print(f"⚠️ WARNING: '{TRANS_MATRIX_FILENAME}' not found. Using flat transition matrix.")
            # Fallback: Simple smoothed matrix 
            smoothing_factor = 1e-6
            flat_matrix = np.full((self.alphabet_size, self.alphabet_size), smoothing_factor)
            flat_matrix += np.random.uniform(0, 1e-4, size=flat_matrix.shape)
            row_sums = flat_matrix.sum(axis=1, keepdims=True)
            return flat_matrix / row_sums

    def create_transition_matrix(self) -> np.ndarray:
        """
        Returns the pre-loaded, high-quality English transition matrix (A).
        """
        return self.transition_matrix
    
    def initialize_hmm(self) -> hmm.CategoricalHMM:
        """
        Initializes the HMM exactly as described in the paper:
        - A fixed to English digraph stats (precomputed transition matrix)
        - π initialized randomly (re-estimated by Baum–Welch)
        - B initialized randomly (re-estimated by Baum–Welch)
        """

        # Number of hidden states (plaintext)
        N = self.alphabet_size     
        # Number of observation symbols (ciphertext)
        M = self.n_unique_symbols  

        # Create the model
        model = hmm.CategoricalHMM(
            n_components=N,
            n_features=M,
            init_params="",          # We manually initialize everything
            params="se",             # Train Startprob (π) and Emissions (B)
            n_iter=200,
            tol=0.01
        )

        # -------------------------------------------------------------
        # 1. Initialize π (start probabilities) RANDOMLY
        # -------------------------------------------------------------
        pi = np.random.rand(N)
        pi /= pi.sum()
        model.startprob_ = pi

        # -------------------------------------------------------------
        # 2. Set A (transition matrix) FIXED (English digraph matrix)
        #    — Should NOT be re-estimated during training
        # -------------------------------------------------------------
        A = self.transition_matrix.copy()
        model.transmat_ = A

        # Prevent hmmlearn from trying to modify transitions:
        # (it obeys params="sb", so it won't update A)
        # Just included for clarity.
        # model.transmat_prior = 0  

        # -------------------------------------------------------------
        # 3. Initialize B (emission matrix) RANDOMLY & normalize rows
        # -------------------------------------------------------------
        B = np.random.rand(N, M)
        B = B / B.sum(axis=1, keepdims=True)
        model.emissionprob_ = B

        return model

    # --- Function for parallel execution ---
    def run_single_hmm(self, X: np.ndarray) -> Tuple[float, str]:
        """
        Runs one HMM training and decoding cycle.
        Returns the log likelihood and the decoded text.
        """
        
        # 1. Suppress WARNING level messages from the hmmlearn logger.
        logging.getLogger('hmmlearn').setLevel(logging.ERROR)

        # 2. Suppress C-extension/NumPy RuntimeWarnings (like degenerate solution).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            
            try:
                model = self.initialize_hmm()
                
                # Train the model
                model.fit(X)
                
                # Calculate log likelihood of the observations
                curr_log_likelihood = model.score(X)
                
                # Decode using Viterbi
                _, hidden_states = model.decode(X, algorithm="viterbi")
                curr_decoded = ''.join([self.en_alphabet[state] for state in hidden_states])
                
                return curr_log_likelihood, curr_decoded
            
            except Exception as e:
                # Return lowest possible score on failure
                return float('-inf'), f"HMM training failed: {e}"

    
    def run_analysis(self, total_restarts=1000, batch_size=100):
        print("=" * 60)
        print("HMM Cryptanalysis Starting (Checkpoint Enabled)")
        print("=" * 60)

        os.makedirs('output', exist_ok=True)
        np.seterr(under='ignore')

        X = self.observations.reshape(-1, 1)
        n_jobs = multiprocessing.cpu_count()

        # --- LOAD CHECKPOINT ---
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            start_batch = checkpoint.get("last_completed_batch", 0) + 1
            best_log_likelihood = checkpoint.get("best_log_likelihood", float('-inf'))
            best_decoded = checkpoint.get("best_decoded", None)
            print(f"Resuming from batch {start_batch}")
        else:
            start_batch = 0
            best_log_likelihood = float('-inf')
            best_decoded = None

        # --- BATCH LOOP ---
        total_batches = int(np.ceil(total_restarts / batch_size))
        for batch_idx in range(start_batch, total_batches):
            print(f"\nRunning batch {batch_idx+1}/{total_batches} ...")

            all_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.run_single_hmm)(X) for _ in range(batch_size)
            )

            # Check for best results in this batch
            for log_likelihood, decoded_text in all_results:
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_decoded = decoded_text

                    # Save best decoded so far
                    with open("output/hmm_decoded_plaintext.txt", "w") as f:
                        f.write(best_decoded)

            # --- SAVE CHECKPOINT ---
            checkpoint = {
                "last_completed_batch": batch_idx,
                "best_log_likelihood": best_log_likelihood,
                "best_decoded": best_decoded,
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f)

            print(f"Checkpoint saved after batch {batch_idx+1}. Best log likelihood: {best_log_likelihood:.2f}")

            # Optional: flush I/O to make sure it's written before timeout
            os.sync()
            
        print("\nAll batches completed.")
        print(f"Best log likelihood: {best_log_likelihood:.2f}")

        # Re-attach reporting logic
        if best_decoded:
            print("\nBest decoded text (first 200 chars):")
            print(best_decoded[:200])

            # Save final best result
            with open("output/hmm_decoded_plaintext.txt", "w") as f:
                f.write(best_decoded)

            # Compare with actual plaintext if available
            if self.plaintext:
                clean_plaintext = ''.join([c.lower() for c in self.plaintext if c.isalpha()])
                print(f"\nComparison (first 200 characters):")
                print(f"Actual:  {clean_plaintext[:200]}")
                print(f"Decoded: {best_decoded[:200]}")

                # --- Symbol Error Rate (SER) ---
                min_len = min(len(clean_plaintext), len(best_decoded))
                mismatches = sum(1 for i in range(min_len) if clean_plaintext[i] != best_decoded[i])
                ser = mismatches / min_len if min_len > 0 else 1.0

                print(f"\nSymbol Error Rate (SER): {ser:.4f}")

                with open("output/actual_plaintext.txt", "w") as f:
                    f.write(clean_plaintext)

                # Also store in checkpoint so it’s visible next time
                checkpoint["symbol_error_rate"] = ser
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump(checkpoint, f)

        print("=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

        # --- Clean up checkpoint file ---
        print(f"Analysis complete. Deleting checkpoint file: {CHECKPOINT_FILE}")
        if os.path.exists(CHECKPOINT_FILE):
            try:
                os.remove(CHECKPOINT_FILE)
                print("Checkpoint file successfully deleted.")
            except OSError as e:
                print(f"Error: Could not delete checkpoint file. {e}")
        else:
            print("Checkpoint file not found (already deleted or never created).")

        return {"best_log_likelihood": best_log_likelihood,
                "decoded": best_decoded,
                "symbol_error_rate": checkpoint.get("symbol_error_rate", None)}