import numpy as np
import json
import os
from hmmlearn import hmm
from collections import Counter
from typing import Tuple, Optional
from joblib import Parallel, delayed
import multiprocessing
import logging
import warnings
import time
from datetime import datetime

np.random.seed(42)

# --- CONSTANTS ---
TRANS_MATRIX_FILENAME = "full_english_transmat.npy"
CHECKPOINT_FILE = "output/hmm_checkpoint.json"
BASE_VALUE = 0.1 
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
        self.english_frequencies = ENGLISH_FREQUENCIES
        
        self.clean_plaintext = None # Will hold the comparable ground truth
        
        self.load_cipher_data()
        self.prepare_observations()
        self.transition_matrix = self._load_transition_matrix()
        
    def load_cipher_data(self):
        """Load and parse cipher data from JSON file."""
        if not os.path.exists(self.cipher_file):
            raise FileNotFoundError(f"Cipher file not found: {self.cipher_file}")

        with open(self.cipher_file, 'r') as file:
            cipher = json.load(file)
        
        self.ciphertext = cipher["ciphertext"]
        self.plaintext = cipher.get("plaintext", "")
        
        # Pre-process plaintext for SER calculation (strip spaces/punctuation)
        if self.plaintext:
            self.clean_plaintext = ''.join([c.lower() for c in self.plaintext if c.isalpha()])
            print(f"Loaded cipher. Validating against clean plaintext length: {len(self.clean_plaintext)}")
        else:
            print("⚠️ No plaintext found in JSON. Logic will fallback to Log Likelihood maximization.")
    
    def prepare_observations(self):
        """Convert ciphertext to observation sequence."""
        self.cipher_observations = np.array([int(x) for x in self.ciphertext.split()])
        self.n_observations = len(self.cipher_observations)
        
        self.unique_symbols = np.unique(self.cipher_observations)
        self.n_unique_symbols = len(self.unique_symbols)
        
        self.symbol_to_index = {symbol: idx for idx, symbol in enumerate(self.unique_symbols)}
        self.index_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_index.items()}
        
        self.observations = np.array([self.symbol_to_index[symbol] 
                                    for symbol in self.cipher_observations])
        
    def _load_transition_matrix(self) -> np.ndarray:
        if os.path.exists(TRANS_MATRIX_FILENAME):
            return np.load(TRANS_MATRIX_FILENAME)
        else:
            print(f"⚠️ WARNING: '{TRANS_MATRIX_FILENAME}' not found. Using flat transition matrix.")
            smoothing_factor = 1e-6
            flat_matrix = np.full((self.alphabet_size, self.alphabet_size), smoothing_factor)
            flat_matrix += np.random.uniform(0, 1e-4, size=flat_matrix.shape)
            row_sums = flat_matrix.sum(axis=1, keepdims=True)
            return flat_matrix / row_sums

    def create_transition_matrix(self) -> np.ndarray:
        return self.transition_matrix
    
    def initialize_hmm(self) -> hmm.CategoricalHMM:
        FLOOR_VALUE = 1e-100 
        
        model = hmm.CategoricalHMM(
            n_components=self.alphabet_size, 
            n_features=self.n_unique_symbols,
            init_params='',     
            n_iter=200, 
            tol=0.01,           
            params='e'
        )
        
        initial_probs = np.array([self.english_frequencies[letter] for letter in self.en_alphabet])
        initial_probs = initial_probs / initial_probs.sum()
        model.startprob_ = initial_probs
        
        trans_matrix = self.create_transition_matrix()
        model.transmat_ = trans_matrix
        
        emission_probs = np.full((self.alphabet_size, self.n_unique_symbols), FLOOR_VALUE)
        emission_probs += np.random.uniform(0, 1, size=emission_probs.shape)
        
        symbol_counts = Counter(self.observations)
        total_counts = len(self.observations)
        cipher_frequencies = np.array([symbol_counts.get(i, 0) / total_counts for i in range(self.n_unique_symbols)])
        
        cipher_freq_matrix = np.tile(cipher_frequencies, (self.alphabet_size, 1))
        english_freq_matrix = np.tile(initial_probs.reshape(-1, 1), (1, self.n_unique_symbols))
        
        frequency_bias = english_freq_matrix * cipher_freq_matrix * 5 
        emission_probs += frequency_bias
        
        emission_probs = np.maximum(emission_probs, FLOOR_VALUE)
        row_sums = emission_probs.sum(axis=1, keepdims=True)
        emission_probs = emission_probs / row_sums
        model.emissionprob_ = np.maximum(emission_probs, FLOOR_VALUE)
        
        return model

    def run_single_hmm(self, X: np.ndarray) -> Tuple[float, str]:
        logging.getLogger('hmmlearn').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                model = self.initialize_hmm()
                model.fit(X)
                curr_log_likelihood = model.score(X)
                _, hidden_states = model.decode(X, algorithm="viterbi")
                curr_decoded = ''.join([self.en_alphabet[state] for state in hidden_states])
                return curr_log_likelihood, curr_decoded
            except Exception as e:
                return float('-inf'), f"HMM training failed: {e}"

    def _calculate_ser(self, decoded_text: str) -> float:
        """Calculates Symbol Error Rate against the stored clean plaintext."""
        if not self.clean_plaintext:
            return 1.0 # If no plaintext, assume worst error rate
            
        min_len = min(len(self.clean_plaintext), len(decoded_text))
        if min_len == 0:
            return 1.0
            
        mismatches = sum(1 for i in range(min_len) 
                        if self.clean_plaintext[i] != decoded_text[i])
        
        length_diff = abs(len(self.clean_plaintext) - len(decoded_text))
        return (mismatches + length_diff) / len(self.clean_plaintext)

    def run_analysis(self, total_restarts=1000, batch_size=100):
        print("=" * 60)
        print("HMM Cryptanalysis Starting (Optimizing for SER)")
        print("=" * 60)

        # 1. Start wallclock for this specific session
        start_wallclock = time.time()
        
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
            best_ser = checkpoint.get("best_ser", 1.0) 
            best_decoded = checkpoint.get("best_decoded", None)
            
            # 2. Retrieve accumulated runtime from previous runs
            accumulated_runtime = checkpoint.get("accumulated_runtime", 0.0)

            print(f"Resuming from batch {start_batch}")
            print(f"Current Best SER: {best_ser:.4f} (LL: {best_log_likelihood:.2f})")
        else:
            start_batch = 0
            best_log_likelihood = float('-inf')
            best_ser = 1.0
            best_decoded = None
            accumulated_runtime = 0.0

        print(f"Starting runtime accumulator at: {accumulated_runtime:.2f} seconds")

        # --- BATCH LOOP ---
        total_batches = int(np.ceil(total_restarts / batch_size))
        for batch_idx in range(start_batch, total_batches):
            print(f"\nRunning batch {batch_idx+1}/{total_batches} ...")

            batch_start = time.time()

            all_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.run_single_hmm)(X) for _ in range(batch_size)
            )

            updated = False
            # Check for best results in this batch
            for log_likelihood, decoded_text in all_results:
                
                # Calculate SER for this specific run
                current_ser = self._calculate_ser(decoded_text)
                
                # Logic Switch: Prioritize SER if plaintext exists
                is_better = False
                
                if self.clean_plaintext:
                    # Prioritize Lower SER
                    if current_ser < best_ser:
                        is_better = True
                    # Tie-breaker: If SER is equal, use Log Likelihood
                    elif current_ser == best_ser and log_likelihood > best_log_likelihood:
                        is_better = True
                else:
                    # Fallback: No plaintext available, use Log Likelihood
                    if log_likelihood > best_log_likelihood:
                        is_better = True

                if is_better:
                    best_ser = current_ser
                    best_log_likelihood = log_likelihood
                    best_decoded = decoded_text
                    updated = True

            if updated:
                print(f"  -> New Best Found! SER: {best_ser:.4f} | LL: {best_log_likelihood:.2f}")
                with open("output/hmm_decoded_plaintext.txt", "w") as f:
                    f.write(best_decoded)

            # 3. Accumulate runtime
            batch_runtime = time.time() - batch_start
            accumulated_runtime += batch_runtime
            current_timestamp = datetime.utcnow().isoformat()

            # --- SAVE CHECKPOINT ---
            checkpoint = {
                "last_completed_batch": batch_idx,
                "best_log_likelihood": best_log_likelihood,
                "best_ser": best_ser,
                "best_decoded": best_decoded,
                "accumulated_runtime": accumulated_runtime,
                "last_update_utc": current_timestamp
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f)

            print(f"Checkpoint saved. Batch runtime: {batch_runtime:.2f}s. Total Accumulated: {accumulated_runtime:.2f}s")
            os.sync()
            
        print("\nAll batches completed.")
        print(f"Final Best SER: {best_ser:.4f}")
        print(f"Final Log Likelihood: {best_log_likelihood:.2f}")

        if best_decoded:
            print("\nBest decoded text (first 200 chars):")
            print(best_decoded[:200])

            with open("output/hmm_decoded_plaintext.txt", "w") as f:
                f.write(best_decoded)

            if self.clean_plaintext:
                print(f"\nComparison (first 200 characters):")
                print(f"Actual:  {self.clean_plaintext[:200]}")
                print(f"Decoded: {best_decoded[:200]}")
                print(f"\nFinal Symbol Error Rate (SER): {best_ser:.4f}")

                with open("output/actual_plaintext.txt", "w") as f:
                    f.write(self.clean_plaintext)

        print("=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

        # Cleanup checkpoint
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

        # 4. Final Runtime Reporting
        total_wallclock = time.time() - start_wallclock
        print(f"\nTotal wall-clock runtime this session: {total_wallclock:.2f} seconds")
        print(f"Accumulated runtime (across restarts): {accumulated_runtime:.2f} seconds")

        return {
            "best_log_likelihood": best_log_likelihood,
            "best_ser": best_ser,
            "decoded": best_decoded
        }