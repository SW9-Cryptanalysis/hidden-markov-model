import numpy as np
import json
import os
from hmmlearn import hmm
from joblib import Parallel, delayed
import multiprocessing
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger('hmmlearn').setLevel(logging.ERROR)

# --- CONSTANTS ---
TRANS_MATRIX_FILENAME = "full_english_transmat.npy"
CHECKPOINT_FILE = "output/hmm_checkpoint.json"

# Paper Finding: Base = 0.1 gives better performance than 50.5, 5.5, or 2.5[cite: 233].
BASE_VALUE = 0.1 

# Standard English letter frequencies (for Pi vector)
ENGLISH_FREQUENCIES = np.array([
    0.1270, 0.0906, 0.0817, 0.0751, 0.0697, 0.0675, 0.0633, 0.0609, 0.0599, 0.0425,
    0.0403, 0.0278, 0.0276, 0.0241, 0.0236, 0.0223, 0.0202, 0.0197, 0.0193, 0.0149,
    0.0098, 0.0077, 0.0015, 0.0015, 0.0010, 0.0007
])

class HMMCryptanalysis:
    def __init__(self, cipher_file_path: str = 'ciphers/z408.json'):
        self.cipher_file = cipher_file_path
        self.en_alphabet = [chr(i + ord('a')) for i in range(26)]
        self.alphabet_size = len(self.en_alphabet)
        
        # Load Data
        self.load_cipher_data()
        self.prepare_observations()
        
        # Pre-load Transition Matrix (A)
        # The paper emphasizes keeping A fixed and derived from a good corpus[cite: 216].
        self.transition_matrix = self._load_transition_matrix()
        
        # Normalize Start Probabilities (Pi)
        self.start_probs = ENGLISH_FREQUENCIES / ENGLISH_FREQUENCIES.sum()

    def load_cipher_data(self):
        if not os.path.exists(self.cipher_file):
            raise FileNotFoundError(f"Cipher file not found: {self.cipher_file}")

        with open(self.cipher_file, 'r') as file:
            cipher = json.load(file)
        
        self.ciphertext = cipher["ciphertext"]
        self.plaintext = cipher.get("plaintext", "")
        print(f"Loaded cipher length: {len(self.ciphertext.split())}")

    def prepare_observations(self):
        # Parse ciphertext into integer indices
        raw_observations = [int(x) for x in self.ciphertext.split()]
        self.unique_symbols = np.unique(raw_observations)
        self.n_unique_symbols = len(self.unique_symbols)
        
        # Map distinct cipher symbols to 0..M-1
        self.symbol_to_index = {sym: i for i, sym in enumerate(self.unique_symbols)}
        self.observations = np.array([self.symbol_to_index[sym] for sym in raw_observations])
        
        print(f"Distinct cipher symbols (M): {self.n_unique_symbols}")

    def _load_transition_matrix(self) -> np.ndarray:
        if os.path.exists(TRANS_MATRIX_FILENAME):
            return np.load(TRANS_MATRIX_FILENAME)
        else:
            print(f"⚠️ {TRANS_MATRIX_FILENAME} not found. Generating flat placeholder.")
            # Fallback: Flat matrix (Not recommended for real solving, but prevents crash)
            flat = np.full((26, 26), 1/26)
            return flat

    def generate_random_B_matrix(self):
        """
        Implements Equation 12 from the paper: bi(j) ~ base + uniform[0, 1].
        """
        # 1. Initialize with Base Value [cite: 233]
        emission_probs = np.full((self.alphabet_size, self.n_unique_symbols), BASE_VALUE)
        
        # 2. Add Uniform Random Noise 
        emission_probs += np.random.uniform(0, 1, size=emission_probs.shape)
        
        # 3. Normalize row by row [cite: 237]
        row_sums = emission_probs.sum(axis=1, keepdims=True)
        return emission_probs / row_sums

    def run_single_restart(self, X):
        """
        Runs a single random restart. 
        Optimized to create the model efficiently.
        """
        try:
            # Paper: Training steps set to 200 (sufficient for convergence)[cite: 220].
            model = hmm.CategoricalHMM(
                n_components=self.alphabet_size,
                n_features=self.n_unique_symbols,
                init_params='',    # We provide all init matrices manually
                params='e',        # Paper: Keep A fixed, only update B (emissions) [cite: 216]
                n_iter=200,
                tol=0.01
            )
            
            # Set Fixed Parameters
            model.startprob_ = self.start_probs
            model.transmat_ = self.transition_matrix
            
            # Set Random B Matrix
            model.emissionprob_ = self.generate_random_B_matrix()
            
            # Train
            model.fit(X)
            
            # Score (Log Likelihood) - Correlates with accuracy (0.895) [cite: 334]
            score = model.score(X)
            
            # Decode (Viterbi)
            _, hidden_states = model.decode(X, algorithm="viterbi")
            decoded_text = ''.join([self.en_alphabet[s] for s in hidden_states])
            
            return score, decoded_text
            
        except Exception:
            return float('-inf'), ""
        
    def calculate_ser(self, decoded_text):
        """
        Calculates Symbol Error Rate (SER) and Accuracy.
        Accuracy >= 0.80 is generally considered success.
        """
        if not self.plaintext:
            return None

        # Clean strings: keep only alpha, lower case
        def clean(s): return ''.join(filter(str.isalpha, s)).lower()
        
        truth = clean(self.plaintext)
        pred = clean(decoded_text)
        
        # Truncate to minimum length for comparison
        min_len = min(len(truth), len(pred))
        if min_len == 0:
            return 1.0 # 100% error if empty

        # Count mismatches
        errors = sum(1 for i in range(min_len) if truth[i] != pred[i])
        
        ser = errors / min_len
        accuracy = 1.0 - ser
        
        return ser, accuracy

    def run_analysis(self, total_restarts=10000, batch_size=100):
        """
        Paper: 10,000 restarts were sufficient for Fake Zodiac 340[cite: 283].
        """
        print(f"Starting HMM Attack with {total_restarts} restarts...")
        
        X = self.observations.reshape(-1, 1)
        best_score = float('-inf')
        best_text = ""
        
        # Use all available cores
        n_jobs = multiprocessing.cpu_count()
        n_batches = int(np.ceil(total_restarts / batch_size))
        
        for i in range(n_batches):
            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.run_single_restart)(X) for _ in range(batch_size)
            )
            
            # Find best in batch
            current_batch_best_score = float('-inf')
            current_batch_best_text = ""
            
            for score, text in results:
                if score > current_batch_best_score:
                    current_batch_best_score = score
                    current_batch_best_text = text
            
            # Update global best
            if current_batch_best_score > best_score:
                best_score = current_batch_best_score
                best_text = current_batch_best_text
                print(f"Batch {i+1}/{n_batches}: New Best Score: {best_score:.2f}")
                
                # Check SER if plaintext exists (for monitoring progress)
                if self.plaintext:
                    metrics = self.calculate_ser(best_text)
                    if metrics:
                        ser, acc = metrics
                        print(f"Current Accuracy: {acc*100:.2f}% (SER: {ser:.4f})")

                # Save progress
                with open("output/best_result.txt", "w") as f:
                    f.write(best_text)
        
        print("\n" + "="*50)
        print(f"Final Best Score: {best_score}")
        
        # Final SER Calculation
        if self.plaintext:
            metrics = self.calculate_ser(best_text)
            if metrics:
                ser, acc = metrics
                print(f"Final Symbol Error Rate (SER): {ser:.4f}")
                print(f"Final Accuracy: {acc*100:.2f}%")
                if acc >= 0.80:
                    print("Result: SUCCESS (Accuracy >= 80%) ")
                else:
                    print("Result: FAIL (< 80% accuracy)")
        
        print("Decoded Text Head:")
        print(best_text[:200])
        print("="*50)
        
        return best_text