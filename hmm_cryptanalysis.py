import numpy as np
import json
import os
import string
from typing import Dict, List, Tuple, Optional
import math

# New imports for parallel processing
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
        # keep pi fixed to english prior (paper recommends fixed pi)
        self.fixed_pi = np.array([self.english_frequencies[letter] for letter in self.en_alphabet])
        self.fixed_pi = self.fixed_pi / self.fixed_pi.sum()
        
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
            mat = np.load(TRANS_MATRIX_FILENAME)
            # Ensure proper shape
            if mat.shape != (self.alphabet_size, self.alphabet_size):
                raise ValueError(f"Loaded transition matrix has wrong shape: {mat.shape}")
            # normalize rows
            mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-300)
            return mat.astype(float)
        else:
            print(f"⚠️ WARNING: '{TRANS_MATRIX_FILENAME}' not found. Using flat transition matrix.")
            # Fallback: Simple smoothed matrix 
            smoothing_factor = 1e-6
            flat_matrix = np.full((self.alphabet_size, self.alphabet_size), smoothing_factor, dtype=float)
            flat_matrix += np.random.uniform(0, 1e-4, size=flat_matrix.shape)
            row_sums = flat_matrix.sum(axis=1, keepdims=True)
            return (flat_matrix / row_sums).astype(float)

    def create_transition_matrix(self) -> np.ndarray:
        """
        Returns the pre-loaded, high-quality English transition matrix (A).
        """
        return self.transition_matrix
    
    def _init_em_params(self, rng: np.random.Generator, base=BASE_VALUE):
        """
        Initialize HMM parameters.
        - A: loaded transition matrix (26x26) (fixed by default)
        - pi: fixed to English frequencies
        - B: emission matrix initialized as base + Uniform(0,1), rows normalized
        """
        N = self.alphabet_size          # 26
        M = self.n_unique_symbols       # number of cipher symbols

        # pi (initial state distribution) - keep fixed
        pi = self.fixed_pi.copy()

        # A (transition matrix)
        A = self.create_transition_matrix()
        # ensure rows sum to 1
        A = A / (A.sum(axis=1, keepdims=True) + 1e-16)

        # B (emission matrix) per paper: b_ij = base + Uniform(0,1), then normalize row-wise
        B = base + rng.random((N, M))
        B = B / (B.sum(axis=1, keepdims=True) + 1e-16)

        # Sanity checks
        assert B.shape == (N, M)
        assert A.shape == (N, N)
        # rows sum to 1
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(B.sum(axis=1) > 0)

        return pi, A, B
    
    def _forward_scaled(self, pi, A, B, O):
        """
        Scaled forward algorithm.
        Returns:
          alpha (T x N) scaled so each row sums to 1,
          c (T,) scaling factors (the sums before scaling),
          loglik = -sum(log(c))
        """
        T = len(O)
        N = A.shape[0]

        alpha = np.zeros((T, N), dtype=float)
        c = np.zeros(T, dtype=float)

        # t = 0
        alpha[0, :] = pi * B[:, O[0]]
        c[0] = alpha[0, :].sum()
        if c[0] <= 0:
            c[0] = 1e-300
        alpha[0, :] /= c[0]

        for t in range(1, T):
            # predict-transmit
            alpha[t, :] = (alpha[t-1, :].dot(A)) * B[:, O[t]]
            c[t] = alpha[t, :].sum()
            if c[t] <= 0:
                c[t] = 1e-300
            alpha[t, :] /= c[t]

        # log-likelihood: log P(O|λ) = - sum_t log(c[t])
        # c contains the evidence scaling factors; we use negative sum of logs per Rabiner
        loglik = -np.sum(np.log(c + 1e-300))
        return alpha, c, loglik

    def _backward_scaled(self, A, B, O, c):
        """
        Correct scaled backward algorithm (consistent with forward scaling).
        Returns beta scaled so that alpha_t * beta_t sums to 1.
        """
        T = len(O)
        N = A.shape[0]
        beta = np.zeros((T, N), dtype=float)

        # initialize last beta
        beta[T - 1, :] = 1.0

        # backward recursion: note the division by c[t+1]
        for t in range(T - 2, -1, -1):
            tmp = B[:, O[t + 1]] * beta[t + 1, :]
            # beta[t,i] = sum_j a_ij * b_j(o_{t+1}) * beta[t+1, j]
            beta[t, :] = A.dot(tmp)
            # scale by c[t+1] (important: not c[t])
            beta[t, :] /= (c[t + 1] + 1e-300)

        return beta

    def _compute_gamma_xi(self, alpha, beta, A, B, O):
        """
        Compute gamma_t(i) and xi_t(i,j) for each t using scaled alpha and beta.
        gamma_t(i) = P(x_t = i | O, λ)  (should sum to 1 over i)
        xi_t(i,j)  = P(x_t=i, x_{t+1}=j | O, λ)
        For numerical stability we compute xi numerator and normalize by its sum.
        """
        T, N = alpha.shape
        gamma = alpha * beta  # element-wise
        # normalize gamma rows to sum 1 (should already be normalized)
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum[gamma_sum == 0] = 1e-300
        gamma /= gamma_sum

        xi = np.zeros((T - 1, N, N), dtype=float)
        for t in range(T - 1):
            # numerator: alpha_t(i) * a_ij * b_j(o_{t+1}) * beta_{t+1}(j)
            numerator = (alpha[t, :][:, None] * A) * (B[:, O[t + 1]] * beta[t + 1, :])[None, :]
            denom = numerator.sum()
            if denom <= 0:
                denom = 1e-300
            xi[t, :, :] = numerator / denom
        return gamma, xi

    def _reestimate(self, gamma, xi, O, reestimate_A=False, rng: Optional[np.random.Generator] = None, base=BASE_VALUE):
        """
        Re-estimate parameters using gamma and xi.
        - pi' = gamma_0   (we will not update pi in training loop to keep it fixed)
        - A' = sum_t xi_t(i,j) / sum_t gamma_t(i)   (if reestimate_A True)
        - B' = for each state j and symbol k: sum_{t:O_t=k} gamma_t(j) / sum_t gamma_t(j)
        """
        T, N = gamma.shape
        M = self.n_unique_symbols

        # pi_new (returned but typically ignored by training loop)
        pi_new = gamma[0, :].copy()

        A_new = None
        if reestimate_A:
            numer = xi.sum(axis=0)  # shape (N,N)
            denom = gamma[:-1, :].sum(axis=0)[:, None]  # shape (N,1)
            denom[denom == 0] = 1e-300
            A_new = numer / denom
            # normalize rows
            A_new = A_new / (A_new.sum(axis=1, keepdims=True) + 1e-300)

        # Re-estimate B
        B_new = np.zeros((N, M), dtype=float)
        EPSILON = 1e-6
        denom = gamma.sum(axis=0)  # shape (N,)
        denom[denom == 0] = 1e-300
        for k in range(M):
            mask = (O == k)
            if mask.any():
                # sum gamma_t(j) over times t where observation == k
                B_new[:, k] = gamma[mask, :].sum(axis=0)
            # else leave zeros; we'll normalize later

        B_new += EPSILON
        denom_smoothed = denom + M * EPSILON
        B_new = B_new / (denom_smoothed[:, None])

        # Reinitialize any degenerate rows (shouldn't be common but safe)
        bad_rows = np.where(~np.isfinite(B_new).all(axis=1) | (B_new.sum(axis=1) == 0))[0]
        if len(bad_rows) > 0:
            if rng is None:
                rng = np.random.default_rng()
            for r in bad_rows:
                B_new[r, :] = base + rng.random(M)
            B_new = B_new / (B_new.sum(axis=1, keepdims=True) + 1e-300)

        return pi_new, A_new, B_new

    def _train_once(self, rng: np.random.Generator, max_iter=200, tol=1e-6, base=BASE_VALUE, reestimate_A=False):
        """
        Run EM (Baum-Welch) starting from a random B init (base + U(0,1)).
        Returns: best_loglik, best_B, decoded_text
        """
        O = self.observations  # indices 0..M-1
        T = len(O)
        # initialize
        pi, A, B = self._init_em_params(rng=rng, base=base)
        best_loglik = -np.inf
        best_B = None
        prev_loglik = -np.inf

        # NOTE: we intentionally DO NOT update pi during training — keep it fixed
        for iteration in range(max_iter):
            # forward/backward with scaling
            alpha, c, loglik = self._forward_scaled(pi, A, B, O)
            beta = self._backward_scaled(A, B, O, c)
            gamma, xi = self._compute_gamma_xi(alpha, beta, A, B, O)

            # re-estimate (we will not assign pi = pi_new to keep it fixed)
            pi_new, A_new, B_new = self._reestimate(gamma, xi, O, reestimate_A=reestimate_A, rng=rng, base=base)

            # update parameters (only replace A if reestimate_A True)
            if reestimate_A and (A_new is not None):
                A = A_new
            B = B_new

            # track progress
            if loglik > best_loglik:
                best_loglik = loglik
                best_B = B.copy()

            # check convergence
            if iteration > 0 and abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik

        # decode via argmax over best_B: map each cipher symbol k -> argmax_j best_B[j,k]
        mapping = np.argmax(best_B, axis=0)  # length M, values in 0..N-1
        decoded_chars = [self.en_alphabet[mapping[sym]] for sym in self.observations]
        decoded_text = ''.join(decoded_chars)

        return best_loglik, best_B, decoded_text

    def run_single_hmm(self, X_dummy=None, max_iter=200, base=BASE_VALUE, reestimate_A=False):
        """
        Run a single random-restart EM training
        Return: loglik, decoded_text
        Note: X_dummy exists only to fit into Parallel(...)(delayed(self.run_single_hmm)(X) ...)
        """
        try:
            # create an independent RNG for this worker
            seed_bytes = os.urandom(8)
            seed = int.from_bytes(seed_bytes, "little")
            rng = np.random.default_rng(seed)

            loglik, best_B, decoded = self._train_once(rng=rng, max_iter=max_iter, base=base, reestimate_A=reestimate_A)
            return float(loglik), decoded
        except Exception as e:
            print("ERROR IN RESTART:", e)
            return float('-inf'), f"HMM training failed: {e}"
    
    def run_analysis(self, total_restarts=1000, batch_size=100):
        print("=" * 60)
        print("HMM Cryptanalysis Starting (Checkpoint Enabled)")
        print("=" * 60)

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
            best_decoded = checkpoint.get("best_decoded", None)
            accumulated_runtime = checkpoint.get("accumulated_runtime", 0.0)

            print(f"Resuming from batch {start_batch}")
        else:
            start_batch = 0
            best_log_likelihood = float('-inf')
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

            # Check for best results in this batch
            for log_likelihood, decoded_text in all_results:
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_decoded = decoded_text

                    # Save best decoded so far
                    with open("output/hmm_decoded_plaintext.txt", "w") as f:
                        f.write(best_decoded)

            batch_runtime = time.time() - batch_start
            accumulated_runtime += batch_runtime

            current_timestamp = datetime.utcnow().isoformat()

            # --- SAVE CHECKPOINT ---
            checkpoint = {
                "last_completed_batch": batch_idx,
                "best_log_likelihood": best_log_likelihood,
                "best_decoded": best_decoded,
                "accumulated_runtime": accumulated_runtime,
                "last_update_utc": current_timestamp
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint, f)

            print(f"Checkpoint saved. Batch runtime: {batch_runtime:.2f}s. "
                f"Total runtime: {accumulated_runtime:.2f}s")
            try:
                os.sync()
            except Exception:
                pass
            
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

        total_wallclock = time.time() - start_wallclock
        print(f"\nTotal wall-clock runtime this session: {total_wallclock:.2f} seconds")
        print(f"Accumulated runtime (across restarts): {accumulated_runtime:.2f} seconds")

        return {"best_log_likelihood": best_log_likelihood,
                "decoded": best_decoded,
                "symbol_error_rate": checkpoint.get("symbol_error_rate", None)}
