"""
Study 3: Non-Gradient Self-Reference Test
==========================================

PURPOSE: Test whether self-reference produces e-governed dynamics
         WITHOUT information-theoretic optimization (no gradients).

RATIONALE:
  Study 2 showed both self-referential and non-self-referential 
  gradient-based learning produce e-governed dynamics. This could be
  because gradient descent inherently involves log-likelihoods.
  
  To isolate self-reference, we use HEBBIAN LEARNING:
  - No error signal, no backpropagation
  - Updates based on correlation: Δw ∝ pre × post
  - NO logs anywhere in the learning rule

HYPOTHESIS (Danan, 2025):
  Self-referential systems exhibit e-governed dynamics due to 
  state-dependent self-modeling, not just because of optimization.

PREDICTION:
  - Self-referential Hebbian: e-governed dynamics
  - Non-self-referential Hebbian: Different dynamics (NOT e-governed)

REFERENCES:
  - Hebb (1949). The Organization of Behavior. [Hebbian learning]
  - Oja (1982). J Math Biol. [Normalized Hebbian rule]
  - Strogatz (2015). Nonlinear Dynamics and Chaos. [State-dependent dynamics]

STATUS: EMPIRICAL HYPOTHESIS TEST (isolating self-reference)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    extract_time_constants, 
    print_model_comparison, 
    FitResult, 
    compute_aic_bic
)
from scipy.optimize import curve_fit


# =============================================================================
# CURVE FITTING (same as Study 2)
# =============================================================================

def exponential_approach(t, y_max, k, y0):
    return y_max - (y_max - y0) * np.exp(-k * t)

def logistic_model(t, K, k, t0):
    return K / (1 + np.exp(-k * (t - t0)))

def power_law_learning(t, a, b, c):
    return c - a / np.power(t + 1, b)

def linear_model(t, y0, slope):
    return y0 + slope * t

def fit_learning_curves_robust(t: np.ndarray, y: np.ndarray, 
                                verbose: bool = False) -> Dict[str, FitResult]:
    """Robust curve fitting for learning curves."""
    results = {}
    n = len(t)
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_mean = y.mean()
    t_max = t.max()
    
    ss_total = np.sum((y - y_mean)**2)
    
    if verbose:
        print(f"  Data range: [{y_min:.4f}, {y_max:.4f}], range={y_range:.4f}")
    
    if y_range < 0.01:
        if verbose:
            print("  WARNING: Data range < 0.01, curve is flat")
        try:
            popt, _ = curve_fit(linear_model, t, y, p0=[y_mean, 0], maxfev=5000)
            pred = linear_model(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 2)
            r2 = 1 - sse / ss_total if ss_total > 1e-10 else 0
            results['linear'] = FitResult(
                model_name='Linear (NOT e-governed)',
                params={'y0': popt[0], 'slope': popt[1]},
                predictions=pred, sse=sse, n_params=2, aic=aic, bic=bic, r_squared=r2
            )
        except:
            pass
        return results
    
    # 1. Exponential approach
    try:
        y_max_bound = min(y_max * 1.2, 1.0) if y_max < 1 else y_max * 1.2
        y_min_bound = max(y_min * 0.8, 0) if y_min > 0 else y_min - 0.1
        
        if y_max_bound > y_min and y_max > y_min_bound:
            p0 = [y_max * 1.05, 0.01, y_min]
            bounds = ([y_max * 0.95, 1e-5, y_min_bound], 
                      [y_max_bound, 1.0, y_min * 1.1 + 0.01])
            
            valid = all(bounds[0][i] < bounds[1][i] for i in range(len(p0)))
            if not valid:
                bounds = ([-np.inf]*3, [np.inf]*3)
            
            popt, _ = curve_fit(exponential_approach, t, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = exponential_approach(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 3)
            r2 = 1 - sse / ss_total if ss_total > 1e-10 else 0
            
            results['exponential'] = FitResult(
                model_name='Exponential (e-governed)',
                params={'y_max': popt[0], 'k': popt[1], 'y0': popt[2]},
                predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
            )
    except Exception as e:
        if verbose: print(f"  Exponential fit failed: {e}")
    
    # 2. Logistic
    try:
        K_bound = min(y_max * 1.2, 1.0) if y_max < 1 else y_max * 1.2
        
        if K_bound > y_max * 0.95:
            p0 = [y_max * 1.02, 0.05, t_max / 4]
            bounds = ([y_max * 0.95, 1e-5, 0], [K_bound, 1.0, t_max * 2])
            
            valid = all(bounds[0][i] < bounds[1][i] for i in range(len(p0)))
            if not valid:
                bounds = ([-np.inf]*3, [np.inf]*3)
            
            popt, _ = curve_fit(logistic_model, t, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = logistic_model(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 3)
            r2 = 1 - sse / ss_total if ss_total > 1e-10 else 0
            
            results['logistic'] = FitResult(
                model_name='Logistic (e-governed)',
                params={'K': popt[0], 'k': popt[1], 't0': popt[2]},
                predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
            )
    except Exception as e:
        if verbose: print(f"  Logistic fit failed: {e}")
    
    # 3. Power law
    try:
        c_bound = min(y_max * 1.2, 1.0) if y_max < 1 else y_max * 1.2
        
        if c_bound > y_max * 0.95:
            p0 = [y_range, 0.5, y_max * 1.02]
            bounds = ([0.001, 0.01, y_max * 0.95], [y_range * 5 + 0.1, 3.0, c_bound])
            
            valid = all(bounds[0][i] < bounds[1][i] for i in range(len(p0)))
            if not valid:
                bounds = ([-np.inf]*3, [np.inf]*3)
            
            popt, _ = curve_fit(power_law_learning, t, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = power_law_learning(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 3)
            r2 = 1 - sse / ss_total if ss_total > 1e-10 else 0
            
            results['power_law'] = FitResult(
                model_name='Power Law (NOT e-governed)',
                params={'a': popt[0], 'b': popt[1], 'c': popt[2]},
                predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
            )
    except Exception as e:
        if verbose: print(f"  Power law fit failed: {e}")
    
    # 4. Linear
    try:
        slope_est = y_range / t_max if t_max > 0 else 0
        p0 = [y_min, slope_est]
        popt, _ = curve_fit(linear_model, t, y, p0=p0, maxfev=5000)
        pred = linear_model(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 2)
        r2 = 1 - sse / ss_total if ss_total > 1e-10 else 0
        
        results['linear'] = FitResult(
            model_name='Linear (NOT e-governed)',
            params={'y0': popt[0], 'slope': popt[1]},
            predictions=pred, sse=sse, n_params=2, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"  Linear fit failed: {e}")
    
    return results


# =============================================================================
# HEBBIAN NETWORK
# =============================================================================

class HebbianNetwork:
    """
    Simple Hebbian network for classification.
    
    Learning rule (Oja's normalized Hebbian, 1982):
        Δw = η * y * (x - y * w)
    
    This is:
    - NO backpropagation
    - NO error signal  
    - NO log-likelihood
    - Pure correlation-based learning
    
    Reference: Oja (1982). A simplified neuron model as a principal component analyzer.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        np.random.seed(seed)
        
        # Xavier-like initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._hidden = None
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        Returns: (output_probs, hidden_activations)
        """
        h = self.sigmoid(x @ self.W1)
        self._hidden = h
        out = self.softmax(h @ self.W2)
        return out, h
    
    def hebbian_update(self, x: np.ndarray, lr: float = 0.01):
        """
        Oja's normalized Hebbian learning rule.
        
        For W1: Δw = η * h * (x - h * w)
        For W2: Δw = η * o * (h - o * w)
        
        This learns principal components without any error signal.
        """
        out, h = self.forward(x)
        
        # Update W1: input -> hidden
        # Δw_ij = η * h_j * (x_i - h_j * w_ij)
        for j in range(self.hidden_dim):
            h_j = h[:, j:j+1]  # (batch, 1)
            delta = lr * (x * h_j - h_j * h_j * self.W1[:, j:j+1].T).mean(axis=0)
            self.W1[:, j] += delta
        
        # Update W2: hidden -> output
        for k in range(self.output_dim):
            o_k = out[:, k:k+1]
            delta = lr * (h * o_k - o_k * o_k * self.W2[:, k:k+1].T).mean(axis=0)
            self.W2[:, k] += delta
        
        # Normalize weights to prevent explosion
        self.W1 = self.W1 / (np.linalg.norm(self.W1, axis=0, keepdims=True) + 1e-8)
        self.W2 = self.W2 / (np.linalg.norm(self.W2, axis=0, keepdims=True) + 1e-8)


class SelfReferentialHebbianNetwork(HebbianNetwork):
    """
    Hebbian network WITH self-referential component.
    
    The self-model predicts the network's own hidden states.
    This creates self-referential dynamics WITHOUT gradients.
    
    Key: The self-model is ALSO updated via Hebbian learning,
    so there are NO logs anywhere in the system.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        
        # Self-model: predicts hidden state from previous hidden state
        np.random.seed(seed + 1000)
        self.W_self = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        
        self._prev_hidden = None
        self._predicted_hidden = None
    
    def forward_with_self_model(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass with self-model prediction.
        
        Returns: (output_probs, actual_hidden, predicted_hidden)
        """
        out, h = self.forward(x)
        
        # Predict current hidden from previous
        if self._prev_hidden is not None and self._prev_hidden.shape[0] == x.shape[0]:
            self._predicted_hidden = self.sigmoid(self._prev_hidden @ self.W_self)
        else:
            self._predicted_hidden = None
        
        self._prev_hidden = h.copy()
        
        return out, h, self._predicted_hidden
    
    def hebbian_update_with_self(self, x: np.ndarray, lr: float = 0.01, self_lr: float = 0.01):
        """
        Hebbian update including self-model.
        
        The self-model learns to predict hidden states via Hebbian rule:
        If predicted and actual are similar, strengthen connection.
        
        Δw_self = η * h_actual * h_prev  (correlation-based)
        
        This is PURE Hebbian - no error signal, no logs.
        """
        # Standard Hebbian update for main network
        self.hebbian_update(x, lr)
        
        # Self-model Hebbian update
        if self._prev_hidden is not None and self._predicted_hidden is not None:
            h_actual = self._hidden
            h_prev = self._prev_hidden
            
            # Hebbian: strengthen connections for correlated activations
            # Δw = η * post * pre^T (outer product)
            delta = self_lr * (h_actual.T @ h_prev) / x.shape[0]
            self.W_self += delta
            
            # Normalize
            self.W_self = self.W_self / (np.linalg.norm(self.W_self) + 1e-8)
    
    def reset(self):
        self._prev_hidden = None
        self._predicted_hidden = None


class NonSelfReferentialHebbianNetwork(HebbianNetwork):
    """
    Hebbian network WITHOUT self-reference (control).
    
    Has an auxiliary model that predicts INPUT (external),
    not hidden states (internal). Same architecture, no self-reference.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        super().__init__(input_dim, hidden_dim, output_dim, seed)
        
        # External model: predicts input from hidden (like autoencoder)
        np.random.seed(seed + 1000)
        self.W_ext = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
    
    def forward_with_external(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass with external prediction.
        
        Returns: (output_probs, actual_hidden, predicted_input)
        """
        out, h = self.forward(x)
        pred_input = self.sigmoid(h @ self.W_ext)
        return out, h, pred_input
    
    def hebbian_update_with_external(self, x: np.ndarray, lr: float = 0.01, ext_lr: float = 0.01):
        """
        Hebbian update including external model.
        """
        self.hebbian_update(x, lr)
        
        # External model Hebbian update
        h = self._hidden
        # Hebbian: correlate hidden with input
        delta = ext_lr * (h.T @ x) / x.shape[0]
        self.W_ext += delta
        self.W_ext = self.W_ext / (np.linalg.norm(self.W_ext) + 1e-8)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_classification_data(
    n_samples: int = 3000,
    n_features: int = 20,
    n_classes: int = 5,
    separation: float = 2.0,
    noise: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate classification data.
    
    Using moderate difficulty - Hebbian learning is weaker than gradient descent,
    so we need data that's learnable but not trivial.
    """
    np.random.seed(seed)
    
    # Generate well-separated class centers
    centers = np.random.randn(n_classes, n_features) * separation
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        X_class = centers[i] + np.random.randn(samples_per_class, n_features) * noise
        X.append(X_class)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Shuffle
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    
    return X, y


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_self_referential_hebbian(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 0.05,
    self_lr: float = 0.05,
    hidden_dim: int = 32,
    eval_every: int = 5,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Train self-referential Hebbian network.
    
    NO gradients, NO logs - pure correlation-based learning.
    """
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    model = SelfReferentialHebbianNetwork(n_features, hidden_dim, n_classes, seed)
    
    accuracies = []
    epochs_recorded = []
    
    n_samples = len(X)
    
    for epoch in range(n_epochs):
        model.reset()
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        
        # Training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            model.forward_with_self_model(X_batch)
            model.hebbian_update_with_self(X_batch, lr=lr, self_lr=self_lr)
        
        # Evaluate
        if epoch % eval_every == 0:
            model.reset()
            out, _, _ = model.forward_with_self_model(X)
            preds = out.argmax(axis=1)
            acc = (preds == y).mean()
            
            accuracies.append(acc)
            epochs_recorded.append(epoch)
            
            if verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch}: acc={acc:.4f}")
    
    return {
        'epochs': np.array(epochs_recorded),
        'accuracy': np.array(accuracies),
        'condition': 'self_referential_hebbian'
    }


def train_non_self_referential_hebbian(
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 0.05,
    ext_lr: float = 0.05,
    hidden_dim: int = 32,
    eval_every: int = 5,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Train non-self-referential Hebbian network (control).
    """
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    model = NonSelfReferentialHebbianNetwork(n_features, hidden_dim, n_classes, seed)
    
    accuracies = []
    epochs_recorded = []
    
    n_samples = len(X)
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        
        # Training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            model.forward_with_external(X_batch)
            model.hebbian_update_with_external(X_batch, lr=lr, ext_lr=ext_lr)
        
        # Evaluate
        if epoch % eval_every == 0:
            out, _, _ = model.forward_with_external(X)
            preds = out.argmax(axis=1)
            acc = (preds == y).mean()
            
            accuracies.append(acc)
            epochs_recorded.append(epoch)
            
            if verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch}: acc={acc:.4f}")
    
    return {
        'epochs': np.array(epochs_recorded),
        'accuracy': np.array(accuracies),
        'condition': 'non_self_referential_hebbian'
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_study3(
    n_runs: int = 10,
    n_epochs: int = 500,
    n_samples: int = 3000,
    n_classes: int = 5,
    save_dir: str = "figures"
) -> Dict:
    """
    Run Study 3: Non-gradient self-reference test.
    
    Tests whether self-reference produces e-governed dynamics
    WITHOUT information-theoretic optimization.
    
    This isolates self-reference from gradient descent.
    """
    
    print("=" * 70)
    print("STUDY 3: NON-GRADIENT SELF-REFERENCE TEST")
    print("=" * 70)
    print("""
RATIONALE:
  Study 2 showed both conditions exhibit e-governed dynamics with
  gradient descent. But gradient descent uses log-likelihoods,
  which inherently involve e.
  
  To isolate self-reference, we use HEBBIAN LEARNING:
  - NO backpropagation
  - NO error signal
  - NO log-likelihood
  - Pure correlation-based: Δw ∝ pre × post
  
HYPOTHESIS (Danan, 2025):
  Self-reference produces e-governed dynamics due to state-dependent
  self-modeling, independent of information-theoretic optimization.

PREDICTION:
  - Self-referential Hebbian → e-governed dynamics
  - Non-self-referential Hebbian → Different dynamics

REFERENCES:
  - Hebb (1949). The Organization of Behavior.
  - Oja (1982). J Math Biol.
""")
    print("-" * 70)
    
    # Generate data
    print(f"\nGenerating data: {n_samples} samples, {n_classes} classes")
    X, y = generate_classification_data(
        n_samples=n_samples,
        n_classes=n_classes,
        separation=2.0,
        noise=0.5,
        seed=42
    )
    
    baseline_acc = 1.0 / n_classes
    print(f"Baseline (random) accuracy: {baseline_acc:.4f}")
    
    all_self_ref = []
    all_control = []
    
    print(f"\nRunning {n_runs} replications ({n_epochs} epochs each)...")
    print("Using HEBBIAN learning (no gradients, no logs)\n")
    
    for run in range(n_runs):
        print(f"--- Replication {run+1}/{n_runs} ---")
        
        print("  Training self-referential Hebbian...")
        self_ref_data = train_self_referential_hebbian(
            X, y, n_epochs=n_epochs, seed=run*100+42, verbose=True
        )
        
        print("  Training control Hebbian...")
        control_data = train_non_self_referential_hebbian(
            X, y, n_epochs=n_epochs, seed=run*100+42, verbose=True
        )
        
        all_self_ref.append(self_ref_data)
        all_control.append(control_data)
        
        print(f"  Self-ref final acc: {self_ref_data['accuracy'][-1]:.4f}")
        print(f"  Control final acc: {control_data['accuracy'][-1]:.4f}\n")
    
    # Aggregate
    epochs = all_self_ref[0]['epochs']
    
    self_ref_mean = np.mean([r['accuracy'] for r in all_self_ref], axis=0)
    self_ref_std = np.std([r['accuracy'] for r in all_self_ref], axis=0)
    
    control_mean = np.mean([r['accuracy'] for r in all_control], axis=0)
    control_std = np.std([r['accuracy'] for r in all_control], axis=0)
    
    # Diagnostics
    print("\n" + "=" * 60)
    print("LEARNING CURVE DIAGNOSTICS")
    print("=" * 60)
    print(f"\nSelf-Referential Hebbian:")
    print(f"  Start accuracy: {self_ref_mean[0]:.4f}")
    print(f"  End accuracy: {self_ref_mean[-1]:.4f}")
    print(f"  Range: {self_ref_mean[-1] - self_ref_mean[0]:.4f}")
    
    print(f"\nControl Hebbian:")
    print(f"  Start accuracy: {control_mean[0]:.4f}")
    print(f"  End accuracy: {control_mean[-1]:.4f}")
    print(f"  Range: {control_mean[-1] - control_mean[0]:.4f}")
    
    # Fit models
    print("\n" + "=" * 60)
    print("CONDITION A: SELF-REFERENTIAL HEBBIAN")
    print("=" * 60)
    
    self_ref_fits = fit_learning_curves_robust(epochs, self_ref_mean, verbose=True)
    print_model_comparison(self_ref_fits, "Self-Referential Hebbian Learning Curve")
    
    self_ref_tc = extract_time_constants(epochs, self_ref_mean)
    if 'ratio_865_632' in self_ref_tc:
        print(f"\nTime constant ratios (e-governed signature):")
        print(f"  t(86.5%)/τ = {self_ref_tc['ratio_865_632']:.3f} (expected: 2.0)")
        print(f"  t(95.0%)/τ = {self_ref_tc['ratio_950_632']:.3f} (expected: 3.0)")
    
    print("\n" + "=" * 60)
    print("CONDITION B: NON-SELF-REFERENTIAL HEBBIAN (CONTROL)")
    print("=" * 60)
    
    control_fits = fit_learning_curves_robust(epochs, control_mean, verbose=True)
    print_model_comparison(control_fits, "Non-Self-Referential Hebbian Learning Curve")
    
    control_tc = extract_time_constants(epochs, control_mean)
    if 'ratio_865_632' in control_tc:
        print(f"\nTime constant ratios (e-governed signature):")
        print(f"  t(86.5%)/τ = {control_tc['ratio_865_632']:.3f} (expected: 2.0)")
        print(f"  t(95.0%)/τ = {control_tc['ratio_950_632']:.3f} (expected: 3.0)")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    colors = {
        'exponential': '#e41a1c',
        'logistic': '#377eb8',
        'power_law': '#4daf4a',
        'linear': '#984ea3'
    }
    
    # Row 1: Self-referential
    ax = axes[0, 0]
    ax.fill_between(epochs, self_ref_mean - self_ref_std, self_ref_mean + self_ref_std,
                    alpha=0.3, color='steelblue')
    ax.plot(epochs, self_ref_mean, 'o-', color='steelblue', markersize=3, label='Data')
    for name, fit in self_ref_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2, 
                label=f"{fit.model_name} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Chance ({baseline_acc:.2f})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('A: Self-Referential HEBBIAN\n(Predicts Own Hidden States)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Log-linear plot
    ax = axes[0, 1]
    asymptote = self_ref_mean[-1] * 1.02
    gap = asymptote - self_ref_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='steelblue', markersize=3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(Asymptote - Accuracy)')
    ax.set_title('Log-Linear Plot\n(Exponential → linear slope)')
    ax.grid(True, alpha=0.3)
    
    # AIC comparison
    ax = axes[0, 2]
    if self_ref_fits:
        names = list(self_ref_fits.keys())
        aics = [self_ref_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['steelblue'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.capitalize() for n in names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Control
    ax = axes[1, 0]
    ax.fill_between(epochs, control_mean - control_std, control_mean + control_std,
                    alpha=0.3, color='coral')
    ax.plot(epochs, control_mean, 'o-', color='coral', markersize=3, label='Data')
    for name, fit in control_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Chance ({baseline_acc:.2f})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('B: Non-Self-Referential HEBBIAN\n(Predicts External Targets)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Log-linear plot
    ax = axes[1, 1]
    asymptote = control_mean[-1] * 1.02
    gap = asymptote - control_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='coral', markersize=3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(Asymptote - Accuracy)')
    ax.set_title('Log-Linear Plot\n(Exponential → linear slope)')
    ax.grid(True, alpha=0.3)
    
    # AIC comparison
    ax = axes[1, 2]
    if control_fits:
        names = list(control_fits.keys())
        aics = [control_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['coral'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.capitalize() for n in names], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Study 3: Hebbian Learning (No Gradients)\nSelf-Referential vs Non-Self-Referential',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/study3_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/study3_results.pdf', bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 70)
    print("STUDY 3 SUMMARY")
    print("=" * 70)
    
    if self_ref_fits and control_fits:
        best_self_ref = min(self_ref_fits.keys(), key=lambda x: self_ref_fits[x].aic)
        best_control = min(control_fits.keys(), key=lambda x: control_fits[x].aic)
        
        self_ref_e_governed = best_self_ref in ['exponential', 'logistic']
        control_e_governed = best_control in ['exponential', 'logistic']
        
        print(f"""
CONTEXT:
  This study uses HEBBIAN learning - NO gradients, NO log-likelihoods.
  This isolates self-reference from information-theoretic optimization.

HYPOTHESIS:
  Self-reference produces e-governed dynamics INDEPENDENT of optimization.

RESULTS:
  Condition A (Self-Referential Hebbian):
    Best fit: {self_ref_fits[best_self_ref].model_name}
    R²: {self_ref_fits[best_self_ref].r_squared:.4f}
    e-governed: {'YES' if self_ref_e_governed else 'NO'}
    
  Condition B (Non-Self-Referential Hebbian):
    Best fit: {control_fits[best_control].model_name}
    R²: {control_fits[best_control].r_squared:.4f}
    e-governed: {'YES' if control_e_governed else 'NO'}
""")
        
        # Interpretation
        if self_ref_e_governed and not control_e_governed:
            print("=" * 60)
            print("INTERPRETATION: HYPOTHESIS STRONGLY SUPPORTED")
            print("=" * 60)
            print("""
  Self-referential Hebbian shows e-governed dynamics.
  Non-self-referential Hebbian does NOT.
  
  This is the KEY RESULT:
  Without gradient descent, without log-likelihoods, self-reference
  ALONE produces e-governed dynamics.
  
  This supports Danan (2025): Self-referential dynamics are inherently
  e-governed because they create state-dependent change (dS/dt = f(S)).
""")
        elif self_ref_e_governed and control_e_governed:
            print("=" * 60)
            print("INTERPRETATION: BOTH SHOW e-GOVERNED DYNAMICS")
            print("=" * 60)
            print("""
  Even without gradients, both conditions show e-governed dynamics.
  
  Possible explanations:
  1. Hebbian learning itself creates state-dependent dynamics
  2. The manipulation still doesn't isolate self-reference
  3. e-governed dynamics are more universal than hypothesized
""")
        elif not self_ref_e_governed and not control_e_governed:
            print("=" * 60)
            print("INTERPRETATION: NEITHER SHOWS e-GOVERNED DYNAMICS")
            print("=" * 60)
            print("""
  Hebbian learning produces different dynamics than gradient descent.
  Neither condition shows e-governed dynamics.
  
  This suggests:
  1. e-governed dynamics in Study 2 came from gradient descent, not self-reference
  2. The hypothesis needs revision
  3. OR: Hebbian learning is too weak to reveal the underlying dynamics
""")
        else:
            print("=" * 60)
            print("INTERPRETATION: OPPOSITE PATTERN")
            print("=" * 60)
            print("""
  Control shows e-governed, self-referential does not.
  This contradicts the hypothesis.
""")
    
    print(f"\nFigures saved to {save_dir}/")
    
    return {
        'self_ref_fits': self_ref_fits,
        'control_fits': control_fits,
        'self_ref_tc': self_ref_tc,
        'control_tc': control_tc,
        'self_ref_mean': self_ref_mean,
        'control_mean': control_mean,
        'epochs': epochs
    }


if __name__ == "__main__":
    results = run_study3(n_runs=10, n_epochs=500)