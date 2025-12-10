"""
Study 4b: Representational Similarity Analysis of Compositional Dynamics
=========================================================================

PURPOSE: Test whether COMPOSITIONAL task structure produces e-governed
         dynamics in REPRESENTATIONAL STRUCTURE (not just accuracy).

THEORETICAL MOTIVATION:
  Studies 2-4 found gradient descent produces e-governed accuracy dynamics
  regardless of architecture or task. But accuracy is OUTPUT, not STATE.
  
  The theoretical claim (Danan, 2025) is about representational capacity:
  "The state IS the system's representational capacity itself."
  
  We should measure e-dynamics of REPRESENTATION, not accuracy.

HYPOTHESIS:
  - Compositional tasks produce HIERARCHICAL representational structure
  - This structure emerges following e-governed dynamics (dS/dt ∝ S)
  - Non-compositional tasks produce FLAT representational structure
  - This structure emerges following different dynamics (power law)

METHOD:
  1. Train networks on hierarchical vs parallel XOR (same as Study 4)
  2. At each epoch, extract hidden layer representations
  3. Compute neural RDM (Representational Dissimilarity Matrix)
  4. Define theoretical RDMs for compositional vs non-compositional structure
  5. Track RSA correlation over training
  6. Fit e-governed vs power law models to RSA trajectories

THEORETICAL RDMs:
  Compositional (Hierarchical XOR):
    - Inputs share structure if they share INTERMEDIATE computations
    - Nested hierarchy: same parity of bits 1-2, then 1-4, then 1-6, then 1-8
    
  Non-Compositional (Parallel XOR):
    - Inputs share structure only if they have SAME OUTPUTS
    - Flat clustering by output vector

REFERENCES:
  - Kriegeskorte, Mur, & Bandettini (2008). Frontiers in Systems Neuroscience.
  - Kornblith et al. (2019). Similarity of neural network representations. ICML.
  - Lake & Baroni (2018). Generalization without systematicity. ICML.

STATUS: EMPIRICAL HYPOTHESIS TEST (representational dynamics)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import os


# =============================================================================
# MODEL FITTING (same as previous studies)
# =============================================================================

@dataclass
class FitResult:
    model_name: str
    params: Dict
    predictions: np.ndarray
    sse: float
    n_params: int
    aic: float
    bic: float
    r_squared: float


def compute_aic_bic(sse: float, n: int, k: int) -> Tuple[float, float]:
    if sse <= 0:
        return float('inf'), float('inf')
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return aic, bic


def exponential_approach(t, y_max, k, y0):
    return y_max - (y_max - y0) * np.exp(-k * t)

def logistic_model(t, K, k, t0):
    return K / (1 + np.exp(-k * (t - t0)))

def power_law_learning(t, a, b, c):
    return c - a / np.power(t + 1, b)

def linear_model(t, y0, slope):
    return y0 + slope * t


def fit_rsa_curves(t: np.ndarray, y: np.ndarray, 
                   verbose: bool = False) -> Dict[str, FitResult]:
    """
    Fit candidate models to RSA trajectory.
    RSA values range from -1 to 1 (Spearman correlation).
    """
    results = {}
    n = len(t)
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_mean = y.mean()
    t_max = t.max()
    
    ss_total = np.sum((y - y_mean)**2)
    
    if verbose:
        print(f"  RSA range: [{y_min:.4f}, {y_max:.4f}], range={y_range:.4f}")
    
    if y_range < 0.02 or ss_total < 1e-10:
        if verbose:
            print("  WARNING: RSA range too small, curve essentially flat")
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
    
    # 1. Exponential approach (for RSA: y_max can be up to 1.0)
    try:
        y_max_bound = min(y_max * 1.5, 1.0)
        y_min_bound = max(y_min * 0.5, -1.0)
        
        p0 = [min(y_max * 1.1, 0.95), 0.01, y_min]
        bounds = ([y_max * 0.9, 1e-6, y_min_bound], 
                  [y_max_bound, 0.5, y_min * 0.5 + 0.1])
        
        # Validate bounds
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if not valid:
            bounds = ([-1, 1e-6, -1], [1, 1, 1])
        
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
        K_bound = min(y_max * 1.5, 1.0)
        
        p0 = [min(y_max * 1.1, 0.95), 0.02, t_max / 4]
        bounds = ([y_max * 0.9, 1e-6, 0], [K_bound, 0.5, t_max * 2])
        
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if not valid:
            bounds = ([0, 1e-6, 0], [1, 1, t_max * 2])
        
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
    
    # 3. Power law (for RSA approaching asymptote)
    try:
        c_bound = min(y_max * 1.5, 1.0)
        
        p0 = [y_range, 0.5, min(y_max * 1.1, 0.95)]
        bounds = ([0.001, 0.01, y_max * 0.9], [y_range * 10 + 0.1, 3.0, c_bound])
        
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if not valid:
            bounds = ([0.001, 0.01, 0], [2, 3, 1])
        
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


def print_model_comparison(results: Dict[str, FitResult], title: str):
    """Print formatted model comparison table."""
    if not results:
        print(f"No models fit successfully for {title}")
        return
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].aic)
    min_aic = sorted_results[0][1].aic
    
    print(f"{'Model':<35} {'R²':<10} {'AIC':<12} {'ΔAIC':<10}")
    print("-" * 60)
    
    for name, fit in sorted_results:
        delta_aic = fit.aic - min_aic
        marker = "← BEST" if delta_aic == 0 else ""
        print(f"{fit.model_name:<35} {fit.r_squared:<10.4f} {fit.aic:<12.2f} {delta_aic:<10.2f} {marker}")
    
    print("=" * 60)


# =============================================================================
# THEORETICAL RDMs
# =============================================================================

def compute_hierarchical_rdm(n_bits: int = 8) -> np.ndarray:
    """
    Compute theoretical RDM for HIERARCHICAL (compositional) structure.
    
    For hierarchical XOR (parity), inputs are similar if they share
    intermediate computation results. This creates NESTED structure:
    
    - Same parity of bits 1-2 → distance 0.2
    - Same parity of bits 1-4 → distance 0.4  
    - Same parity of bits 1-6 → distance 0.6
    - Same parity of bits 1-8 (final output) → distance 0.8
    - Different on all → distance 1.0
    
    This captures the COMPOSITIONAL HIERARCHY: representations should
    reflect the nested structure of intermediate computations.
    """
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)])
    
    # Compute intermediate parities at different depths
    parities = []
    for depth in [2, 4, 6, 8]:
        if depth <= n_bits:
            parity = X[:, :depth].sum(axis=1) % 2
            parities.append(parity)
    
    # Build RDM based on shared intermediate computations
    rdm = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Count how many intermediate parities match
            matches = sum(parities[k][i] == parities[k][j] for k in range(len(parities)))
            # More matches = smaller distance (more similar)
            # Scale: 0 matches → 1.0, all matches → 0.0
            distance = 1.0 - (matches / len(parities))
            rdm[i, j] = distance
            rdm[j, i] = distance
    
    return rdm


def compute_parallel_rdm(n_bits: int = 8) -> np.ndarray:
    """
    Compute theoretical RDM for PARALLEL (non-compositional) structure.
    
    For parallel XORs, inputs are similar ONLY if they produce the
    SAME OUTPUT VECTOR. This creates FLAT clustering:
    
    - Same output vector → distance 0.0
    - Different output vector → distance proportional to Hamming distance
    
    There is NO hierarchy, NO nested structure.
    """
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)])
    
    # Compute parallel XOR outputs
    n_outputs = n_bits // 2
    outputs = np.zeros((n_samples, n_outputs))
    for k in range(n_outputs):
        outputs[:, k] = (X[:, 2*k] + X[:, 2*k + 1]) % 2
    
    # Build RDM based on output similarity (Hamming distance)
    rdm = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # Hamming distance between output vectors
            hamming = np.sum(outputs[i] != outputs[j]) / n_outputs
            rdm[i, j] = hamming
            rdm[j, i] = hamming
    
    return rdm


def compute_neural_rdm(activations: np.ndarray) -> np.ndarray:
    """
    Compute neural RDM from hidden layer activations.
    
    Uses correlation distance (1 - correlation) as in Kriegeskorte et al. (2008).
    """
    # Correlation distance
    rdm = squareform(pdist(activations, metric='correlation'))
    return rdm


def compute_rsa(neural_rdm: np.ndarray, theoretical_rdm: np.ndarray) -> float:
    """
    Compute RSA: Spearman correlation between neural and theoretical RDMs.
    
    Following Kriegeskorte et al. (2008), we correlate the upper triangular
    portions of the RDMs.
    """
    # Extract upper triangular (excluding diagonal)
    n = neural_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    
    neural_vec = neural_rdm[triu_idx]
    theoretical_vec = theoretical_rdm[triu_idx]
    
    # Handle constant vectors
    if np.std(neural_vec) < 1e-10 or np.std(theoretical_vec) < 1e-10:
        return 0.0
    
    # Spearman correlation
    rho, _ = spearmanr(neural_vec, theoretical_vec)
    
    return rho if not np.isnan(rho) else 0.0


# =============================================================================
# NETWORK
# =============================================================================

class XORNetworkWithHooks(nn.Module):
    """
    MLP for XOR tasks with hooks to extract hidden representations.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Store activations
        self._activations = {}
    
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        self._activations['layer1'] = h1.detach()
        
        h2 = self.relu(self.fc2(h1))
        self._activations['layer2'] = h2.detach()
        
        h3 = self.relu(self.fc3(h2))
        self._activations['layer3'] = h3.detach()
        
        out = self.sigmoid(self.fc4(h3))
        return out
    
    def get_activations(self, layer: str = 'layer2') -> np.ndarray:
        """Get activations from specified layer."""
        return self._activations[layer].cpu().numpy()


# =============================================================================
# DATA GENERATION (same as Study 4)
# =============================================================================

def generate_hierarchical_xor_data(n_bits: int = 8):
    """Generate hierarchical XOR (parity) data."""
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)], dtype=np.float32)
    y = (X.sum(axis=1) % 2).astype(np.float32)
    return torch.tensor(X), torch.tensor(y).unsqueeze(1)


def generate_parallel_xor_data(n_bits: int = 8):
    """Generate parallel XOR data."""
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)], dtype=np.float32)
    
    n_outputs = n_bits // 2
    y = np.zeros((n_samples, n_outputs), dtype=np.float32)
    for i in range(n_outputs):
        y[:, i] = (X[:, 2*i] + X[:, 2*i + 1]) % 2
    
    return torch.tensor(X), torch.tensor(y)


# =============================================================================
# TRAINING WITH RSA TRACKING
# =============================================================================

def train_with_rsa_tracking(
    X: torch.Tensor,
    y: torch.Tensor,
    theoretical_rdm: np.ndarray,
    hidden_dim: int = 64,
    n_epochs: int = 500,
    lr: float = 0.01,
    eval_every: int = 5,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Train network and track RSA to theoretical RDM over time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    
    model = XORNetworkWithHooks(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    accuracies = []
    rsa_layer1 = []
    rsa_layer2 = []
    rsa_layer3 = []
    epochs_recorded = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X)
        loss = criterion(output, y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                _ = model(X)  # Forward pass to populate activations
                
                # Compute neural RDMs for each layer
                act1 = model.get_activations('layer1')
                act2 = model.get_activations('layer2')
                act3 = model.get_activations('layer3')
                
                neural_rdm1 = compute_neural_rdm(act1)
                neural_rdm2 = compute_neural_rdm(act2)
                neural_rdm3 = compute_neural_rdm(act3)
                
                # Compute RSA
                rsa1 = compute_rsa(neural_rdm1, theoretical_rdm)
                rsa2 = compute_rsa(neural_rdm2, theoretical_rdm)
                rsa3 = compute_rsa(neural_rdm3, theoretical_rdm)
                
                # Accuracy
                pred = (model(X) > 0.5).float()
                acc = (pred == y).float().mean().item()
            
            accuracies.append(acc)
            rsa_layer1.append(rsa1)
            rsa_layer2.append(rsa2)
            rsa_layer3.append(rsa3)
            epochs_recorded.append(epoch)
            
            if verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch}: acc={acc:.4f}, RSA(L2)={rsa2:.4f}")
    
    return {
        'epochs': np.array(epochs_recorded),
        'accuracy': np.array(accuracies),
        'rsa_layer1': np.array(rsa_layer1),
        'rsa_layer2': np.array(rsa_layer2),
        'rsa_layer3': np.array(rsa_layer3)
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_study4b(
    n_runs: int = 10,
    n_bits: int = 8,
    n_epochs: int = 500,
    hidden_dim: int = 64,
    lr: float = 0.01,
    save_dir: str = "figures"
) -> Dict:
    """
    Run Study 4b: RSA analysis of compositional vs non-compositional dynamics.
    """
    
    print("=" * 70)
    print("STUDY 4b: RSA ANALYSIS OF REPRESENTATIONAL DYNAMICS")
    print("=" * 70)
    print("""
THEORETICAL MOTIVATION:
  Studies 2-4 measured e-dynamics of ACCURACY (output).
  But the claim is about REPRESENTATIONAL CAPACITY (internal state):
  
  "The state IS the system's representational capacity itself."
  
  We should measure e-dynamics of REPRESENTATION, not accuracy.

HYPOTHESIS:
  - Compositional task → Hierarchical representation → e-governed RSA dynamics
  - Non-compositional task → Flat representation → Different RSA dynamics

METHOD:
  1. Train on hierarchical vs parallel XOR
  2. At each epoch, compute neural RDM from hidden activations
  3. Define theoretical RDMs:
     - Compositional: Nested structure (shared intermediate parities)
     - Non-compositional: Flat structure (output vector similarity)
  4. Track RSA(neural, theoretical) over training
  5. Fit e-governed vs power law to RSA trajectories

REFERENCES:
  - Kriegeskorte, Mur, & Bandettini (2008). Frontiers in Systems Neuroscience.
""")
    print("-" * 70)
    
    # Generate data
    print(f"\nGenerating data: {n_bits} bits")
    X_hier, y_hier = generate_hierarchical_xor_data(n_bits)
    X_para, y_para = generate_parallel_xor_data(n_bits)
    
    # Compute theoretical RDMs
    print("Computing theoretical RDMs...")
    hier_theoretical_rdm = compute_hierarchical_rdm(n_bits)
    para_theoretical_rdm = compute_parallel_rdm(n_bits)
    
    print(f"  Hierarchical RDM: {hier_theoretical_rdm.shape}")
    print(f"  Parallel RDM: {para_theoretical_rdm.shape}")
    
    all_hierarchical = []
    all_parallel = []
    
    print(f"\nRunning {n_runs} replications ({n_epochs} epochs each)...")
    
    for run in range(n_runs):
        print(f"\n--- Replication {run+1}/{n_runs} ---")
        
        print("  Training hierarchical (compositional) with RSA tracking...")
        hier_data = train_with_rsa_tracking(
            X_hier, y_hier, hier_theoretical_rdm,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=lr,
            seed=run*100+42,
            verbose=True
        )
        
        print("  Training parallel (non-compositional) with RSA tracking...")
        para_data = train_with_rsa_tracking(
            X_para, y_para, para_theoretical_rdm,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            lr=lr,
            seed=run*100+42,
            verbose=True
        )
        
        all_hierarchical.append(hier_data)
        all_parallel.append(para_data)
        
        print(f"  Hierarchical final RSA(L2): {hier_data['rsa_layer2'][-1]:.4f}")
        print(f"  Parallel final RSA(L2): {para_data['rsa_layer2'][-1]:.4f}")
    
    # Aggregate RSA trajectories
    epochs = all_hierarchical[0]['epochs']
    
    hier_rsa_mean = np.mean([r['rsa_layer2'] for r in all_hierarchical], axis=0)
    hier_rsa_std = np.std([r['rsa_layer2'] for r in all_hierarchical], axis=0)
    
    para_rsa_mean = np.mean([r['rsa_layer2'] for r in all_parallel], axis=0)
    para_rsa_std = np.std([r['rsa_layer2'] for r in all_parallel], axis=0)
    
    # Also get accuracy for reference
    hier_acc_mean = np.mean([r['accuracy'] for r in all_hierarchical], axis=0)
    para_acc_mean = np.mean([r['accuracy'] for r in all_parallel], axis=0)
    
    # Diagnostics
    print("\n" + "=" * 60)
    print("RSA TRAJECTORY DIAGNOSTICS")
    print("=" * 60)
    print(f"\nHierarchical (Compositional) - RSA to Hierarchical RDM:")
    print(f"  Start RSA: {hier_rsa_mean[0]:.4f}")
    print(f"  End RSA: {hier_rsa_mean[-1]:.4f}")
    print(f"  Range: {hier_rsa_mean[-1] - hier_rsa_mean[0]:.4f}")
    
    print(f"\nParallel (Non-Compositional) - RSA to Parallel RDM:")
    print(f"  Start RSA: {para_rsa_mean[0]:.4f}")
    print(f"  End RSA: {para_rsa_mean[-1]:.4f}")
    print(f"  Range: {para_rsa_mean[-1] - para_rsa_mean[0]:.4f}")
    
    # Fit models to RSA trajectories
    print("\n" + "=" * 60)
    print("CONDITION A: HIERARCHICAL XOR (COMPOSITIONAL)")
    print("Fitting models to RSA trajectory...")
    print("=" * 60)
    
    hier_fits = fit_rsa_curves(epochs, hier_rsa_mean, verbose=True)
    print_model_comparison(hier_fits, "Hierarchical RSA Dynamics")
    
    print("\n" + "=" * 60)
    print("CONDITION B: PARALLEL XORs (NON-COMPOSITIONAL)")
    print("Fitting models to RSA trajectory...")
    print("=" * 60)
    
    para_fits = fit_rsa_curves(epochs, para_rsa_mean, verbose=True)
    print_model_comparison(para_fits, "Parallel RSA Dynamics")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    colors = {
        'exponential': '#e41a1c',
        'logistic': '#377eb8',
        'power_law': '#4daf4a',
        'linear': '#984ea3'
    }
    
    # Row 1: Hierarchical (Compositional)
    # RSA trajectory
    ax = axes[0, 0]
    ax.fill_between(epochs, hier_rsa_mean - hier_rsa_std, hier_rsa_mean + hier_rsa_std,
                    alpha=0.3, color='steelblue')
    ax.plot(epochs, hier_rsa_mean, 'o-', color='steelblue', markersize=2, label='Data')
    for name, fit in hier_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RSA (Spearman ρ)')
    ax.set_title('A: Hierarchical XOR (Compositional)\nRSA to Hierarchical Theoretical RDM')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Accuracy for reference
    ax = axes[0, 1]
    ax.plot(epochs, hier_acc_mean, 'o-', color='steelblue', markersize=2)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy (Reference)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    # Log-linear plot of RSA
    ax = axes[0, 2]
    asymptote = hier_rsa_mean[-1] * 1.05
    gap = asymptote - hier_rsa_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='steelblue', markersize=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(Asymptote - RSA)')
    ax.set_title('Log-Linear Plot\n(e-governed → linear slope)')
    ax.grid(True, alpha=0.3)
    
    # AIC comparison
    ax = axes[0, 3]
    if hier_fits:
        names = list(hier_fits.keys())
        aics = [hier_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['steelblue'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Parallel (Non-Compositional)
    ax = axes[1, 0]
    ax.fill_between(epochs, para_rsa_mean - para_rsa_std, para_rsa_mean + para_rsa_std,
                    alpha=0.3, color='coral')
    ax.plot(epochs, para_rsa_mean, 'o-', color='coral', markersize=2, label='Data')
    for name, fit in para_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('RSA (Spearman ρ)')
    ax.set_title('B: Parallel XORs (Non-Compositional)\nRSA to Parallel Theoretical RDM')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(epochs, para_acc_mean, 'o-', color='coral', markersize=2)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy (Reference)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    ax = axes[1, 2]
    asymptote = para_rsa_mean[-1] * 1.05
    gap = asymptote - para_rsa_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='coral', markersize=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('log(Asymptote - RSA)')
    ax.set_title('Log-Linear Plot\n(e-governed → linear slope)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 3]
    if para_fits:
        names = list(para_fits.keys())
        aics = [para_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['coral'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Study 4b: RSA Dynamics of Compositional vs Non-Compositional Representations\n'
                 'Does compositional REPRESENTATIONAL structure emerge following e-dynamics?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/study4b_rsa_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/study4b_rsa_results.pdf', bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 70)
    print("STUDY 4b SUMMARY")
    print("=" * 70)
    
    if hier_fits and para_fits:
        best_hier = min(hier_fits.keys(), key=lambda x: hier_fits[x].aic)
        best_para = min(para_fits.keys(), key=lambda x: para_fits[x].aic)
        
        hier_e_governed = best_hier in ['exponential', 'logistic']
        para_e_governed = best_para in ['exponential', 'logistic']
        
        # Calculate ΔAIC between e-governed and non-e-governed
        hier_e_models = [k for k in hier_fits if k in ['exponential', 'logistic']]
        hier_non_e = [k for k in hier_fits if k in ['power_law', 'linear']]
        para_e_models = [k for k in para_fits if k in ['exponential', 'logistic']]
        para_non_e = [k for k in para_fits if k in ['power_law', 'linear']]
        
        hier_delta = None
        if hier_e_models and hier_non_e:
            best_hier_e_aic = min(hier_fits[k].aic for k in hier_e_models)
            best_hier_non_e_aic = min(hier_fits[k].aic for k in hier_non_e)
            hier_delta = best_hier_non_e_aic - best_hier_e_aic
            
        para_delta = None
        if para_e_models and para_non_e:
            best_para_e_aic = min(para_fits[k].aic for k in para_e_models)
            best_para_non_e_aic = min(para_fits[k].aic for k in para_non_e)
            para_delta = best_para_non_e_aic - best_para_e_aic
        
        print(f"""
KEY QUESTION:
  Does compositional REPRESENTATIONAL structure (not accuracy) emerge
  following e-governed dynamics?

RESULTS:
  Condition A (Hierarchical/Compositional):
    Best fit for RSA trajectory: {hier_fits[best_hier].model_name}
    R²: {hier_fits[best_hier].r_squared:.4f}
    e-governed: {'YES' if hier_e_governed else 'NO'}
    ΔAIC (e-governed vs non-e-governed): {hier_delta:.2f if hier_delta is not None else 'N/A'}
    Final RSA: {hier_rsa_mean[-1]:.4f}
    
  Condition B (Parallel/Non-Compositional):
    Best fit for RSA trajectory: {para_fits[best_para].model_name}
    R²: {para_fits[best_para].r_squared:.4f}
    e-governed: {'YES' if para_e_governed else 'NO'}
    ΔAIC (e-governed vs non-e-governed): {para_delta:.2f if para_delta is not None else 'N/A'}
    Final RSA: {para_rsa_mean[-1]:.4f}
""")
        
        # Interpretation
        if hier_e_governed and not para_e_governed:
            print("=" * 60)
            print("INTERPRETATION: HYPOTHESIS SUPPORTED")
            print("=" * 60)
            print("""
  Compositional REPRESENTATIONAL structure emerges following e-governed dynamics.
  Non-compositional structure does NOT.
  
  This is the KEY FINDING:
  When we measure the STATE (representational structure) rather than
  OUTPUT (accuracy), compositional tasks show e-governed dynamics
  while non-compositional tasks show different dynamics.
  
  This supports Danan (2025): The state-dependent growth of
  compositional representations follows dS/dt ∝ S → e-governed.
""")
        elif hier_e_governed and para_e_governed:
            # Check if there's a QUANTITATIVE difference
            if hier_delta is not None and para_delta is not None:
                print("=" * 60)
                print("INTERPRETATION: BOTH e-GOVERNED, CHECKING STRENGTH")
                print("=" * 60)
                print(f"""
  Both conditions show e-governed RSA dynamics.
  
  However, the STRENGTH of evidence differs:
  - Hierarchical ΔAIC (e vs non-e): {hier_delta:.2f}
  - Parallel ΔAIC (e vs non-e): {para_delta:.2f}
""")
                if hier_delta > para_delta + 10:
                    print("""
  Compositional condition shows STRONGER e-governed signature.
  This partial support suggests compositional structure amplifies
  e-governed dynamics even if it doesn't exclusively produce them.
""")
                else:
                    print("""
  Both conditions show similar e-governed strength.
  The dynamics of representational structure may be dominated
  by optimization rather than task structure.
""")
            else:
                print("=" * 60)
                print("INTERPRETATION: BOTH SHOW e-GOVERNED DYNAMICS")
                print("=" * 60)
        elif not hier_e_governed and not para_e_governed:
            print("=" * 60)
            print("INTERPRETATION: NEITHER SHOWS e-GOVERNED RSA DYNAMICS")
            print("=" * 60)
            print("""
  Representational structure evolution follows power law or linear dynamics
  in both conditions. This suggests:
  1. RSA dynamics differ from accuracy dynamics
  2. Representational change may not follow simple exponential models
  3. The hypothesis needs further refinement
""")
        else:
            print("=" * 60)
            print("INTERPRETATION: OPPOSITE PATTERN")
            print("=" * 60)
            print("""
  Non-compositional shows e-governed RSA, compositional does not.
  This contradicts the hypothesis.
""")
    
    print(f"\nFigures saved to {save_dir}/")
    
    return {
        'hier_fits': hier_fits,
        'para_fits': para_fits,
        'hier_rsa_mean': hier_rsa_mean,
        'para_rsa_mean': para_rsa_mean,
        'hier_rsa_std': hier_rsa_std,
        'para_rsa_std': para_rsa_std,
        'hier_acc_mean': hier_acc_mean,
        'para_acc_mean': para_acc_mean,
        'epochs': epochs,
        'all_hierarchical': all_hierarchical,
        'all_parallel': all_parallel,
        'hier_theoretical_rdm': hier_theoretical_rdm,
        'para_theoretical_rdm': para_theoretical_rdm
    }


if __name__ == "__main__":
    results = run_study4b(
        n_runs=10,
        n_bits=8,
        n_epochs=500,
        hidden_dim=64,
        lr=0.01,
        save_dir="figures"
    )