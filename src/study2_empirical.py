"""
Study 2: Empirical Test of Self-Referential Dynamics
=====================================================

PURPOSE: Test whether GENUINE self-reference produces e-governed dynamics.

HYPOTHESIS (Danan, 2025):
  Self-referential systems exhibit e-governed learning dynamics.

STATUS: EMPIRICAL HYPOTHESIS TEST
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    SelfReferentialNetwork, 
    NonSelfReferentialNetwork,
    compute_self_referential_loss,
    compute_external_loss
)
from src.utils import extract_time_constants, print_model_comparison, FitResult, compute_aic_bic

from scipy.optimize import curve_fit


# =============================================================================
# ROBUST CURVE FITTING FOR LEARNING CURVES
# =============================================================================

def exponential_approach(t, y_max, k, y0):
    """y(t) = y_max - (y_max - y0) * e^(-kt)"""
    return y_max - (y_max - y0) * np.exp(-k * t)

def logistic_model(t, K, k, t0):
    """y(t) = K / (1 + e^(-k(t - t0)))"""
    return K / (1 + np.exp(-k * (t - t0)))

def power_law_learning(t, a, b, c):
    """y(t) = c - a / (t + 1)^b"""
    return c - a / np.power(t + 1, b)

def linear_model(t, y0, slope):
    """y(t) = y0 + slope * t"""
    return y0 + slope * t


def fit_learning_curves_robust(t: np.ndarray, y: np.ndarray, 
                                verbose: bool = False) -> Dict[str, FitResult]:
    """
    Robust curve fitting for learning curves.
    Handles edge cases where y_max ≈ y_min.
    """
    results = {}
    n = len(t)
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_mean = y.mean()
    t_max = t.max()
    
    ss_total = np.sum((y - y_mean)**2)
    
    if verbose:
        print(f"  Data range: [{y_min:.4f}, {y_max:.4f}], range={y_range:.4f}")
    
    # If data is essentially flat, only linear makes sense
    if y_range < 0.01:
        if verbose:
            print("  WARNING: Data range < 0.01, learning curve is flat")
        # Just fit linear
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
        # Ensure bounds make sense
        y_max_bound = min(y_max * 1.2, 1.0) if y_max < 1 else y_max * 1.2
        y_min_bound = max(y_min * 0.8, 0) if y_min > 0 else y_min - 0.1
        
        if y_max_bound > y_min and y_max > y_min_bound:
            p0 = [y_max * 1.05, 0.01, y_min]
            bounds = ([y_max * 0.95, 1e-5, y_min_bound], 
                      [y_max_bound, 1.0, y_min * 1.1 + 0.01])
            
            # Validate bounds
            for i in range(len(p0)):
                if bounds[0][i] >= bounds[1][i]:
                    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    break
            
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
            
            for i in range(len(p0)):
                if bounds[0][i] >= bounds[1][i]:
                    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    break
            
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
            
            for i in range(len(p0)):
                if bounds[0][i] >= bounds[1][i]:
                    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                    break
            
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
    
    # 4. Linear (always try)
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
# DATA GENERATION - MAKE IT HARDER
# =============================================================================

def generate_hard_classification_data(
    n_samples: int = 5000,
    n_features: int = 30,
    n_classes: int = 15,
    noise: float = 1.0,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate VERY hard classification data.
    
    Key: We need the network to learn SLOWLY so we can observe dynamics.
    - More classes (15)
    - Higher noise (1.0)
    - More features with correlation
    """
    np.random.seed(seed)
    
    # Generate class centers - closer together for more overlap
    centers = np.random.randn(n_classes, n_features) * 0.8
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # High noise creates significant overlap
        X_class = centers[i] + np.random.randn(samples_per_class, n_features) * noise
        X.append(X_class)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Shuffle
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    
    return torch.tensor(X), torch.tensor(y)


# =============================================================================
# TRAINING FUNCTIONS - SLOWER LEARNING
# =============================================================================

def train_self_referential(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 128,
    lr: float = 0.001,  # SLOWER
    hidden_dim: int = 32,  # SMALLER - harder to learn
    self_weight: float = 0.3,
    eval_every: int = 1,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Train self-referential network with slower learning."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X.shape[1]
    n_classes = len(torch.unique(y))
    
    model = SelfReferentialNetwork(n_features, hidden_dim, n_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # SGD is slower than Adam
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    self_errors = []
    epochs_recorded = []
    
    n_samples = len(X)
    
    for epoch in range(n_epochs):
        model.train()
        model.reset_hidden()
        
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        epoch_self_error = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            
            output, predicted_hidden, actual_hidden = model(X_batch)
            
            class_loss = criterion(output, y_batch)
            total_loss, self_err = compute_self_referential_loss(
                class_loss, predicted_hidden, actual_hidden, self_weight
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_self_error.append(self_err)
        
        if epoch % eval_every == 0:
            model.eval()
            model.reset_hidden()
            
            with torch.no_grad():
                output, _, _ = model(X)
                preds = output.argmax(dim=1)
                acc = (preds == y).float().mean().item()
            
            accuracies.append(acc)
            self_errors.append(np.mean(epoch_self_error))
            epochs_recorded.append(epoch)
            
            if verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch}: acc={acc:.4f}")
    
    return {
        'epochs': np.array(epochs_recorded),
        'accuracy': np.array(accuracies),
        'self_error': np.array(self_errors),
        'condition': 'self_referential'
    }


def train_non_self_referential(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 128,
    lr: float = 0.001,
    hidden_dim: int = 32,
    external_weight: float = 0.3,
    eval_every: int = 1,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """Train non-self-referential network (control)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_features = X.shape[1]
    n_classes = len(torch.unique(y))
    
    model = NonSelfReferentialNetwork(n_features, hidden_dim, n_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    external_errors = []
    epochs_recorded = []
    
    n_samples = len(X)
    
    for epoch in range(n_epochs):
        model.train()
        
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        epoch_ext_error = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            
            output, predicted_input, actual_input = model(X_batch)
            
            class_loss = criterion(output, y_batch)
            total_loss, ext_err = compute_external_loss(
                class_loss, predicted_input, actual_input, external_weight
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_ext_error.append(ext_err)
        
        if epoch % eval_every == 0:
            model.eval()
            
            with torch.no_grad():
                output, _, _ = model(X)
                preds = output.argmax(dim=1)
                acc = (preds == y).float().mean().item()
            
            accuracies.append(acc)
            external_errors.append(np.mean(epoch_ext_error))
            epochs_recorded.append(epoch)
            
            if verbose and epoch % 50 == 0:
                print(f"    Epoch {epoch}: acc={acc:.4f}")
    
    return {
        'epochs': np.array(epochs_recorded),
        'accuracy': np.array(accuracies),
        'external_error': np.array(external_errors),
        'condition': 'non_self_referential'
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_study2(
    n_runs: int = 5,
    n_epochs: int = 500,
    n_samples: int = 5000,
    n_classes: int = 15,
    noise: float = 1.0,
    save_dir: str = "figures"
) -> Dict:
    """Run Study 2: Empirical test."""
    
    print("=" * 70)
    print("STUDY 2: EMPIRICAL TEST OF SELF-REFERENTIAL DYNAMICS")
    print("=" * 70)
    print("\nHYPOTHESIS (Danan, 2025):")
    print("  Self-referential systems exhibit e-governed learning dynamics.")
    print("\nDESIGN:")
    print("  Condition A: Network predicts OWN HIDDEN STATES (self-referential)")
    print("  Condition B: Network predicts EXTERNAL TARGETS (control)")
    print("\nSTATUS: EMPIRICAL HYPOTHESIS TEST")
    print("-" * 70)
    
    print(f"\nGenerating data: {n_samples} samples, {n_classes} classes, noise={noise}")
    X, y = generate_hard_classification_data(
        n_samples=n_samples,
        n_classes=n_classes,
        noise=noise,
        n_features=30,
        seed=42
    )
    
    # Check baseline accuracy
    baseline_acc = 1.0 / n_classes
    print(f"Baseline (random) accuracy: {baseline_acc:.4f}")
    
    all_self_ref = []
    all_control = []
    
    print(f"\nRunning {n_runs} replications ({n_epochs} epochs each)...")
    print("(Using SGD with lr=0.001 for slower learning dynamics)\n")
    
    for run in range(n_runs):
        print(f"--- Replication {run+1}/{n_runs} ---")
        
        print("  Training self-referential...")
        self_ref_data = train_self_referential(
            X, y, n_epochs=n_epochs, seed=run*100+42, verbose=True
        )
        
        print("  Training control...")
        control_data = train_non_self_referential(
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
    
    # Print diagnostics
    print("\n" + "=" * 60)
    print("LEARNING CURVE DIAGNOSTICS")
    print("=" * 60)
    print(f"\nSelf-Referential:")
    print(f"  Start accuracy: {self_ref_mean[0]:.4f}")
    print(f"  End accuracy: {self_ref_mean[-1]:.4f}")
    print(f"  Range: {self_ref_mean[-1] - self_ref_mean[0]:.4f}")
    
    print(f"\nControl:")
    print(f"  Start accuracy: {control_mean[0]:.4f}")
    print(f"  End accuracy: {control_mean[-1]:.4f}")
    print(f"  Range: {control_mean[-1] - control_mean[0]:.4f}")
    
    # Fit models
    print("\n" + "=" * 60)
    print("CONDITION A: SELF-REFERENTIAL")
    print("=" * 60)
    
    self_ref_fits = fit_learning_curves_robust(epochs, self_ref_mean, verbose=True)
    print_model_comparison(self_ref_fits, "Self-Referential Learning Curve")
    
    self_ref_tc = extract_time_constants(epochs, self_ref_mean)
    if 'ratio_865_632' in self_ref_tc:
        print(f"\nTime constant ratios (e-governed signature):")
        print(f"  t(86.5%)/τ = {self_ref_tc['ratio_865_632']:.3f} (expected: 2.0)")
        print(f"  t(95.0%)/τ = {self_ref_tc['ratio_950_632']:.3f} (expected: 3.0)")
    
    print("\n" + "=" * 60)
    print("CONDITION B: NON-SELF-REFERENTIAL (CONTROL)")
    print("=" * 60)
    
    control_fits = fit_learning_curves_robust(epochs, control_mean, verbose=True)
    print_model_comparison(control_fits, "Non-Self-Referential Learning Curve")
    
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
    ax.plot(epochs, self_ref_mean, 'o-', color='steelblue', markersize=2, label='Data')
    for name, fit in self_ref_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2, 
                label=f"{fit.model_name} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Chance ({baseline_acc:.2f})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('A: Self-Referential\n(Predicts Own Hidden States)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Log-linear plot
    ax = axes[0, 1]
    asymptote = self_ref_mean[-1] * 1.02
    gap = asymptote - self_ref_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='steelblue', markersize=2)
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
    ax.plot(epochs, control_mean, 'o-', color='coral', markersize=2, label='Data')
    for name, fit in control_fits.items():
        ax.plot(epochs, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Chance ({baseline_acc:.2f})')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('B: Non-Self-Referential\n(Predicts External Targets)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Log-linear plot
    ax = axes[1, 1]
    asymptote = control_mean[-1] * 1.02
    gap = asymptote - control_mean
    gap = np.maximum(gap, 1e-6)
    ax.plot(epochs, np.log(gap), 'o-', color='coral', markersize=2)
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
    
    plt.suptitle('Study 2: Self-Referential vs Non-Self-Referential Learning Dynamics',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/study2_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/study2_results.pdf', bbox_inches='tight')
    
    # Time constants figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (tc, title, color) in enumerate([
        (self_ref_tc, 'A: Self-Referential', 'steelblue'),
        (control_tc, 'B: Non-Self-Referential', 'coral')
    ]):
        ax = axes2[idx]
        if 'ratio_865_632' in tc:
            observed = [tc['ratio_865_632'], tc['ratio_950_632']]
            expected = [2.0, 3.0]
            x = np.arange(2)
            ax.bar(x - 0.15, observed, 0.3, label='Observed', color=color)
            ax.bar(x + 0.15, expected, 0.3, label='Expected (e-governed)', color='gray', alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(['t(86.5%)/τ', 't(95.0%)/τ'])
            ax.axhline(2.0, color='gray', linestyle='--', alpha=0.3)
            ax.axhline(3.0, color='gray', linestyle='--', alpha=0.3)
        ax.set_ylabel('Ratio')
        ax.set_title(title)
        ax.legend()
    
    plt.suptitle('Time Constant Analysis: e-Governed Signature Test', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/study2_time_constants.png', dpi=150, bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 70)
    print("STUDY 2 SUMMARY")
    print("=" * 70)
    
    if self_ref_fits and control_fits:
        best_self_ref = min(self_ref_fits.keys(), key=lambda x: self_ref_fits[x].aic)
        best_control = min(control_fits.keys(), key=lambda x: control_fits[x].aic)
        
        self_ref_e_governed = best_self_ref in ['exponential', 'logistic']
        control_e_governed = best_control in ['exponential', 'logistic']
        
        print(f"""
HYPOTHESIS (Danan, 2025):
  Self-referential systems exhibit e-governed learning dynamics.

RESULTS:
  Condition A (Self-Referential):
    Best fit: {self_ref_fits[best_self_ref].model_name}
    R²: {self_ref_fits[best_self_ref].r_squared:.4f}
    e-governed: {'YES' if self_ref_e_governed else 'NO'}
    
  Condition B (Non-Self-Referential):
    Best fit: {control_fits[best_control].model_name}
    R²: {control_fits[best_control].r_squared:.4f}
    e-governed: {'YES' if control_e_governed else 'NO'}
""")
        
        # Interpretation
        if self_ref_e_governed and not control_e_governed:
            print("INTERPRETATION: Hypothesis SUPPORTED")
            print("  Self-referential condition shows e-governed dynamics.")
            print("  Non-self-referential condition shows different dynamics.")
            print("  This is consistent with Danan (2025).")
        elif self_ref_e_governed and control_e_governed:
            print("INTERPRETATION: Both conditions show e-governed dynamics")
            print("  Possible explanations:")
            print("  1. All gradient-based learning follows e-dynamics")
            print("  2. The manipulation was insufficient to differentiate")
            print("  3. The hypothesis may need refinement")
        elif not self_ref_e_governed and control_e_governed:
            print("INTERPRETATION: OPPOSITE pattern (control e-governed, self-ref not)")
            print("  This contradicts the hypothesis.")
        else:
            print("INTERPRETATION: Neither condition shows e-governed dynamics")
            print("  Possible explanations:")
            print("  1. Task/architecture not suited for testing")
            print("  2. Hypothesis requires revision")
            print("  3. Need longer training or different learning rate")
    
    print(f"\nFigures saved to {save_dir}/")
    
    return {
        'self_ref_fits': self_ref_fits,
        'control_fits': control_fits,
        'self_ref_tc': self_ref_tc,
        'control_tc': control_tc,
        'self_ref_mean': self_ref_mean,
        'control_mean': control_mean,
        'self_ref_std': self_ref_std,
        'control_std': control_std,
        'epochs': epochs,
        'all_self_ref': all_self_ref,
        'all_control': all_control
    }


if __name__ == "__main__":
    results = run_study2(n_runs=5, n_epochs=500)