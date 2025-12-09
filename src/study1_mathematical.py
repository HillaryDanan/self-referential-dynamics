"""
Study 1: Mathematical Verification
==================================

PURPOSE: Verify that compositional dynamics produce e-governed growth.
STATUS: MATHEMATICAL VERIFICATION (established mathematics, not hypothesis).

Reference: Strogatz (2015), Tenenbaum & Pollard (1985)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamics import simulate_compositional_growth
from src.utils import fit_all_models, extract_time_constants, print_model_comparison


def run_study1(
    beta_values: List[float] = [0.0, 0.5, 0.75, 1.0],
    A0: float = 10.0,
    k: float = 0.1,
    K: float = 500.0,
    T: float = 100.0,
    noise_std: float = 0.02,
    save_dir: str = "figures"
) -> Dict:
    """
    Run Study 1: Mathematical verification.
    """
    
    print("=" * 70)
    print("STUDY 1: MATHEMATICAL VERIFICATION")
    print("=" * 70)
    print("\nPURPOSE: Verify that compositional dynamics (β → 1) produce")
    print("         e-governed growth. This is ESTABLISHED MATHEMATICS.")
    print("\nSTATUS: Mathematical verification, NOT empirical hypothesis testing")
    print("-" * 70)
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    predictions = {
        0.0: ("Linear", False),
        0.5: ("Intermediate", None),
        0.75: ("Near-exponential", True),
        1.0: ("Exponential/Logistic", True)
    }
    
    colors = {
        'exponential': '#e41a1c',
        'logistic': '#377eb8',
        'power_law': '#4daf4a',
        'linear': '#984ea3'
    }
    
    for idx, beta in enumerate(beta_values):
        print(f"\n{'='*50}")
        print(f"β = {beta}")
        print(f"{'='*50}")
        
        # Simulate
        t, A = simulate_compositional_growth(
            A0=A0, k=k, beta=beta, K=K, T=T,
            noise_std=noise_std, seed=42 + idx
        )
        
        # Fit models (NOT bounded for growth data)
        fits = fit_all_models(t, A, is_bounded=True, verbose=True)
        results[beta] = {'t': t, 'A': A, 'fits': fits}
        
        # Print comparison
        print_model_comparison(fits, f"β = {beta}")
        
        if not fits:
            print("WARNING: No models fit successfully!")
            continue
        
        # Determine best model
        best_name = min(fits.keys(), key=lambda x: fits[x].aic)
        best_fit = fits[best_name]
        
        # Check prediction
        predicted_type, predicted_e_governed = predictions[beta]
        actual_e_governed = best_name in ['exponential', 'logistic']
        
        if predicted_e_governed is not None:
            status = "✓ CONFIRMED" if actual_e_governed == predicted_e_governed else "✗ UNEXPECTED"
        else:
            status = "~ INTERMEDIATE"
        
        print(f"\nPredicted: {predicted_type}")
        print(f"Best fit: {best_fit.model_name}")
        print(f"Status: {status}")
        
        # Time constant analysis
        tc = extract_time_constants(t, A)
        if 'ratio_865_632' in tc:
            print(f"\nTime constant ratios (e-governed signature):")
            print(f"  t(86.5%)/τ = {tc['ratio_865_632']:.3f} (expected: 2.0)")
            print(f"  t(95.0%)/τ = {tc['ratio_950_632']:.3f} (expected: 3.0)")
        
        results[beta]['time_constants'] = tc
        results[beta]['status'] = status
        
        # Plot
        ax = axes[idx]
        ax.scatter(t, A, alpha=0.3, s=10, color='black', label='Data')
        
        for name, fit in fits.items():
            ax.plot(t, fit.predictions, color=colors.get(name, 'gray'),
                    linewidth=2, alpha=0.8, label=fit.model_name)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Abstraction Count', fontsize=12)
        ax.set_title(f'β = {beta}\nBest: {best_fit.model_name}\n{status}', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Study 1: Compositional Dynamics Across β Values\n(Mathematical Verification)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/study1_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/study1_results.pdf', bbox_inches='tight')
    print(f"\nFigure saved to {save_dir}/study1_results.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("STUDY 1 SUMMARY")
    print("=" * 70)
    print("""
MATHEMATICAL CLAIM (Strogatz, 2015; Tenenbaum & Pollard, 1985):
  - State-dependent growth dA/dt = kA^β with β → 1 produces e-governed dynamics
  
VERIFICATION RESULTS:
""")
    
    for beta in beta_values:
        if results[beta]['fits']:
            best = min(results[beta]['fits'].keys(), key=lambda x: results[beta]['fits'][x].aic)
            status = results[beta].get('status', 'N/A')
            print(f"  β = {beta}: Best fit = {results[beta]['fits'][best].model_name} [{status}]")
        else:
            print(f"  β = {beta}: No models fit")
    
    print("""
INTERPRETATION:
  This verifies our implementation correctly reproduces known mathematics.
  This is NOT evidence for Danan (2025) - the empirical hypothesis is tested in Study 2.
""")
    
    return results


if __name__ == "__main__":
    results = run_study1()