#!/usr/bin/env python3
"""
Run Study 3: Non-Gradient Self-Reference Test

Tests whether self-reference produces e-governed dynamics
WITHOUT information-theoretic optimization (Hebbian learning).
"""

from src.study3_hebbian import run_study3

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING STUDY 3: HEBBIAN (NON-GRADIENT) SELF-REFERENCE TEST")
    print("="*70 + "\n")
    
    results = run_study3(
        n_runs=10,
        n_epochs=500,
        n_samples=3000,
        n_classes=5,
        save_dir="figures"
    )
    
    print("\nStudy 3 complete! Check figures/ directory for plots.")