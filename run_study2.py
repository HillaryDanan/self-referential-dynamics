#!/usr/bin/env python3
"""Run Study 2: Empirical Test of Self-Referential Dynamics"""

from src.study2_empirical import run_study2

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING STUDY 2: EMPIRICAL TEST")
    print("="*70 + "\n")
    
    results = run_study2(
        n_runs=10,
        n_epochs=300,
        n_samples=5000,
        n_classes=10,
        noise=0.5,
        save_dir="figures"
    )
    
    print("\nStudy 2 complete! Check figures/ directory for plots.")