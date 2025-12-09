#!/usr/bin/env python3
"""Run Study 1: Mathematical Verification"""

from src.study1_mathematical import run_study1

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING STUDY 1: MATHEMATICAL VERIFICATION")
    print("="*70 + "\n")
    
    results = run_study1(
        beta_values=[0.0, 0.5, 0.75, 1.0],
        A0=10.0,
        k=0.1,
        K=500.0,
        T=100.0,
        save_dir="figures"
    )
    
    print("\nStudy 1 complete! Check figures/ directory for plots.")