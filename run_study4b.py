#!/usr/bin/env python3
"""
Run Study 4b: RSA Analysis of Compositional vs Non-Compositional Dynamics

Tests whether compositional REPRESENTATIONAL STRUCTURE emerges following
e-governed dynamics, rather than just accuracy.

This addresses the key insight from feedback:
"The state IS the system's representational capacity itself."

We measure e-dynamics of REPRESENTATION (via RSA), not just accuracy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.study4b_rsa import run_study4b

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING STUDY 4b: RSA ANALYSIS OF REPRESENTATIONAL DYNAMICS")
    print("="*70 + "\n")
    
    results = run_study4b(
        n_runs=10,
        n_bits=8,
        n_epochs=500,
        hidden_dim=64,
        lr=0.01,
        save_dir="figures"
    )
    
    print("\nStudy 4b complete! Check figures/ directory for plots.")