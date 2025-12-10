#!/usr/bin/env python3
"""
Run Study 5: Genetic Programming (Compositional Program Synthesis)

Tests whether GENUINELY COMPOSITIONAL systems exhibit e-governed dynamics.

This removes the neural network / gradient descent confound:
- Programs are compositional by definition
- Evolution is gradient-free
- If e-dynamics emerge, it's from compositionality, not optimization

Reference: Koza (1992). Genetic Programming. MIT Press.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.study5_gp import run_study5

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING STUDY 5: GENETIC PROGRAMMING (COMPOSITIONAL)")
    print("="*70 + "\n")
    
    results = run_study5(
        n_runs=10,
        n_bits=6,           # 6-bit parity (64 samples, tractable for GP)
        n_generations=150,  # Enough to see dynamics
        pop_size=200,       # Reasonable population
        save_dir="figures"
    )
    
    print("\nStudy 5 complete! Check figures/ directory for plots.")