# Self-Referential Dynamics

Testing the hypothesis from Danan (2025) "Recursive Abstraction" that self-referential systems exhibit e-governed dynamics.

## Quick Start
```bash
# Clone and enter
git clone https://github.com/HillaryDanan/self-referential-dynamics.git
cd self-referential-dynamics

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run everything
python3 run_all.py

# Or run individually
python3 run_study1.py  # Mathematical verification
python3 run_study2.py  # Empirical test
```

## Studies

### Study 1: Mathematical Verification
Verifies that compositional dynamics (dA/dt = kA^β with β → 1) produce e-governed growth.
This is **established mathematics** (Strogatz, 2015).

### Study 2: Empirical Test  
Tests whether **genuine self-reference** (predicting own hidden states) produces e-governed learning dynamics.

**Key insight**: True self-reference means predicting your own *internal states*, not just accuracy.

## References

- Burnham & Anderson (2002). Model Selection and Multimodel Inference.
- Heathcote, Brown, & Mewhort (2000). Psychonomic Bulletin & Review.
- Jaynes (1957). Physical Review.
- Shannon (1948). Bell System Technical Journal.
- Strogatz (2015). Nonlinear Dynamics and Chaos.