# Self-Referential Dynamics

Testing the hypothesis from Danan (2025) "Recursive Abstraction" that self-referential systems exhibit *e*-governed dynamics.

## Key Finding

**Multiple convergent pathways produce *e*-governed dynamics:**

| Pathway | Mechanism | Study |
|---------|-----------|-------|
| Compositional growth | dA/dt ∝ A^β, β → 1 | Study 1 ✓ |
| Information-theoretic optimization | Cross-entropy uses ln() | Study 2 ✓ |
| Self-referential computation | dS/dt = f(S, Ŝ) | Study 3 (tentative) |

Self-reference is *sufficient* but not uniquely *necessary* for *e*-governed dynamics.

---

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
python3 run_study1.py  # Mathematical verification (~30 sec)
python3 run_study2.py  # Gradient-based learning (~5 min)
python3 run_study3.py  # Hebbian learning (~3 min)
```

---

## Theoretical Framework

Euler's number *e* (≈ 2.71828) emerges universally in systems where **the rate of change depends on current state**:

```
dY/dt = kY  →  Y(t) = Y₀e^(kt)
```

Danan (2025) proposes three convergent pathways to *e*-governed dynamics:

1. **Compositionality**: New abstractions depend on existing ones → dA/dt ∝ A^β
2. **Information theory**: Optimal inference uses log-likelihoods → ln() → *e*
3. **Self-reference**: System models own state → dS/dt = f(S, Ŝ)

**The empirical challenge**: If multiple mechanisms produce the same signature, how do we isolate each contribution?

---

## Studies

### Study 1: Mathematical Verification

**Purpose**: Verify that compositional dynamics produce *e*-governed growth. This is **established mathematics** (Strogatz, 2015), not hypothesis testing.

**Method**: Simulate dA/dt = kA^β(1 - A/K) with β ∈ {0, 0.5, 0.75, 1.0}

**Results**:

| β | Best Model | R² | Status |
|---|-----------|-----|--------|
| 0.0 | Linear (NOT *e*-governed) | 0.991 | ✓ Confirmed |
| 0.5 | Power Law | 0.998 | ~ Intermediate |
| 0.75 | Power Law | 0.999 | ~ Intermediate |
| 1.0 | **Logistic (*e*-governed)** | 0.999 | ✓ Confirmed |

**Conclusion**: Implementation correctly reproduces known mathematics. β → 1 produces *e*-governed dynamics.

---

### Study 2: Gradient-Based Learning

**Purpose**: Test whether self-referential neural networks show *e*-governed learning dynamics.

**Key Insight**: TRUE self-reference = predicting own **internal states** (hidden layer activations), not just accuracy.

**Design**:
- **Self-referential**: Network predicts its own hidden states at t+1 from states at t
- **Control**: Network predicts external targets (input reconstruction)
- Both trained with SGD on cross-entropy loss

**Results**:

| Condition | Best Model | R² | ΔAIC vs Linear |
|-----------|-----------|-----|----------------|
| Self-Referential | Logistic (*e*-governed) | 0.9946 | 1526.50 |
| Non-Self-Referential | Logistic (*e*-governed) | 0.9947 | 1532.18 |

**Conclusion**: Both conditions show identical *e*-governed dynamics. **Gradient descent dominates the signature** (cross-entropy involves ln()).

**Implication**: To isolate self-reference, we need non-gradient learning.

---

### Study 3: Hebbian Learning (Non-Gradient)

**Purpose**: Test whether self-reference produces *e*-governed dynamics **without** information-theoretic optimization.

**Method**: Hebbian learning (Hebb, 1949; Oja, 1982)
- NO backpropagation
- NO error signal
- NO log-likelihood
- Pure correlation-based: Δw ∝ pre × post

**Results**:

| Condition | Best Model | R² | ΔAIC vs Linear |
|-----------|-----------|-----|----------------|
| Self-Referential | Exponential (*e*-governed) | 0.7647 | **56.65** |
| Non-Self-Referential | Exponential (*e*-governed) | 0.4303 | 3.28 |

**Interpretation** (Burnham & Anderson, 2002):
- ΔAIC < 2: Models equivalent
- ΔAIC > 10: Strong evidence

**Self-referential**: Strong evidence for exponential (ΔAIC = 56.65)  
**Control**: No evidence for exponential over linear (ΔAIC = 3.28)

**Limitation**: Both networks showed weak learning (accuracy near chance). The differential signal is suggestive but requires replication with stronger non-gradient algorithms.

**Conclusion**: Tentative evidence that self-reference contributes to *e*-governed dynamics independent of gradient descent.

---

## Summary of Findings

```
STUDY 1: β → 1 produces e-governed dynamics
         STATUS: CONFIRMED (mathematical verification)

STUDY 2: Both self-ref and control show e-governed dynamics with gradient descent
         STATUS: CONFIRMED (gradient descent dominates)

STUDY 3: Self-ref shows stronger e-signature than control with Hebbian learning
         STATUS: TENTATIVE (weak learning limits power)
```

**Overall**: The hypothesis that self-referential systems exhibit *e*-governed dynamics is **supported but not uniquely confirmed**. Multiple pathways converge on *e*-governed dynamics, and self-reference appears to be one of them.

---

## Repository Structure

```
self-referential-dynamics/
├── src/
│   ├── __init__.py
│   ├── models.py              # Neural network architectures
│   ├── dynamics.py            # ODE simulation for Study 1
│   ├── study1_mathematical.py # Mathematical verification
│   ├── study2_empirical.py    # Gradient-based learning test
│   ├── study3_hebbian.py      # Hebbian learning test
│   ├── analysis.py            # Model fitting utilities
│   └── utils.py               # AIC/BIC, time constants
├── figures/                   # Generated plots
├── run_study1.py
├── run_study2.py
├── run_study3.py
├── run_all.py
├── requirements.txt
└── README.md
```

---

## Future Directions

1. **Evolutionary algorithms**: Gradient-free optimization to further isolate self-reference
2. **BCM learning**: Stronger Hebbian variant (Bienenstock, Cooper, & Munro, 1982)
3. **Biological data**: Meta-cognition vs simple conditioning learning curves
4. **Clinical applications**: *e*-governed degradation in neurodegenerative self-referential deficits

---

## References

- Bienenstock, E. L., Cooper, L. N., & Munro, P. W. (1982). Theory for the development of neuron selectivity. *Journal of Neuroscience*, 2(1), 32-48.
- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference* (2nd ed.). Springer.
- Danan, H. (2025). Recursive abstraction: When computation requires self-reference. *Working paper*.
- Hebb, D. O. (1949). *The Organization of Behavior*. Wiley.
- Heathcote, A., Brown, S., & Mewhort, D. J. K. (2000). The power law repealed. *Psychonomic Bulletin & Review*, 7(2), 185-207.
- Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620-630.
- Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition. In *Cognitive Skills and Their Acquisition*. Erlbaum.
- Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267-273.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos* (2nd ed.). Westview Press.

---

## Citation

```bibtex
@misc{danan2025selfreferential,
  author = {Danan, Hillary},
  title = {Self-Referential Dynamics: Testing e-Governed Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HillaryDanan/self-referential-dynamics}
}
```

---

## License

MIT License. See LICENSE file for details.

---

*Questions or feedback? Open an issue or contact hillary@example.com*