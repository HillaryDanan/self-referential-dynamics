# Dissociating Sources of *e*-Governed Dynamics in Learning Systems: Gradient Descent, Compositionality, and Self-Reference

**Hillary Danan, PhD**

*December 2025*

---

## Abstract

We report a systematic empirical investigation of the hypothesis that self-referential and compositional systems exhibit learning dynamics governed by Euler's number *e* (≈ 2.71828). Across five studies employing neural networks with gradient descent, Hebbian learning, and genetic programming, we find that *e*-governed dynamics (exponential/logistic learning curves) consistently emerge when gradient-based optimization is present and disappear when it is absent—regardless of whether systems are self-referential, whether tasks are compositional, or whether representations are hierarchical. Study 1 verified the mathematical foundation linking compositional growth (dA/dt ∝ A^β, β → 1) to *e*-governed dynamics. Studies 2 and 4 found that both self-referential and non-self-referential neural networks, and both compositional and non-compositional tasks, produce identical *e*-governed dynamics when trained with gradient descent. Study 3 (Hebbian learning) showed a differential signal: self-referential networks exhibited stronger *e*-governed signatures (ΔAIC = 56.65) than controls (ΔAIC = 3.28), though weak learning limited statistical power. Study 4b revealed that neural networks do not spontaneously develop compositional representations even when solving compositional tasks (RSA = 0.08 vs. theoretical). Study 5 (genetic programming) found that genuinely compositional systems without gradient descent produce power law dynamics (R² = 0.96), not *e*-governed dynamics. Together, these findings suggest that *e*-governed learning dynamics in artificial systems primarily reflect the information-theoretic structure of gradient descent (log-likelihoods in cross-entropy loss), not compositionality or self-reference per se. We propose a refined hypothesis: *e*-governed dynamics should emerge specifically in systems where **learning itself is compositional**—where new knowledge is constructed from existing knowledge—rather than in systems that merely have compositional representations or solve compositional tasks.

**Keywords:** Euler's number, learning dynamics, compositionality, self-reference, gradient descent, genetic programming, representational similarity analysis

---

## 1. Introduction

### 1.1 Theoretical Background

Euler's number *e* (≈ 2.71828) appears universally in systems where the rate of change depends on current state. Any system governed by dY/dt = kY produces solutions Y(t) = Y₀e^(kt)—a mathematical theorem arising from the definition of the exponential function (Strogatz, 2015; Tenenbaum & Pollard, 1985). This structure underlies phenomena from radioactive decay to population growth to capacitor discharge.

Danan (2025) proposed that self-referential computational systems—those requiring models of their own internal states—exhibit *e*-governed dynamics as a consequence of their recursive structure. The theoretical argument proceeds through three convergent pathways:

1. **State-dependence**: Self-referential systems model their own state S, creating dynamics where dS/dt = f(S)
2. **Compositionality**: When abstractions compose to form new abstractions, growth depends on current abstraction count: dA/dt ∝ A^β, where β → 1 yields *e*-governed dynamics
3. **Information theory**: Optimal self-modeling under uncertainty involves entropy and log-likelihoods, naturally expressed in base *e* (Shannon, 1948; Jaynes, 1957)

These pathways suggest *e* is not an arbitrary constant but a necessary signature of state-dependent, self-referential computation.

### 1.2 The Empirical Challenge

If multiple mechanisms produce *e*-governed dynamics, empirical investigation faces a confound: any observed *e*-dynamics could arise from compositionality, self-reference, information-theoretic optimization, or some combination. To test the hypothesis rigorously, we must systematically manipulate each factor while controlling others.

A particular concern is gradient descent, the dominant optimization algorithm in modern machine learning. Gradient descent minimizes cross-entropy loss, which involves natural logarithms:

L = -Σ y log(ŷ)

The gradient ∂L/∂w inherently involves *e* through the logarithm. If gradient descent produces *e*-governed dynamics regardless of task or architecture, observed *e*-dynamics might reflect optimization rather than the cognitive structures hypothesized by Danan (2025).

### 1.3 The Present Investigation

We designed five studies to dissociate potential sources of *e*-governed dynamics:

**Study 1** (Mathematical Verification): Verify that compositional growth dynamics (dA/dt = kA^β) produce *e*-governed curves when β → 1.

**Study 2** (Gradient + Self-Reference): Compare self-referential versus non-self-referential neural networks trained with gradient descent.

**Study 3** (Hebbian Learning): Remove gradient descent; compare self-referential versus non-self-referential networks with correlation-based learning.

**Study 4** (Gradient + Task Structure): Compare compositional versus non-compositional tasks with identical neural network architecture and gradient descent.

**Study 4b** (Representational Structure): Measure whether compositional representations emerge using Representational Similarity Analysis (Kriegeskorte et al., 2008).

**Study 5** (Genetic Programming): Test genuinely compositional systems (symbolic programs) without gradient descent.

### 1.4 Predictions

Based on the theoretical framework:

- If **compositionality** drives *e*-dynamics: Compositional tasks/systems should show *e*-governed dynamics regardless of optimization method
- If **gradient descent** drives *e*-dynamics: All gradient-trained systems should show *e*-governed dynamics regardless of task/architecture
- If **self-reference** drives *e*-dynamics: Self-referential systems should show stronger *e*-governed signatures

---

## 2. General Methods

### 2.1 Model Fitting

We fit four candidate models to all learning curves:

1. **Exponential approach**: y(t) = y_max - (y_max - y₀)e^(-kt) [*e*-governed]
2. **Logistic**: y(t) = K / (1 + e^(-k(t-t₀))) [*e*-governed]
3. **Power law**: y(t) = c - a/(t+1)^b [NOT *e*-governed]
4. **Linear**: y(t) = y₀ + slope·t [NOT *e*-governed]

Models 1-2 contain *e* explicitly; models 3-4 do not.

### 2.2 Model Comparison

Following Burnham and Anderson (2002), we used the Akaike Information Criterion:

AIC = n·ln(SSE/n) + 2k

Model selection guidelines:
- ΔAIC < 2: Models statistically equivalent
- ΔAIC 4-7: Moderate evidence for preferred model
- ΔAIC > 10: Strong evidence for preferred model

### 2.3 Time Constant Analysis

For *e*-governed dynamics, characteristic time constants produce fixed ratios (Strogatz, 2015):
- At t = τ: 63.2% of asymptotic improvement (1 - 1/e)
- At t = 2τ: 86.5% (1 - 1/e²)
- At t = 3τ: 95.0% (1 - 1/e³)

Ratios of 2.0 and 3.0 for t(86.5%)/τ and t(95%)/τ serve as diagnostic signatures.

### 2.4 Software and Reproducibility

All analyses used Python 3.12 with NumPy 1.26, SciPy 1.11, and PyTorch 2.1. Curve fitting used scipy.optimize.curve_fit. Code is available at https://github.com/HillaryDanan/self-referential-dynamics.

---

## 3. Study 1: Mathematical Verification

### 3.1 Purpose

Verify that compositional growth dynamics produce *e*-governed curves when the compositionality exponent β → 1. This confirms our implementation reproduces established mathematics (Strogatz, 2015).

### 3.2 Method

We simulated compositional abstraction growth via numerical integration of:

dA/dt = k·A^β·(1 - A/K)

Parameters: k = 0.1, K = 500 (carrying capacity), A₀ = 10, T = 100 time units, dt = 0.5, measurement noise σ = 0.02·mean(A). We tested β ∈ {0.0, 0.5, 0.75, 1.0}.

### 3.3 Results

**Table 1.** Study 1: Mathematical Verification Results

| β | Best Model | R² | ΔAIC (2nd best) | Predicted | Status |
|---|-----------|-----|-----------------|-----------|--------|
| 0.0 | Linear | 0.9906 | 0.23 | Linear | ✓ Confirmed |
| 0.5 | Power Law | 0.9981 | 31.19 | Intermediate | Plausible |
| 0.75 | Power Law | 0.9988 | 26.17 | Near-exponential | Intermediate |
| 1.0 | Logistic | 0.9990 | 857.76 | Logistic/Exponential | ✓ Confirmed |

The critical predictions were confirmed: β = 0 produced linear (non-*e*-governed) dynamics, and β = 1 produced logistic (*e*-governed) dynamics with overwhelming evidence (ΔAIC = 857.76).

### 3.4 Discussion

Study 1 confirms that our implementation correctly reproduces established mathematical relationships. When growth follows dA/dt = kA (β = 1), the resulting dynamics are *e*-governed by mathematical necessity.

---

## 4. Study 2: Self-Reference with Gradient Descent

### 4.1 Purpose

Test whether self-referential neural networks exhibit *e*-governed learning dynamics when trained with gradient descent.

### 4.2 Method

#### 4.2.1 Architecture

**Self-Referential Network**: Three-layer MLP with auxiliary self-model predicting its own hidden states at time t+1 from states at time t.

**Non-Self-Referential Network**: Identical architecture with auxiliary model predicting external targets (input reconstruction).

#### 4.2.2 Training

Data: 5,000 samples, 10 classes, 20 features, noise σ = 0.5. Optimizer: SGD (lr = 0.001, momentum = 0.9). Epochs: 300. Replications: 10.

### 4.3 Results

**Table 2.** Study 2: Gradient-Based Self-Reference Results

| Condition | Best Model | R² | ΔAIC vs Linear | Time Constant Ratios |
|-----------|-----------|-----|----------------|---------------------|
| Self-Referential | Logistic | 0.9946 | 1526.50 | 1.669, 2.250 |
| Non-Self-Referential | Logistic | 0.9947 | 1532.18 | 1.676, 2.250 |

Both conditions showed virtually identical *e*-governed dynamics. The logistic model provided excellent fits (R² > 0.99) with overwhelming evidence over alternatives (ΔAIC > 1500). Time constant ratios were nearly identical between conditions.

### 4.4 Discussion

The hypothesis that self-referential systems exhibit *e*-governed dynamics was confirmed—but so did the non-self-referential control. This suggests gradient descent may dominate the dynamical signature regardless of architecture.

---

## 5. Study 3: Hebbian Learning (Gradient-Free)

### 5.1 Purpose

Test whether self-reference produces *e*-governed dynamics without gradient descent.

### 5.2 Method

We implemented Oja's normalized Hebbian rule (Oja, 1982):

Δw = η·y·(x - y·w)

This is pure correlation-based learning with no backpropagation, no error signal, and no logarithms.

**Self-Referential Hebbian**: Self-model learns via Hebbian correlation between successive hidden states.

**Non-Self-Referential Hebbian**: Auxiliary model predicts external inputs via Hebbian learning.

Training: 3,000 samples, 5 classes, 500 epochs, 10 replications.

### 5.3 Results

**Table 3.** Study 3: Hebbian Learning Results

| Condition | Best Model | R² | ΔAIC vs Linear |
|-----------|-----------|-----|----------------|
| Self-Referential | Exponential | 0.7647 | **56.65** |
| Non-Self-Referential | Exponential | 0.4303 | 3.28 |

Both conditions were technically best fit by exponential models. However, the strength of evidence differed substantially:

- Self-referential: ΔAIC = 56.65 over linear (strong evidence)
- Non-self-referential: ΔAIC = 3.28 over linear (no evidence; models equivalent per Burnham & Anderson, 2002)

#### 5.3.1 Limitations

Both networks showed weak learning (accuracy near chance, range < 0.08). The lower R² values (0.43-0.76) compared to Study 2 (>0.99) indicate noisier dynamics with limited statistical power.

### 5.4 Discussion

Despite methodological limitations, a differential signal emerged. Self-referential networks showed substantially stronger evidence for *e*-governed dynamics than controls when gradient descent was removed. This is suggestive—though not conclusive—evidence that self-reference may independently contribute to *e*-governed dynamics.

---

## 6. Study 4: Task Structure with Gradient Descent

### 6.1 Purpose

Test whether compositional task structure produces different dynamics than non-compositional tasks.

### 6.2 Method

**Hierarchical XOR (Compositional)**: Compute 8-bit parity ((((x₁⊕x₂)⊕x₃)⊕x₄)⊕...⊕x₈). Each intermediate computation depends on previous results.

**Parallel XOR (Non-Compositional)**: Compute independent XORs of adjacent pairs [x₁⊕x₂, x₃⊕x₄, x₅⊕x₆, x₇⊕x₈]. Each output is independent.

Same network architecture (3-layer MLP), optimizer (Adam, lr = 0.01), and training procedure (2000 epochs, 10 replications).

### 6.3 Results

**Table 4.** Study 4: Task Structure Results (Accuracy Dynamics)

| Condition | Best Model | R² | ΔAIC vs Linear |
|-----------|-----------|-----|----------------|
| Hierarchical (Compositional) | Logistic | 0.9885 | 1763.42 |
| Parallel (Non-Compositional) | Exponential | 0.9969 | 2296.80 |

Both conditions showed *e*-governed dynamics with excellent fits and overwhelming evidence.

### 6.4 Discussion

The hypothesis that compositional tasks specifically produce *e*-governed dynamics was not supported—non-compositional tasks showed identical (or stronger) *e*-governed signatures. Gradient descent appears to dominate over task structure.

---

## 7. Study 4b: Representational Similarity Analysis

### 7.1 Purpose

Determine whether compositional task structure produces compositional representations, and whether representational structure emergence follows *e*-governed dynamics.

### 7.2 Method

Following Kriegeskorte, Mur, and Bandettini (2008), we computed:

1. **Neural RDM**: Pairwise correlation distances between hidden layer activations
2. **Theoretical RDMs**:
   - *Hierarchical*: Inputs similar if they share intermediate parity computations (nested structure)
   - *Parallel*: Inputs similar if they produce same output vector (flat clustering)
3. **RSA**: Spearman correlation between neural and theoretical RDMs

We tracked RSA over training and fit dynamics models.

### 7.3 Results

**Table 5.** Study 4b: RSA Dynamics Results

| Condition | Best RSA Model | R² | Final RSA |
|-----------|---------------|-----|-----------|
| Hierarchical → Hierarchical RDM | Logistic | 0.9853 | **0.076** |
| Parallel → Parallel RDM | Exponential | 0.9894 | **0.320** |

Both RSA trajectories followed *e*-governed dynamics. However, the critical finding is the **final RSA values**:

- Hierarchical task: RSA = 0.076 (near zero correlation with compositional structure)
- Parallel task: RSA = 0.320 (moderate correlation with flat structure)

### 7.4 Discussion

Neural networks do **not** spontaneously develop compositional representations even when solving compositional tasks. The network achieved 100% accuracy on parity but its internal representations showed almost zero correlation with the hierarchical structure of the task (RSA ≈ 0.08).

This is consistent with Lake and Baroni (2018) and Fodor and Pylyshyn (1988): neural networks can solve compositional tasks without compositional representations, likely through memorization or non-hierarchical feature combinations.

---

## 8. Study 5: Genetic Programming

### 8.1 Purpose

Test whether genuinely compositional systems (symbolic programs) exhibit *e*-governed dynamics without gradient descent.

### 8.2 Rationale

Genetic programming (GP) evolves expression trees that are **compositional by definition**: subexpressions compose into expressions (Koza, 1992). Unlike neural networks, GP representations are provably compositional. Unlike gradient descent, GP uses evolutionary search (crossover + mutation + selection) with no log-likelihoods.

### 8.3 Method

**Primitives**: XOR, AND, OR, NOT, NAND (arity 1-2), input variables x₀-x₅.

**Parity Task** (Compositional): Evolve program computing 6-bit parity. Optimal solution requires deep composition.

**Parallel XOR Task** (Non-Compositional): Evolve separate programs for each of three independent 2-bit XORs.

Parameters: Population 200, 150 generations, tournament selection (size 7), crossover probability 0.8, mutation probability 0.2, max depth 8. 10 replications.

### 8.4 Results

**Table 6.** Study 5: Genetic Programming Results

| Condition | Measure | Best Model | R² |
|-----------|---------|-----------|-----|
| Parity (Compositional) | Fitness | **Power Law** | 0.9572 |
| Parity (Compositional) | Complexity | **Power Law** | 0.8664 |
| Parallel (Non-Compositional) | Fitness | **Power Law** | 0.9033 |
| Parallel (Non-Compositional) | Complexity | Linear | 0.0485 |

**Neither condition showed *e*-governed dynamics.**

Additional observations:
- Parity: Start fitness 0.59, end fitness 0.96; Start complexity 25.8 nodes, end complexity 39.5 nodes
- Parallel: Start fitness 0.75, end fitness 1.00; Start complexity 13.0 nodes, end complexity 11.4 nodes

### 8.5 Discussion

Genetic programming—a genuinely compositional system without gradient descent—produces **power law dynamics**, not *e*-governed dynamics. This finding has critical implications:

1. **The *e*-dynamics in Studies 2-4 came from gradient descent**, not compositionality
2. **Compositional representations ≠ compositional learning**: GP has compositional programs but random recombination-based search
3. **Power law emerges from averaging independent components** (Newell & Rosenbloom, 1981): crossover swaps random subtrees; selection keeps fit solutions

---

## 9. General Discussion

### 9.1 Summary of Findings

**Table 7.** Integrated Results Across All Studies

| Study | System | Learning | *e*-governed? | Key Finding |
|-------|--------|----------|---------------|-------------|
| 1 | Math simulation | — | Yes (β=1) | Math verified |
| 2 | Neural net | Gradient | Yes (both conditions) | Gradient dominates |
| 3 | Neural net | Hebbian | Self-ref: Yes (ΔAIC=57); Control: No (ΔAIC=3) | Differential signal |
| 4 | Neural net | Gradient | Yes (both conditions) | Task structure doesn't matter |
| 4b | Neural net | Gradient | Yes (RSA dynamics) | Representations not compositional |
| 5 | Genetic programming | Evolutionary | **No (power law)** | No gradient → no *e* |

### 9.2 The Role of Gradient Descent

The consistent pattern across studies is:

- **With gradient descent**: *e*-governed dynamics regardless of architecture (Study 2), task structure (Study 4), or representation (Study 4b)
- **Without gradient descent**: 
  - Hebbian: Differential signal, self-reference matters (Study 3)
  - Evolutionary: Power law dynamics (Study 5)

This suggests gradient descent imposes *e*-governed dynamics through its information-theoretic structure. Cross-entropy loss L = -Σy log(ŷ) inherently involves natural logarithms. The gradient ∂L/∂w propagates this logarithmic structure throughout learning.

### 9.3 Compositional Representations vs. Compositional Learning

Study 4b revealed that neural networks do not develop compositional representations (RSA ≈ 0.08) even when achieving perfect accuracy on compositional tasks. Study 5 showed that GP—which has compositional representations by construction—produces power law rather than *e*-governed dynamics.

This dissociation suggests the critical variable is not whether **representations** are compositional, but whether **learning** is compositional. In GP, crossover swaps random subtrees; evolution does not say "use this abstraction to build the next one." The search process is recombinatorial, not constructive.

### 9.4 Refined Hypothesis

Based on our findings, we propose refining the Danan (2025) hypothesis:

**Original hypothesis**: "Self-referential and compositional systems exhibit *e*-governed dynamics."

**Refined hypothesis**: "Systems with **compositional learning**—where new knowledge is constructed *from* existing knowledge—exhibit *e*-governed dynamics. Systems that merely have compositional representations, or that search randomly through compositional structures, do not necessarily exhibit *e*-governed dynamics."

The key distinction is:

| Property | Produces *e*-dynamics? | Example |
|----------|------------------------|---------|
| Compositional representations | No (Study 5) | GP trees |
| Compositional task structure | No (Study 4) | Parity |
| Gradient descent optimization | Yes (Studies 2, 4) | Neural networks |
| Compositional learning process | Hypothesized yes | Human learning? |

### 9.5 What Would Test the Refined Hypothesis?

To test whether compositional **learning** produces *e*-governed dynamics, we need systems where:

1. New knowledge is genuinely constructed from existing knowledge (not random recombination)
2. Learning does not involve log-likelihoods (to remove gradient descent confound)

Candidate systems include:
- Human concept learning (Piaget, 1954; Carey, 2009)
- Library learning systems like DreamCoder (Ellis et al., 2021)
- Hierarchical reinforcement learning with skill reuse
- Language acquisition data

### 9.6 Theoretical Implications

Our findings have implications for understanding the mathematical structure of learning:

1. ***e* as optimizer signature**: In artificial systems, *e*-governed dynamics may primarily reflect information-theoretic optimization rather than cognitive structure

2. **Power law as search signature**: Random recombination + selection produces power law dynamics (Newell & Rosenbloom, 1981), explaining GP results

3. **The compositionality gap**: Neural networks solve compositional tasks without compositional representations—the "algebraic blindness" noted by Marcus (2003) and the systematicity failures documented by Lake and Baroni (2018)

### 9.7 Limitations

1. **Hebbian learning weakness**: Study 3's differential signal emerged from weak learning curves, limiting statistical power

2. **Task domain**: All studies used Boolean classification. Generalization to other domains requires further investigation

3. **GP search characteristics**: GP's random recombination may not represent all compositional systems; library learning systems might show different dynamics

4. **No human data**: The refined hypothesis about compositional learning is best tested with human data, which we did not collect

### 9.8 Contributions

Despite not confirming the original hypothesis, our investigation makes several contributions:

1. **Identified the dominant source**: Gradient descent produces *e*-governed dynamics in artificial systems

2. **Ruled out spurious confirmations**: Task structure and architectural self-reference do not independently produce *e*-dynamics when gradient descent is present

3. **Documented the compositionality gap**: Neural networks don't develop compositional representations (RSA ≈ 0.08)

4. **Refined the theory**: Distinguished compositional representations from compositional learning

5. **Identified appropriate test systems**: Library learning and human cognition as targets for future investigation

---

## 10. Conclusion

We conducted five studies testing whether self-referential and compositional systems exhibit *e*-governed learning dynamics. Our findings indicate that *e*-governed dynamics in artificial systems primarily reflect gradient descent optimization, not compositionality or self-reference per se. Neural networks show *e*-governed dynamics regardless of whether they are self-referential (Study 2) or whether tasks are compositional (Study 4), and they fail to develop compositional representations even when solving compositional tasks (Study 4b, RSA = 0.08). When gradient descent is removed, genetic programming shows power law dynamics (Study 5, R² = 0.96), while Hebbian networks show a differential signal favoring self-reference (Study 3, ΔAIC = 57 vs. 3).

These findings refine the Danan (2025) hypothesis: *e*-governed dynamics may emerge specifically from **compositional learning**—where new knowledge is constructed from existing knowledge—rather than from compositional representations or tasks alone. Testing this refined hypothesis requires systems with genuinely compositional learning processes, such as human cognition or library learning systems, which remain targets for future investigation.

The broader implication is that the mathematical signature of *e* in learning systems carries information about **how** learning occurs, not just **what** is learned. In gradient-based systems, *e* reflects information-theoretic optimization. In evolutionary systems, its absence (replaced by power law) reflects random recombinatorial search. Identifying the learning processes that genuinely produce *e*-governed dynamics through compositional construction remains an open question at the intersection of cognitive science, machine learning, and mathematics.

---

## References

Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer.

Carey, S. (2009). *The Origin of Concepts*. Oxford University Press.

Danan, H. (2025). Recursive abstraction: When computation requires self-reference. *Working paper*.

Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., ... & Tenenbaum, J. B. (2021). DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. *Proceedings of the 42nd ACM SIGPLAN Conference on Programming Language Design and Implementation*, 835-850.

Fodor, J. A., & Pylyshyn, Z. W. (1988). Connectionism and cognitive architecture: A critical analysis. *Cognition*, 28(1-2), 3-71.

Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory*. Wiley.

Jaynes, E. T. (1957). Information theory and statistical mechanics. *Physical Review*, 106(4), 620-630.

Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis—connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *Proceedings of the 35th International Conference on Machine Learning*, 2873-2882.

Marcus, G. F. (2003). *The Algebraic Mind: Integrating Connectionism and Cognitive Science*. MIT Press.

Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice. In J. R. Anderson (Ed.), *Cognitive Skills and Their Acquisition* (pp. 1-55). Lawrence Erlbaum Associates.

Oja, E. (1982). A simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267-273.

Piaget, J. (1954). *The Construction of Reality in the Child*. Basic Books.

Poli, R., Langdon, W. B., & McPhee, N. F. (2008). *A Field Guide to Genetic Programming*. Lulu.com.

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering* (2nd ed.). Westview Press.

Tenenbaum, M., & Pollard, H. (1985). *Ordinary Differential Equations*. Dover Publications.

---

## Appendix A: Detailed Results

### A.1 Study 1 Complete Model Fits

**β = 0.0:**
- Linear: R² = 0.9906, AIC = -512.03
- Logistic: R² = 0.9907, AIC = -511.80, ΔAIC = 0.23
- Power Law: R² = 0.9906, AIC = -510.41, ΔAIC = 1.63
- Exponential: R² = 0.9884, AIC = -468.59, ΔAIC = 43.44

**β = 1.0:**
- Logistic: R² = 0.9990, AIC = 712.20
- Exponential: R² = 0.9299, AIC = 1569.95, ΔAIC = 857.76
- Power Law: R² = 0.9230, AIC = 1588.81, ΔAIC = 876.61
- Linear: R² = 0.9212, AIC = 1591.23, ΔAIC = 879.04

### A.2 Study 2 Complete Model Fits

**Self-Referential Condition:**
- Logistic: R² = 0.9946, AIC = -2910.09
- Exponential: R² = 0.9939, AIC = -2872.39, ΔAIC = 37.71
- Power Law: R² = 0.8549, AIC = -1922.95, ΔAIC = 987.15
- Linear: R² = 0.1180, AIC = -1383.60, ΔAIC = 1526.50

**Non-Self-Referential Condition:**
- Logistic: R² = 0.9947, AIC = -2914.88
- Exponential: R² = 0.9937, AIC = -2863.97, ΔAIC = 50.91
- Power Law: R² = 0.8542, AIC = -1920.61, ΔAIC = 994.27
- Linear: R² = 0.1183, AIC = -1382.69, ΔAIC = 1532.18

### A.3 Study 3 Complete Model Fits

**Self-Referential Hebbian:**
- Exponential: R² = 0.7647, AIC = -980.86
- Power Law: R² = 0.7219, AIC = -964.14, ΔAIC = 16.71
- Linear: R² = 0.5771, AIC = -924.21, ΔAIC = 56.65
- Logistic: R² = 0.0892, AIC = -845.49, ΔAIC = 135.37

**Non-Self-Referential Hebbian:**
- Exponential: R² = 0.4303, AIC = -922.68
- Linear: R² = 0.3994, AIC = -919.40, ΔAIC = 3.28
- Power Law: R² = 0.3921, AIC = -916.19, ΔAIC = 6.49
- Logistic: R² = -0.1408, AIC = -853.24, ΔAIC = 69.44

### A.4 Study 4 Complete Model Fits

**Hierarchical (Compositional) Accuracy:**
- Logistic: R² = 0.9885, AIC = -4221.48
- Exponential: R² = 0.9727, AIC = -3874.84, ΔAIC = 346.64
- Power Law: R² = 0.6976, AIC = -2912.19, ΔAIC = 1309.29
- Linear: R² = 0.0542, AIC = -2458.06, ΔAIC = 1763.42

**Parallel (Non-Compositional) Accuracy:**
- Exponential: R² = 0.9969, AIC = -5250.36
- Logistic: R² = 0.9786, AIC = -4481.46, ΔAIC = 768.90
- Power Law: R² = 0.9078, AIC = -3897.05, ΔAIC = 1353.31
- Linear: R² = 0.0198, AIC = -2953.55, ΔAIC = 2296.80

### A.5 Study 4b RSA Results

**Hierarchical RSA Trajectory:**
- Start: -0.0050, End: 0.0759, Range: 0.0808
- Logistic: R² = 0.9853, AIC = -1231.90
- Power Law: R² = 0.7683, AIC = -956.39, ΔAIC = 275.51
- Linear: R² = 0.2689, AIC = -843.47, ΔAIC = 388.43

**Parallel RSA Trajectory:**
- Start: 0.0244, End: 0.3199, Range: 0.2955
- Exponential: R² = 0.9894, AIC = -1102.26
- Logistic: R² = 0.9875, AIC = -1085.50, ΔAIC = 16.77
- Power Law: R² = 0.8879, AIC = -866.37, ΔAIC = 235.89
- Linear: R² = 0.1572, AIC = -666.66, ΔAIC = 435.60

### A.6 Study 5 GP Evolution Statistics

**Parity Task:**
- 6/10 runs achieved 100% fitness
- Mean generations to solution (when found): 41.5
- Final complexity range: 15-76 nodes

**Parallel Task:**
- 10/10 runs achieved 100% fitness (all outputs)
- All solved by generation ~50
- Final complexity range: 3.7-17.3 nodes (per output)

---

## Appendix B: Code Availability

Complete code for all studies is available at:
https://github.com/HillaryDanan/self-referential-dynamics

Repository structure:
```
self-referential-dynamics/
├── src/
│   ├── study1_mathematical.py
│   ├── study2_empirical.py
│   ├── study3_hebbian.py
│   ├── study4_compositional.py
│   ├── study4b_rsa.py
│   ├── study5_gp.py
│   └── utils.py
├── figures/
├── run_study[1-5].py
├── run_all.py
└── requirements.txt
```

---

*Corresponding author: Hillary Danan (contact information)*

*Acknowledgments: The author thanks Claude (Anthropic) for assistance with experimental implementation and statistical analysis.*

*Data availability: All data generated during this study are reproducible using the provided code with specified random seeds.*