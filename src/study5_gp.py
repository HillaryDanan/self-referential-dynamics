"""
Study 5: Compositional Program Synthesis via Genetic Programming
=================================================================

PURPOSE: Test whether GENUINELY COMPOSITIONAL systems exhibit e-governed
         dynamics when building abstractions.

THEORETICAL MOTIVATION:
  Studies 2-4b found that gradient descent produces e-governed dynamics
  regardless of task structure or architecture. Neural networks also
  don't spontaneously develop compositional representations (Study 4b).
  
  The Danan (2025) hypothesis is about INTELLIGENCE and ABSTRACTION,
  not about neural networks specifically. We need to test systems that
  actually BUILD compositional representations.

WHY GENETIC PROGRAMMING:
  1. Programs ARE compositional by definition (subexpressions compose)
  2. NO gradient descent (removes optimizer confound)
  3. Evolution can discover hierarchical structure
  4. Program complexity = abstraction complexity

HYPOTHESIS:
  If compositionality produces e-governed dynamics (dA/dt ∝ A),
  then program complexity growth should follow e-governed curves
  when solving compositional tasks.

DESIGN:
  Condition A: Hierarchical task (8-bit parity)
    - Requires deep composition: ((((x1⊕x2)⊕x3)⊕x4)...)
    - Optimal solution has compositional structure
    
  Condition B: Parallel task (independent XORs)
    - Can be solved with shallow composition
    - Optimal solution is flat
    
  Measure:
    - Best fitness (accuracy) over generations
    - Best program complexity (tree depth, node count) over generations
    - Fit e-governed vs power law to both curves

REFERENCES:
  - Koza (1992). Genetic Programming. MIT Press.
  - Poli, Langdon, & McPhee (2008). A Field Guide to Genetic Programming.
  - Lake & Baroni (2018). Generalization without systematicity. ICML.

STATUS: EMPIRICAL HYPOTHESIS TEST (genuinely compositional system)
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from scipy.optimize import curve_fit
import os
from collections import defaultdict


# =============================================================================
# MODEL FITTING (same as previous studies)
# =============================================================================

@dataclass
class FitResult:
    model_name: str
    params: Dict
    predictions: np.ndarray
    sse: float
    n_params: int
    aic: float
    bic: float
    r_squared: float


def compute_aic_bic(sse: float, n: int, k: int) -> Tuple[float, float]:
    if sse <= 0:
        return float('inf'), float('inf')
    aic = n * np.log(sse / n) + 2 * k
    bic = n * np.log(sse / n) + k * np.log(n)
    return aic, bic


def exponential_approach(t, y_max, k, y0):
    return y_max - (y_max - y0) * np.exp(-k * t)

def logistic_model(t, K, k, t0):
    return K / (1 + np.exp(-k * (t - t0)))

def power_law_learning(t, a, b, c):
    return c - a / np.power(t + 1, b)

def linear_model(t, y0, slope):
    return y0 + slope * t


def fit_curves(t: np.ndarray, y: np.ndarray, 
               verbose: bool = False) -> Dict[str, FitResult]:
    """Fit candidate models to trajectory."""
    results = {}
    n = len(t)
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    y_mean = y.mean()
    t_max = t.max()
    
    ss_total = np.sum((y - y_mean)**2)
    
    if verbose:
        print(f"  Data range: [{y_min:.4f}, {y_max:.4f}], range={y_range:.4f}")
    
    if y_range < 0.01 or ss_total < 1e-10:
        if verbose:
            print("  WARNING: Range too small")
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
    
    # 1. Exponential
    try:
        p0 = [y_max * 1.05, 0.02, y_min]
        bounds = ([y_max * 0.9, 1e-6, y_min - abs(y_min)*0.5 - 0.1], 
                  [y_max * 1.5 + 0.1, 1.0, y_min * 0.5 + 0.1])
        
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if valid:
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
        p0 = [y_max * 1.05, 0.05, t_max / 3]
        bounds = ([y_max * 0.9, 1e-6, 0], [y_max * 1.5 + 0.1, 1.0, t_max * 2])
        
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if valid:
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
        p0 = [y_range, 0.5, y_max * 1.05]
        bounds = ([0.001, 0.01, y_max * 0.9], [y_range * 10 + 1, 3.0, y_max * 1.5 + 0.1])
        
        valid = all(bounds[0][i] < bounds[1][i] for i in range(3))
        if valid:
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
    
    # 4. Linear
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


def print_model_comparison(results: Dict[str, FitResult], title: str):
    if not results:
        print(f"No models fit successfully for {title}")
        return
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].aic)
    min_aic = sorted_results[0][1].aic
    
    print(f"{'Model':<35} {'R²':<10} {'AIC':<12} {'ΔAIC':<10}")
    print("-" * 60)
    
    for name, fit in sorted_results:
        delta_aic = fit.aic - min_aic
        marker = "← BEST" if delta_aic == 0 else ""
        print(f"{fit.model_name:<35} {fit.r_squared:<10.4f} {fit.aic:<12.2f} {delta_aic:<10.2f} {marker}")
    
    print("=" * 60)


# =============================================================================
# GENETIC PROGRAMMING: TREE REPRESENTATION
# =============================================================================

class GPNode:
    """Node in a GP expression tree."""
    def __init__(self, value, children=None):
        self.value = value  # Function name or terminal
        self.children = children or []
    
    def is_terminal(self):
        return len(self.children) == 0
    
    def depth(self):
        if self.is_terminal():
            return 1
        return 1 + max(c.depth() for c in self.children)
    
    def size(self):
        if self.is_terminal():
            return 1
        return 1 + sum(c.size() for c in self.children)
    
    def copy(self):
        if self.is_terminal():
            return GPNode(self.value)
        return GPNode(self.value, [c.copy() for c in self.children])
    
    def __str__(self):
        if self.is_terminal():
            return str(self.value)
        args = ", ".join(str(c) for c in self.children)
        return f"{self.value}({args})"


class GPPrimitiveSet:
    """Defines the functions and terminals for GP."""
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        
        # Functions: (name, arity, function)
        self.functions = {
            'XOR': (2, lambda a, b: a ^ b),
            'AND': (2, lambda a, b: a & b),
            'OR': (2, lambda a, b: a | b),
            'NOT': (1, lambda a: 1 - a),
            'NAND': (2, lambda a, b: 1 - (a & b)),
        }
        
        # Terminals: input variables
        self.terminals = [f'x{i}' for i in range(n_inputs)]
    
    def random_terminal(self):
        return GPNode(random.choice(self.terminals))
    
    def random_function(self):
        name = random.choice(list(self.functions.keys()))
        return name, self.functions[name][0]
    
    def evaluate(self, node: GPNode, inputs: Dict[str, int]) -> int:
        """Evaluate the expression tree on given inputs."""
        if node.is_terminal():
            return inputs[node.value]
        
        fname = node.value
        arity, func = self.functions[fname]
        
        args = [self.evaluate(c, inputs) for c in node.children]
        return func(*args)


def generate_random_tree(pset: GPPrimitiveSet, max_depth: int, 
                         method: str = 'grow') -> GPNode:
    """Generate a random expression tree."""
    if max_depth <= 1:
        return pset.random_terminal()
    
    if method == 'grow':
        # Grow: randomly choose function or terminal
        if random.random() < 0.5:
            return pset.random_terminal()
    
    # Full: always choose function until max_depth-1
    fname, arity = pset.random_function()
    children = [generate_random_tree(pset, max_depth - 1, method) 
                for _ in range(arity)]
    return GPNode(fname, children)


def get_all_nodes(node: GPNode) -> List[Tuple[GPNode, GPNode, int]]:
    """Get all nodes with their parent and child index."""
    nodes = [(node, None, -1)]
    
    def collect(n, parent, idx):
        for i, c in enumerate(n.children):
            nodes.append((c, n, i))
            collect(c, n, i)
    
    collect(node, None, -1)
    return nodes


# =============================================================================
# GENETIC OPERATORS
# =============================================================================

def crossover(parent1: GPNode, parent2: GPNode, pset: GPPrimitiveSet) -> Tuple[GPNode, GPNode]:
    """Subtree crossover."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    nodes1 = get_all_nodes(child1)
    nodes2 = get_all_nodes(child2)
    
    if len(nodes1) < 2 or len(nodes2) < 2:
        return child1, child2
    
    # Select crossover points (not root)
    idx1 = random.randint(1, len(nodes1) - 1)
    idx2 = random.randint(1, len(nodes2) - 1)
    
    node1, parent1_node, cidx1 = nodes1[idx1]
    node2, parent2_node, cidx2 = nodes2[idx2]
    
    # Swap subtrees
    if parent1_node and parent2_node:
        parent1_node.children[cidx1] = node2.copy()
        parent2_node.children[cidx2] = node1.copy()
    
    return child1, child2


def mutate(individual: GPNode, pset: GPPrimitiveSet, max_depth: int = 4) -> GPNode:
    """Subtree mutation."""
    mutant = individual.copy()
    nodes = get_all_nodes(mutant)
    
    if len(nodes) < 1:
        return mutant
    
    # Select mutation point
    idx = random.randint(0, len(nodes) - 1)
    node, parent, cidx = nodes[idx]
    
    # Generate new subtree
    new_subtree = generate_random_tree(pset, max_depth, 'grow')
    
    if parent is None:
        return new_subtree
    else:
        parent.children[cidx] = new_subtree
    
    return mutant


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

def evaluate_fitness(individual: GPNode, pset: GPPrimitiveSet,
                     X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate fitness as accuracy."""
    correct = 0
    n_samples = len(X)
    
    for i in range(n_samples):
        inputs = {f'x{j}': int(X[i, j]) for j in range(X.shape[1])}
        try:
            pred = pset.evaluate(individual, inputs)
            if pred == y[i]:
                correct += 1
        except:
            pass  # Invalid program
    
    return correct / n_samples


def evaluate_fitness_multi_output(individual: GPNode, pset: GPPrimitiveSet,
                                  X: np.ndarray, y: np.ndarray, 
                                  output_idx: int) -> float:
    """Evaluate fitness for one output of multi-output task."""
    correct = 0
    n_samples = len(X)
    
    for i in range(n_samples):
        inputs = {f'x{j}': int(X[i, j]) for j in range(X.shape[1])}
        try:
            pred = pset.evaluate(individual, inputs)
            if pred == y[i, output_idx]:
                correct += 1
        except:
            pass
    
    return correct / n_samples


# =============================================================================
# EVOLUTIONARY ALGORITHM
# =============================================================================

def run_gp(pset: GPPrimitiveSet, X: np.ndarray, y: np.ndarray,
           pop_size: int = 100, n_generations: int = 100,
           tournament_size: int = 7, crossover_prob: float = 0.8,
           mutation_prob: float = 0.2, max_depth: int = 8,
           verbose: bool = False, multi_output: bool = False,
           output_idx: int = 0) -> Dict:
    """
    Run genetic programming evolution.
    
    Returns history of best fitness and complexity.
    """
    # Initialize population
    population = []
    for _ in range(pop_size):
        method = 'grow' if random.random() < 0.5 else 'full'
        ind = generate_random_tree(pset, max_depth=5, method=method)
        population.append(ind)
    
    # Evaluate fitness function
    if multi_output:
        eval_func = lambda ind: evaluate_fitness_multi_output(ind, pset, X, y, output_idx)
    else:
        eval_func = lambda ind: evaluate_fitness(ind, pset, X, y)
    
    # Track history
    best_fitness_history = []
    best_complexity_history = []
    avg_fitness_history = []
    avg_complexity_history = []
    
    for gen in range(n_generations):
        # Evaluate all individuals
        fitnesses = [eval_func(ind) for ind in population]
        complexities = [ind.size() for ind in population]
        
        best_idx = np.argmax(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_complexity = complexities[best_idx]
        
        best_fitness_history.append(best_fitness)
        best_complexity_history.append(best_complexity)
        avg_fitness_history.append(np.mean(fitnesses))
        avg_complexity_history.append(np.mean(complexities))
        
        if verbose and gen % 10 == 0:
            print(f"    Gen {gen}: best_fit={best_fitness:.4f}, best_size={best_complexity}, avg_fit={np.mean(fitnesses):.4f}")
        
        # Early stopping
        if best_fitness >= 1.0:
            # Fill remaining history
            for _ in range(gen + 1, n_generations):
                best_fitness_history.append(1.0)
                best_complexity_history.append(best_complexity)
                avg_fitness_history.append(np.mean(fitnesses))
                avg_complexity_history.append(np.mean(complexities))
            break
        
        # Selection and reproduction
        new_population = []
        
        # Elitism: keep best
        new_population.append(population[best_idx].copy())
        
        while len(new_population) < pop_size:
            # Tournament selection
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            parent1 = max(tournament, key=lambda x: x[1])[0]
            
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            parent2 = max(tournament, key=lambda x: x[1])[0]
            
            # Crossover
            if random.random() < crossover_prob:
                child1, child2 = crossover(parent1, parent2, pset)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < mutation_prob:
                child1 = mutate(child1, pset, max_depth=4)
            if random.random() < mutation_prob:
                child2 = mutate(child2, pset, max_depth=4)
            
            # Depth limit
            if child1.depth() <= max_depth:
                new_population.append(child1)
            if len(new_population) < pop_size and child2.depth() <= max_depth:
                new_population.append(child2)
        
        population = new_population[:pop_size]
    
    return {
        'best_fitness': np.array(best_fitness_history),
        'best_complexity': np.array(best_complexity_history),
        'avg_fitness': np.array(avg_fitness_history),
        'avg_complexity': np.array(avg_complexity_history),
        'generations': np.arange(len(best_fitness_history))
    }


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_parity_data(n_bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Generate parity (hierarchical XOR) data."""
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)], dtype=np.int32)
    y = (X.sum(axis=1) % 2).astype(np.int32)
    return X, y


def generate_parallel_xor_data(n_bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Generate parallel XOR data."""
    n_samples = 2 ** n_bits
    X = np.array([[int(b) for b in format(i, f'0{n_bits}b')] 
                  for i in range(n_samples)], dtype=np.int32)
    
    n_outputs = n_bits // 2
    y = np.zeros((n_samples, n_outputs), dtype=np.int32)
    for i in range(n_outputs):
        y[:, i] = (X[:, 2*i] + X[:, 2*i + 1]) % 2
    
    return X, y


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_study5(
    n_runs: int = 10,
    n_bits: int = 6,  # Smaller for GP tractability
    n_generations: int = 150,
    pop_size: int = 200,
    save_dir: str = "figures"
) -> Dict:
    """
    Run Study 5: Genetic Programming comparison.
    
    Tests whether compositional program synthesis shows different
    learning dynamics than non-compositional tasks.
    """
    
    print("=" * 70)
    print("STUDY 5: GENETIC PROGRAMMING (COMPOSITIONAL PROGRAM SYNTHESIS)")
    print("=" * 70)
    print("""
THEORETICAL MOTIVATION:
  Neural networks don't spontaneously develop compositional representations
  (Study 4b). But the Danan (2025) hypothesis is about INTELLIGENCE and
  ABSTRACTION, not neural networks specifically.
  
  Genetic Programming (GP) evolves SYMBOLIC PROGRAMS, which are
  compositional by definition: subexpressions compose into expressions.

WHY GP:
  1. Programs ARE compositional (tree structure)
  2. NO gradient descent (removes optimizer confound)
  3. Evolution discovers structure, not just parameters
  4. Program complexity = abstraction complexity

HYPOTHESIS:
  Compositional task (parity) requires building deep compositional
  programs. This compositional structure building should follow
  e-governed dynamics (dC/dt ∝ C) if the theory is correct.

DESIGN:
  Condition A: 6-bit Parity (hierarchical/compositional)
    - Requires deep XOR composition
    - Optimal: ((((x0⊕x1)⊕x2)⊕x3)⊕x4)⊕x5
    
  Condition B: 3 independent 2-bit XORs (parallel/non-compositional)
    - Each output is x_{2i} ⊕ x_{2i+1}
    - Optimal: Three shallow trees

  Measure:
    - Best fitness (accuracy) over generations
    - Best program complexity (tree size) over generations
    
  Prediction:
    - Parity: e-governed complexity growth (compositions build on compositions)
    - Parallel: Different dynamics (independent subtasks)
""")
    print("-" * 70)
    
    # Generate data
    print(f"\nGenerating data: {n_bits} bits")
    X_parity, y_parity = generate_parity_data(n_bits)
    X_parallel, y_parallel = generate_parallel_xor_data(n_bits)
    
    print(f"Parity (compositional): {X_parity.shape[0]} samples, 1 output")
    print(f"Parallel (non-compositional): {X_parallel.shape[0]} samples, {y_parallel.shape[1]} outputs")
    
    pset = GPPrimitiveSet(n_bits)
    
    all_parity = []
    all_parallel = []
    
    print(f"\nRunning {n_runs} replications ({n_generations} generations each)...")
    print(f"Population size: {pop_size}")
    print("\n*** NOTE: GP is stochastic; results vary across runs ***\n")
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"--- Replication {run+1}/{n_runs} ---")
        print(f"{'='*50}")
        
        random.seed(run * 100 + 42)
        np.random.seed(run * 100 + 42)
        
        print("\n  Training on PARITY (compositional)...")
        parity_result = run_gp(
            pset, X_parity, y_parity,
            pop_size=pop_size,
            n_generations=n_generations,
            verbose=True
        )
        
        print("\n  Training on PARALLEL XORs (non-compositional)...")
        # For parallel, evolve separate programs for each output and average
        parallel_results = []
        n_outputs = y_parallel.shape[1]
        
        for out_idx in range(n_outputs):
            print(f"    Output {out_idx+1}/{n_outputs}...")
            result = run_gp(
                pset, X_parallel, y_parallel,
                pop_size=pop_size,
                n_generations=n_generations,
                verbose=False,
                multi_output=True,
                output_idx=out_idx
            )
            parallel_results.append(result)
        
        # Average across outputs
        parallel_combined = {
            'best_fitness': np.mean([r['best_fitness'] for r in parallel_results], axis=0),
            'best_complexity': np.mean([r['best_complexity'] for r in parallel_results], axis=0),
            'avg_fitness': np.mean([r['avg_fitness'] for r in parallel_results], axis=0),
            'avg_complexity': np.mean([r['avg_complexity'] for r in parallel_results], axis=0),
            'generations': parallel_results[0]['generations']
        }
        
        all_parity.append(parity_result)
        all_parallel.append(parallel_combined)
        
        print(f"\n  Parity final: fitness={parity_result['best_fitness'][-1]:.4f}, complexity={parity_result['best_complexity'][-1]}")
        print(f"  Parallel final: fitness={parallel_combined['best_fitness'][-1]:.4f}, complexity={parallel_combined['best_complexity'][-1]:.1f}")
    
    # Aggregate
    generations = all_parity[0]['generations']
    
    parity_fitness_mean = np.mean([r['best_fitness'] for r in all_parity], axis=0)
    parity_fitness_std = np.std([r['best_fitness'] for r in all_parity], axis=0)
    parity_complexity_mean = np.mean([r['best_complexity'] for r in all_parity], axis=0)
    parity_complexity_std = np.std([r['best_complexity'] for r in all_parity], axis=0)
    
    parallel_fitness_mean = np.mean([r['best_fitness'] for r in all_parallel], axis=0)
    parallel_fitness_std = np.std([r['best_fitness'] for r in all_parallel], axis=0)
    parallel_complexity_mean = np.mean([r['best_complexity'] for r in all_parallel], axis=0)
    parallel_complexity_std = np.std([r['best_complexity'] for r in all_parallel], axis=0)
    
    # Diagnostics
    print("\n" + "=" * 60)
    print("EVOLUTION DIAGNOSTICS")
    print("=" * 60)
    print(f"\nParity (Compositional):")
    print(f"  Start fitness: {parity_fitness_mean[0]:.4f}")
    print(f"  End fitness: {parity_fitness_mean[-1]:.4f}")
    print(f"  Start complexity: {parity_complexity_mean[0]:.1f}")
    print(f"  End complexity: {parity_complexity_mean[-1]:.1f}")
    
    print(f"\nParallel (Non-Compositional):")
    print(f"  Start fitness: {parallel_fitness_mean[0]:.4f}")
    print(f"  End fitness: {parallel_fitness_mean[-1]:.4f}")
    print(f"  Start complexity: {parallel_complexity_mean[0]:.1f}")
    print(f"  End complexity: {parallel_complexity_mean[-1]:.1f}")
    
    # Fit models to FITNESS curves
    print("\n" + "=" * 60)
    print("CONDITION A: PARITY (COMPOSITIONAL) - FITNESS DYNAMICS")
    print("=" * 60)
    
    parity_fitness_fits = fit_curves(generations, parity_fitness_mean, verbose=True)
    print_model_comparison(parity_fitness_fits, "Parity Fitness Dynamics")
    
    print("\n" + "=" * 60)
    print("CONDITION B: PARALLEL (NON-COMPOSITIONAL) - FITNESS DYNAMICS")
    print("=" * 60)
    
    parallel_fitness_fits = fit_curves(generations, parallel_fitness_mean, verbose=True)
    print_model_comparison(parallel_fitness_fits, "Parallel Fitness Dynamics")
    
    # Fit models to COMPLEXITY curves
    print("\n" + "=" * 60)
    print("CONDITION A: PARITY (COMPOSITIONAL) - COMPLEXITY DYNAMICS")
    print("=" * 60)
    
    parity_complexity_fits = fit_curves(generations, parity_complexity_mean, verbose=True)
    print_model_comparison(parity_complexity_fits, "Parity Complexity Dynamics")
    
    print("\n" + "=" * 60)
    print("CONDITION B: PARALLEL (NON-COMPOSITIONAL) - COMPLEXITY DYNAMICS")
    print("=" * 60)
    
    parallel_complexity_fits = fit_curves(generations, parallel_complexity_mean, verbose=True)
    print_model_comparison(parallel_complexity_fits, "Parallel Complexity Dynamics")
    
    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    colors = {
        'exponential': '#e41a1c',
        'logistic': '#377eb8',
        'power_law': '#4daf4a',
        'linear': '#984ea3'
    }
    
    # Row 1: Parity (Compositional)
    ax = axes[0, 0]
    ax.fill_between(generations, parity_fitness_mean - parity_fitness_std,
                    parity_fitness_mean + parity_fitness_std, alpha=0.3, color='steelblue')
    ax.plot(generations, parity_fitness_mean, 'o-', color='steelblue', markersize=2, label='Data')
    for name, fit in parity_fitness_fits.items():
        ax.plot(generations, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Accuracy)')
    ax.set_title('A: Parity (Compositional)\nFitness Dynamics')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.fill_between(generations, parity_complexity_mean - parity_complexity_std,
                    parity_complexity_mean + parity_complexity_std, alpha=0.3, color='steelblue')
    ax.plot(generations, parity_complexity_mean, 'o-', color='steelblue', markersize=2, label='Data')
    for name, fit in parity_complexity_fits.items():
        ax.plot(generations, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Program Size (nodes)')
    ax.set_title('A: Parity (Compositional)\nComplexity Dynamics')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # AIC comparisons
    ax = axes[0, 2]
    if parity_fitness_fits:
        names = list(parity_fitness_fits.keys())
        aics = [parity_fitness_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['steelblue'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Fitness Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 3]
    if parity_complexity_fits:
        names = list(parity_complexity_fits.keys())
        aics = [parity_complexity_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['steelblue'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Complexity Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Row 2: Parallel (Non-Compositional)
    ax = axes[1, 0]
    ax.fill_between(generations, parallel_fitness_mean - parallel_fitness_std,
                    parallel_fitness_mean + parallel_fitness_std, alpha=0.3, color='coral')
    ax.plot(generations, parallel_fitness_mean, 'o-', color='coral', markersize=2, label='Data')
    for name, fit in parallel_fitness_fits.items():
        ax.plot(generations, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (Accuracy)')
    ax.set_title('B: Parallel XORs (Non-Compositional)\nFitness Dynamics')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.fill_between(generations, parallel_complexity_mean - parallel_complexity_std,
                    parallel_complexity_mean + parallel_complexity_std, alpha=0.3, color='coral')
    ax.plot(generations, parallel_complexity_mean, 'o-', color='coral', markersize=2, label='Data')
    for name, fit in parallel_complexity_fits.items():
        ax.plot(generations, fit.predictions, color=colors.get(name, 'gray'), linewidth=2,
                label=f"{fit.model_name.split()[0]} (R²={fit.r_squared:.3f})", alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Program Size (nodes)')
    ax.set_title('B: Parallel XORs (Non-Compositional)\nComplexity Dynamics')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 2]
    if parallel_fitness_fits:
        names = list(parallel_fitness_fits.keys())
        aics = [parallel_fitness_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['coral'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Fitness Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 3]
    if parallel_complexity_fits:
        names = list(parallel_complexity_fits.keys())
        aics = [parallel_complexity_fits[n].aic for n in names]
        min_aic = min(aics)
        delta_aics = [a - min_aic for a in aics]
        bar_colors = ['coral'] * len(names)
        best_idx = np.argmin(delta_aics)
        bar_colors[best_idx] = 'green'
        ax.bar(range(len(names)), delta_aics, color=bar_colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=8)
    ax.set_ylabel('ΔAIC (lower = better)')
    ax.set_title('Complexity Model Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Study 5: Genetic Programming (Compositional Program Synthesis)\n'
                 'Do genuinely compositional systems show e-governed dynamics?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/study5_gp_results.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/study5_gp_results.pdf', bbox_inches='tight')
    
    # Summary
    print("\n" + "=" * 70)
    print("STUDY 5 SUMMARY")
    print("=" * 70)
    
    def get_best(fits):
        if fits:
            return min(fits.keys(), key=lambda x: fits[x].aic)
        return None
    
    def is_e(model_name):
        return model_name in ['exponential', 'logistic']
    
    best_parity_fit = get_best(parity_fitness_fits)
    best_parallel_fit = get_best(parallel_fitness_fits)
    best_parity_comp = get_best(parity_complexity_fits)
    best_parallel_comp = get_best(parallel_complexity_fits)
    
    print(f"""
KEY QUESTION:
  Do GENUINELY COMPOSITIONAL systems (symbolic programs) show
  e-governed dynamics when building compositional solutions?

RESULTS - FITNESS DYNAMICS:
  Parity (Compositional):
    Best model: {parity_fitness_fits[best_parity_fit].model_name if best_parity_fit else 'N/A'}
    R²: {parity_fitness_fits[best_parity_fit].r_squared:.4f if best_parity_fit else 0}
    e-governed: {'YES' if best_parity_fit and is_e(best_parity_fit) else 'NO'}
    
  Parallel (Non-Compositional):
    Best model: {parallel_fitness_fits[best_parallel_fit].model_name if best_parallel_fit else 'N/A'}
    R²: {parallel_fitness_fits[best_parallel_fit].r_squared:.4f if best_parallel_fit else 0}
    e-governed: {'YES' if best_parallel_fit and is_e(best_parallel_fit) else 'NO'}

RESULTS - COMPLEXITY DYNAMICS:
  Parity (Compositional):
    Best model: {parity_complexity_fits[best_parity_comp].model_name if best_parity_comp else 'N/A'}
    R²: {parity_complexity_fits[best_parity_comp].r_squared:.4f if best_parity_comp else 0}
    e-governed: {'YES' if best_parity_comp and is_e(best_parity_comp) else 'NO'}
    
  Parallel (Non-Compositional):
    Best model: {parallel_complexity_fits[best_parallel_comp].model_name if best_parallel_comp else 'N/A'}
    R²: {parallel_complexity_fits[best_parallel_comp].r_squared:.4f if best_parallel_comp else 0}
    e-governed: {'YES' if best_parallel_comp and is_e(best_parallel_comp) else 'NO'}
""")
    
    # Interpretation
    parity_e_fit = is_e(best_parity_fit) if best_parity_fit else False
    parallel_e_fit = is_e(best_parallel_fit) if best_parallel_fit else False
    parity_e_comp = is_e(best_parity_comp) if best_parity_comp else False
    parallel_e_comp = is_e(best_parallel_comp) if best_parallel_comp else False
    
    if parity_e_fit and not parallel_e_fit:
        print("=" * 60)
        print("INTERPRETATION: FITNESS DYNAMICS SUPPORT HYPOTHESIS")
        print("=" * 60)
        print("""
  Compositional task (parity) shows e-governed fitness dynamics.
  Non-compositional task (parallel) shows different dynamics.
  
  In a gradient-free, genuinely compositional system (GP),
  compositional tasks produce e-governed learning.
""")
    
    if parity_e_comp and not parallel_e_comp:
        print("=" * 60)
        print("INTERPRETATION: COMPLEXITY DYNAMICS SUPPORT HYPOTHESIS")
        print("=" * 60)
        print("""
  Compositional task (parity) shows e-governed complexity growth.
  Non-compositional task (parallel) shows different growth.
  
  This directly supports the theory: when abstractions genuinely
  build on abstractions, complexity grows following dC/dt ∝ C → e.
""")
    
    if (parity_e_fit and parallel_e_fit) or (parity_e_comp and parallel_e_comp):
        print("=" * 60)
        print("INTERPRETATION: BOTH CONDITIONS SHOW SIMILAR DYNAMICS")
        print("=" * 60)
        print("""
  Both compositional and non-compositional tasks show similar dynamics.
  
  Possible explanations:
  1. Evolutionary dynamics dominate over task structure
  2. The distinction requires different experimental design
  3. e-governed dynamics may be universal in adaptive systems
""")
    
    print(f"\nFigures saved to {save_dir}/")
    
    return {
        'parity_fitness_fits': parity_fitness_fits,
        'parallel_fitness_fits': parallel_fitness_fits,
        'parity_complexity_fits': parity_complexity_fits,
        'parallel_complexity_fits': parallel_complexity_fits,
        'parity_fitness_mean': parity_fitness_mean,
        'parallel_fitness_mean': parallel_fitness_mean,
        'parity_complexity_mean': parity_complexity_mean,
        'parallel_complexity_mean': parallel_complexity_mean,
        'generations': generations,
        'all_parity': all_parity,
        'all_parallel': all_parallel
    }


if __name__ == "__main__":
    results = run_study5(
        n_runs=10,
        n_bits=6,
        n_generations=150,
        pop_size=200,
        save_dir="figures"
    )