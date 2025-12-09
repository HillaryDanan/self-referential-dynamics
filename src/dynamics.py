"""
Dynamical systems simulation for Study 1 (mathematical verification).

This module verifies the MATHEMATICAL claim that:
- State-dependent growth dA/dt = kA^β produces e-governed dynamics when β → 1
- This is ESTABLISHED MATHEMATICS (Strogatz, 2015)

We are NOT testing an empirical hypothesis here - we are verifying
that our framework correctly implements the known mathematics.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional


def compositional_growth_ode(A: float, t: float, k: float, beta: float, 
                              K: Optional[float] = None) -> float:
    """
    ODE for compositional abstraction growth.
    
    dA/dt = k * A^β * (1 - A/K) if K is specified (bounded)
    dA/dt = k * A^β otherwise (unbounded)
    
    Parameters:
    -----------
    A : float
        Current abstraction count
    t : float
        Time (not used in autonomous ODE, but required by odeint)
    k : float
        Growth rate constant
    beta : float
        Compositionality exponent
        β = 0: constant rate (linear growth)
        β = 1: linear compositionality (exponential growth)
        β > 1: superlinear (finite-time blowup - unrealistic)
    K : float or None
        Carrying capacity (for logistic saturation)
    
    Returns:
    --------
    dA/dt : float
    
    Mathematical basis (Strogatz, 2015):
    - β = 1, K = None: A(t) = A₀ * e^(kt) [pure exponential]
    - β = 1, K finite: logistic growth [e in denominator]
    - β = 0: A(t) = A₀ + kt [linear]
    """
    if A <= 0:
        return 0.0
    
    growth = k * (A ** beta)
    
    if K is not None and K > 0:
        growth *= (1 - A / K)
    
    return max(growth, 0.0)


def simulate_compositional_growth(
    A0: float = 10.0,
    k: float = 0.1,
    beta: float = 1.0,
    K: Optional[float] = None,
    T: float = 100.0,
    dt: float = 0.5,
    noise_std: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate compositional abstraction growth.
    
    Parameters:
    -----------
    A0 : float
        Initial abstraction count
    k : float
        Growth rate constant
    beta : float
        Compositionality exponent (see compositional_growth_ode)
    K : float or None
        Carrying capacity
    T : float
        Total simulation time
    dt : float
        Time step for output
    noise_std : float
        Process noise (for stochastic version)
    seed : int
        Random seed
    
    Returns:
    --------
    t : np.ndarray
        Time points
    A : np.ndarray
        Abstraction counts
    """
    np.random.seed(seed)
    
    t = np.arange(0, T, dt)
    A = odeint(compositional_growth_ode, A0, t, args=(k, beta, K)).flatten()
    
    # Add measurement noise if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std * np.mean(A), size=A.shape)
        A = np.maximum(A + noise, 1.0)
    
    return t, A


def analytical_solutions(t: np.ndarray, A0: float, k: float, beta: float,
                         K: Optional[float] = None) -> np.ndarray:
    """
    Analytical solutions for special cases (for verification).
    
    β = 0: A(t) = A₀ + kt [linear]
    β = 1, K = None: A(t) = A₀ * e^(kt) [exponential]
    β = 1, K finite: A(t) = K / (1 + ((K-A₀)/A₀) * e^(-kt)) [logistic]
    
    Reference: Tenenbaum & Pollard (1985)
    """
    if np.abs(beta) < 1e-10:
        # Linear case
        return A0 + k * t
    
    elif np.abs(beta - 1.0) < 1e-10:
        if K is None:
            # Pure exponential
            return A0 * np.exp(k * t)
        else:
            # Logistic
            return K / (1 + ((K - A0) / A0) * np.exp(-k * t))
    
    else:
        # General case: numerical solution required
        return None