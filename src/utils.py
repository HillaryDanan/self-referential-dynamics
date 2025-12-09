"""
Utility functions for e-dynamics experiments.
Statistical methods following established literature.

References:
- Burnham & Anderson (2002). Model Selection and Multimodel Inference.
- Heathcote, Brown, & Mewhort (2000). Psychonomic Bulletin & Review.
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def exponential_growth(t: np.ndarray, y0: float, k: float) -> np.ndarray:
    """
    Exponential growth: y(t) = y0 * e^(kt)
    Solution to dy/dt = ky (pure e-governed, unbounded)
    """
    return y0 * np.exp(k * t)


def exponential_approach(t: np.ndarray, y_max: float, k: float, y0: float) -> np.ndarray:
    """
    Exponential approach to asymptote: y(t) = y_max - (y_max - y0) * e^(-kt)
    Solution to dy/dt = k(y_max - y) (e-governed, bounded)
    Time constant τ = 1/k. At t = τ: reaches 63.2% of improvement.
    """
    return y_max - (y_max - y0) * np.exp(-k * t)


def logistic_model(t: np.ndarray, K: float, k: float, t0: float) -> np.ndarray:
    """
    Logistic growth: y(t) = K / (1 + e^(-k(t - t0)))
    Still e-governed (exponential in denominator).
    Reference: Verhulst (1838)
    """
    return K / (1 + np.exp(-k * (t - t0)))


def power_law(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Power law: y(t) = a * (t + c)^b
    NOT e-governed. No characteristic time constant. Scale-free.
    Reference: Newell & Rosenbloom (1981)
    """
    return a * np.power(t + c, b)


def linear_model(t: np.ndarray, y0: float, slope: float) -> np.ndarray:
    """Linear: y(t) = y0 + slope * t"""
    return y0 + slope * t


# =============================================================================
# FIT RESULTS
# =============================================================================

@dataclass
class FitResult:
    """Container for model fitting results."""
    model_name: str
    params: Dict[str, float]
    predictions: np.ndarray
    sse: float
    n_params: int
    aic: float
    bic: float
    r_squared: float
    
    def __repr__(self):
        return f"{self.model_name}: R²={self.r_squared:.4f}, AIC={self.aic:.2f}"


# =============================================================================
# MODEL FITTING
# =============================================================================

def compute_aic_bic(sse: float, n: int, k: int) -> Tuple[float, float]:
    """
    Compute AIC and BIC for model comparison.
    Reference: Burnham & Anderson (2002)
    """
    if sse <= 0:
        sse = 1e-10
    ll_term = n * np.log(sse / n)
    aic = ll_term + 2 * k
    bic = ll_term + k * np.log(n)
    return aic, bic


def fit_all_models(t: np.ndarray, y: np.ndarray, 
                   is_bounded: bool = True,
                   verbose: bool = False) -> Dict[str, FitResult]:
    """
    Fit all candidate models to data.
    
    Parameters:
    -----------
    t : time points
    y : observations
    is_bounded : if True, use bounded models (approach/logistic). If False, use unbounded.
    verbose : print errors
    
    Models tested:
    - Exponential (e-governed)
    - Logistic (e-governed)  
    - Power law (NOT e-governed)
    - Linear (NOT e-governed)
    """
    results = {}
    n = len(t)
    ss_total = np.sum((y - np.mean(y))**2)
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    t_max = t.max()
    
    # 1. Exponential (approach to asymptote for bounded, pure growth for unbounded)
    if is_bounded:
        try:
            # Exponential approach: y = y_max - (y_max - y0) * e^(-kt)
            p0 = [y_max * 1.1, 0.05, y_min]
            bounds = ([y_max * 0.9, 1e-6, y_min * 0.5], 
                      [y_max * 2, 1.0, y_min * 1.5 + 1])
            popt, _ = curve_fit(exponential_approach, t, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = exponential_approach(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 3)
            r2 = 1 - sse / ss_total if ss_total > 0 else 0
            
            results['exponential'] = FitResult(
                model_name='Exponential (e-governed)',
                params={'y_max': popt[0], 'k': popt[1], 'y0': popt[2]},
                predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
            )
        except Exception as e:
            if verbose: print(f"Exponential approach fit failed: {e}")
    else:
        try:
            # Pure exponential growth: y = y0 * e^(kt)
            p0 = [y_min, 0.05]
            bounds = ([y_min * 0.5, 1e-6], [y_min * 2, 1.0])
            popt, _ = curve_fit(exponential_growth, t, y, p0=p0, bounds=bounds, maxfev=10000)
            pred = exponential_growth(t, *popt)
            sse = np.sum((y - pred)**2)
            aic, bic = compute_aic_bic(sse, n, 2)
            r2 = 1 - sse / ss_total if ss_total > 0 else 0
            
            results['exponential'] = FitResult(
                model_name='Exponential (e-governed)',
                params={'y0': popt[0], 'k': popt[1]},
                predictions=pred, sse=sse, n_params=2, aic=aic, bic=bic, r_squared=r2
            )
        except Exception as e:
            if verbose: print(f"Exponential growth fit failed: {e}")
    
    # 2. Logistic
    try:
        p0 = [y_max * 1.1, 0.1, t_max / 2]
        bounds = ([y_max * 0.9, 1e-6, 0], [y_max * 2, 2.0, t_max * 2])
        popt, _ = curve_fit(logistic_model, t, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = logistic_model(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 3)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['logistic'] = FitResult(
            model_name='Logistic (e-governed)',
            params={'K': popt[0], 'k': popt[1], 't0': popt[2]},
            predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Logistic fit failed: {e}")
    
    # 3. Power law: y = a * (t + c)^b
    try:
        p0 = [y_min, 1.0, 1.0]
        bounds = ([0, 0.01, 0.1], [y_max, 5.0, 100])
        popt, _ = curve_fit(power_law, t, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = power_law(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 3)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['power_law'] = FitResult(
            model_name='Power Law (NOT e-governed)',
            params={'a': popt[0], 'b': popt[1], 'c': popt[2]},
            predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Power law fit failed: {e}")
    
    # 4. Linear
    try:
        p0 = [y_min, y_range / t_max]
        popt, _ = curve_fit(linear_model, t, y, p0=p0, maxfev=10000)
        pred = linear_model(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 2)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['linear'] = FitResult(
            model_name='Linear (NOT e-governed)',
            params={'y0': popt[0], 'slope': popt[1]},
            predictions=pred, sse=sse, n_params=2, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Linear fit failed: {e}")
    
    return results


def fit_learning_curves(t: np.ndarray, y: np.ndarray, verbose: bool = False) -> Dict[str, FitResult]:
    """
    Fit models specifically for learning curves (accuracy 0-1, approach to ceiling).
    """
    results = {}
    n = len(t)
    ss_total = np.sum((y - np.mean(y))**2)
    
    y_min, y_max = y.min(), y.max()
    t_max = t.max()
    
    # 1. Exponential approach
    try:
        p0 = [min(y_max * 1.05, 1.0), 0.02, max(y_min, 0.1)]
        bounds = ([y_max, 1e-4, 0], [1.0, 0.5, y_max])
        popt, _ = curve_fit(exponential_approach, t, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = exponential_approach(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 3)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['exponential'] = FitResult(
            model_name='Exponential (e-governed)',
            params={'y_max': popt[0], 'k': popt[1], 'y0': popt[2]},
            predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Exponential fit failed: {e}")
    
    # 2. Logistic
    try:
        p0 = [min(y_max * 1.05, 1.0), 0.05, t_max / 3]
        bounds = ([y_max, 1e-4, 0], [1.0, 1.0, t_max])
        popt, _ = curve_fit(logistic_model, t, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = logistic_model(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 3)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['logistic'] = FitResult(
            model_name='Logistic (e-governed)',
            params={'K': popt[0], 'k': popt[1], 't0': popt[2]},
            predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Logistic fit failed: {e}")
    
    # 3. Power law (learning curve form): y = c - a/(t+1)^b
    try:
        def power_law_learning(t, a, b, c):
            return c - a / np.power(t + 1, b)
        
        p0 = [0.5, 0.5, min(y_max * 1.05, 1.0)]
        bounds = ([0, 0.01, y_max], [2, 3, 1.0])
        popt, _ = curve_fit(power_law_learning, t, y, p0=p0, bounds=bounds, maxfev=10000)
        pred = power_law_learning(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 3)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['power_law'] = FitResult(
            model_name='Power Law (NOT e-governed)',
            params={'a': popt[0], 'b': popt[1], 'c': popt[2]},
            predictions=pred, sse=sse, n_params=3, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Power law fit failed: {e}")
    
    # 4. Linear
    try:
        slope_est = (y_max - y_min) / t_max if t_max > 0 else 0
        p0 = [y_min, slope_est]
        popt, _ = curve_fit(linear_model, t, y, p0=p0, maxfev=10000)
        pred = linear_model(t, *popt)
        sse = np.sum((y - pred)**2)
        aic, bic = compute_aic_bic(sse, n, 2)
        r2 = 1 - sse / ss_total if ss_total > 0 else 0
        
        results['linear'] = FitResult(
            model_name='Linear (NOT e-governed)',
            params={'y0': popt[0], 'slope': popt[1]},
            predictions=pred, sse=sse, n_params=2, aic=aic, bic=bic, r_squared=r2
        )
    except Exception as e:
        if verbose: print(f"Linear fit failed: {e}")
    
    return results


def extract_time_constants(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Extract characteristic time constants for e-governed signature test.
    
    For e-governed approach y(t) = y_max(1 - e^(-t/τ)):
    - At t = τ: y = 63.2% of max improvement
    - At t = 2τ: y = 86.5%
    - At t = 3τ: y = 95.0%
    
    DIAGNOSTIC: For true e-governed dynamics, ratios should be 2.0 and 3.0.
    """
    y_min, y_max = y[0], y[-1]
    
    if y_max <= y_min:
        return {'error': 'No improvement detected'}
    
    y_norm = (y - y_min) / (y_max - y_min)
    
    results = {}
    targets = {'t_632': 0.632, 't_865': 0.865, 't_950': 0.950}
    
    for name, target in targets.items():
        idx = np.argmin(np.abs(y_norm - target))
        if idx < len(t) - 1 and y_norm[idx] < target:
            frac = (target - y_norm[idx]) / (y_norm[idx+1] - y_norm[idx] + 1e-10)
            results[name] = t[idx] + frac * (t[idx+1] - t[idx])
        else:
            results[name] = t[idx]
    
    tau = results.get('t_632', 1)
    if tau > 0:
        results['ratio_865_632'] = results.get('t_865', 0) / tau
        results['ratio_950_632'] = results.get('t_950', 0) / tau
        results['expected_865'] = 2.0
        results['expected_950'] = 3.0
    
    return results


def print_model_comparison(results: Dict[str, FitResult], title: str = "Model Comparison"):
    """Pretty print model comparison results."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'R²':<10} {'AIC':<12} {'ΔAIC':<10}")
    print(f"{'-'*60}")
    
    if not results:
        print("No models fit successfully!")
        return
    
    min_aic = min(r.aic for r in results.values())
    
    for name, result in sorted(results.items(), key=lambda x: x[1].aic):
        delta_aic = result.aic - min_aic
        marker = "← BEST" if delta_aic == 0 else ""
        print(f"{result.model_name:<30} {result.r_squared:<10.4f} {result.aic:<12.2f} {delta_aic:<10.2f} {marker}")
    
    print(f"{'='*60}")