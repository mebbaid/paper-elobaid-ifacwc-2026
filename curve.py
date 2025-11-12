"""
Parametric Curve class with utilities for closest point projection,
derivatives, and curvature computation.
"""

import numpy as np
from scipy.optimize import minimize_scalar


class Curve:
    """
    Wrapper for a parametric curve P: [0, 2π] → R².
    
    Provides:
    - Closest point projection
    - Derivatives (first and second order)
    - Curvature computation
    - Lookup tables for fast approximate queries
    """
    
    def __init__(self, parametric_func, num_points=2000):
        """
        Initialize curve from a parametric function.
        
        Parameters
        ----------
        parametric_func : callable
            Function P(t) returning (x, y) for t ∈ [0, 2π]
        num_points : int
            Number of lookup points for fast approximate queries
        """
        self.P = parametric_func
        
        # Build lookup table
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x, y = np.vectorize(self.P)(thetas)
        
        # Filter out NaN/invalid points
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        thetas, x, y = thetas[valid_mask], x[valid_mask], y[valid_mask]
        
        # Remove duplicates (points too close together)
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        duplicate_mask = np.append(True, dists > 1e-7)
        thetas, x, y = thetas[duplicate_mask], x[duplicate_mask], y[duplicate_mask]
        
        # Store bounds and scale
        self.xlim = (np.min(x), np.max(x))
        self.ylim = (np.min(y), np.max(y))
        self.scale = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])
        
        # Lookup tables
        self.lookup_thetas = thetas
        self.lookup_points = np.vstack((x, y)).T
        
        # Geometric centroid (useful for winding number computations)
        self.centroid = (float(np.mean(x)), float(np.mean(y)))
    
    def get_closest_point(self, point):
        """
        Find the curve parameter t that minimizes ||P(t) - point||.
        
        Uses lookup table for initial guess, then local bounded optimization.
        
        Parameters
        ----------
        point : array_like, shape (2,)
            Query point [x, y]
        
        Returns
        -------
        float
            Parameter t ∈ [0, 2π] of closest point on curve
        """
        # Find nearest lookup point as seed
        idx = np.argmin(np.linalg.norm(self.lookup_points - point, axis=1))
        t0 = self.lookup_thetas[idx]
        
        # Local search window around t0
        delta = 6 * (2 * np.pi / len(self.lookup_thetas))
        lo = (t0 - delta) % (2 * np.pi)
        hi = (t0 + delta) % (2 * np.pi)
        
        # Distance function to minimize
        def distance(t):
            return np.linalg.norm(np.array(self.P(t)) - point)
        
        try:
            # Handle wrap-around at 0/2π boundary
            if lo < hi:
                res = minimize_scalar(distance, bounds=(lo, hi), 
                                    method='bounded', options={'xatol': 1e-6})
            else:
                # Split into two intervals
                res1 = minimize_scalar(distance, bounds=(lo, 2*np.pi), 
                                     method='bounded', options={'xatol': 1e-6})
                res2 = minimize_scalar(distance, bounds=(0, hi), 
                                     method='bounded', options={'xatol': 1e-6})
                res = res1 if res1.fun < res2.fun else res2
        except Exception:
            # Fallback: search entire domain
            res = minimize_scalar(distance, bounds=(0, 2*np.pi), 
                                method='bounded', options={'xatol': 1e-6})
        
        return res.x
    
    def derivatives(self, t, h=None):
        """
        Compute first and second derivatives using finite differences.
        
        Parameters
        ----------
        t : float
            Parameter value
        h : float, optional
            Step size for finite differences. If None, uses adaptive sizing.
        
        Returns
        -------
        dp_dt : ndarray, shape (2,)
            First derivative dP/dt
        d2p_dt2 : ndarray, shape (2,)
            Second derivative d²P/dt²
        """
        # Adaptive step size
        if h is None:
            eps = np.finfo(float).eps
            h = max(1e-6, (eps**(1/3)) * max(1.0, abs(t)))
        
        p_plus = np.array(self.P(t + h))
        p_minus = np.array(self.P(t - h))
        p_t = np.array(self.P(t))
        
        # Central differences
        dp_dt = (p_plus - p_minus) / (2 * h)
        d2p_dt2 = (p_plus - 2 * p_t + p_minus) / (h * h)
        
        return dp_dt, d2p_dt2
    
    def curvature(self, t):
        """
        Compute signed curvature at parameter t.
        
        κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        
        Parameters
        ----------
        t : float
            Parameter value
        
        Returns
        -------
        float
            Curvature κ
        """
        dp, d2p = self.derivatives(t)
        dp_norm_sq = dp[0]**2 + dp[1]**2
        
        if dp_norm_sq < 1e-8:
            return 0.0
        
        # Signed curvature
        numerator = np.abs(dp[0] * d2p[1] - dp[1] * d2p[0])
        denominator = dp_norm_sq**1.5
        
        return numerator / denominator
    
    def __call__(self, t):
        """Allow curve to be called directly: curve(t) == curve.P(t)"""
        return self.P(t)
