import numpy as np
from scipy.optimize import minimize_scalar


class Curve:
    def __init__(self, parametric_func, num_points=2000):
        self.P = parametric_func
        thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x, y = np.vectorize(self.P)(thetas)
        
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        thetas, x, y = thetas[valid_mask], x[valid_mask], y[valid_mask]
        
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        duplicate_mask = np.append(True, dists > 1e-7)
        thetas, x, y = thetas[duplicate_mask], x[duplicate_mask], y[duplicate_mask]
        
        self.xlim = (np.min(x), np.max(x))
        self.ylim = (np.min(y), np.max(y))
        self.scale = max(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0])
        
        self.lookup_thetas = thetas
        self.lookup_points = np.vstack((x, y)).T
        
        self.centroid = (float(np.mean(x)), float(np.mean(y)))
    
    def get_closest_point(self, point):
        idx = np.argmin(np.linalg.norm(self.lookup_points - point, axis=1))
        t0 = self.lookup_thetas[idx]
        delta = 6 * (2 * np.pi / len(self.lookup_thetas))
        lo = (t0 - delta) % (2 * np.pi)
        hi = (t0 + delta) % (2 * np.pi)
        
        def distance(t):
            return np.linalg.norm(np.array(self.P(t)) - point)
        
        try:
            if lo < hi:
                res = minimize_scalar(distance, bounds=(lo, hi), 
                                    method='bounded', options={'xatol': 1e-6})
            else:
                res1 = minimize_scalar(distance, bounds=(lo, 2*np.pi), 
                                     method='bounded', options={'xatol': 1e-6})
                res2 = minimize_scalar(distance, bounds=(0, hi), 
                                     method='bounded', options={'xatol': 1e-6})
                res = res1 if res1.fun < res2.fun else res2
        except Exception:
            res = minimize_scalar(distance, bounds=(0, 2*np.pi), 
                                method='bounded', options={'xatol': 1e-6})
        
        return res.x
    
    def derivatives(self, t, h=None):
        if h is None:
            eps = np.finfo(float).eps
            h = max(1e-6, (eps**(1/3)) * max(1.0, abs(t)))
        
        p_plus = np.array(self.P(t + h))
        p_minus = np.array(self.P(t - h))
        p_t = np.array(self.P(t))
        
        dp_dt = (p_plus - p_minus) / (2 * h)
        d2p_dt2 = (p_plus - 2 * p_t + p_minus) / (h * h)
        
        return dp_dt, d2p_dt2
    
    def curvature(self, t):
        dp, d2p = self.derivatives(t)
        dp_norm_sq = dp[0]**2 + dp[1]**2
        
        if dp_norm_sq < 1e-8:
            return 0.0
        
        numerator = np.abs(dp[0] * d2p[1] - dp[1] * d2p[0])
        denominator = dp_norm_sq**1.5
        
        return numerator / denominator
    
    def __call__(self, t):
        return self.P(t)
