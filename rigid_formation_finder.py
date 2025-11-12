"""
Robust N-gon (regular polygon) finder on parametric curves.

Finds inscribed regular polygons (triangles, squares, pentagons, etc.) 
on closed planar curves using damped Gauss-Newton optimization with 
multiple restarts and feasibility constraints.
"""

import numpy as np
import math


def points_from_thetas(P, thetas):
    """
    Evaluate curve P at given parameter values.
    
    Parameters
    ----------
    P : callable
        Parametric curve function
    thetas : array_like
        Parameter values (will be wrapped to [0, 2π])
    
    Returns
    -------
    ndarray, shape (N, 2)
        Points on curve
    """
    thetas = np.mod(np.asarray(thetas), 2*np.pi)
    return np.array([P(t) for t in thetas])


class RigidFormationFinder:
    """
    Find inscribed regular N-gons on a parametric curve.
    
    Uses damped Gauss-Newton optimization with:
    - Multiple random restarts for robustness
    - Feasibility constraints (minimum side length)
    - Regularization on edge lengths, angles, and diagonals
    """
    
    def __init__(self, curve, num_vertices, 
                 min_side_fraction=0.02, 
                 restart_attempts=8, 
                 restart_spread=0.4,
                 max_iterations=200):
        """
        Initialize the formation finder.
        
        Parameters
        ----------
        curve : Curve
            Parametric curve object
        num_vertices : int
            Number of vertices (N) in the polygon
        min_side_fraction : float
            Minimum side length as fraction of curve scale
        restart_attempts : int
            Number of random restarts per initial guess
        restart_spread : float
            Magnitude of random perturbations for restarts
        max_iterations : int
            Maximum Gauss-Newton iterations
        """
        self.curve = curve
        self.P = curve.P
        self.N = int(num_vertices)
        self.min_side_fraction = float(min_side_fraction)
        self.restart_attempts = int(restart_attempts)
        self.restart_spread = float(restart_spread)
        self.max_iterations = int(max_iterations)
        
        # Compute minimum side threshold
        self.min_side_threshold = self.min_side_fraction * self.curve.scale
        
        # Special case: pentagon gets absolute minimum of 1.0
        if self.N == 5:
            self.min_side_threshold = max(self.min_side_threshold, 1.0)
    
    def _residuals_and_points(self, thetas):
        """
        Compute residual vector for regular N-gon constraints.
        
        Constraints:
        1. Equal edge lengths (N-1 independent)
        2. Equal diagonal lengths (for k=2..N//2)
        3. Equal interior angles (N-1 independent, via dot products)
        
        Parameters
        ----------
        thetas : ndarray, shape (N,)
            Parameter values for N vertices
        
        Returns
        -------
        residuals : ndarray
            Stacked residual vector
        points : ndarray, shape (N, 2)
            Evaluated vertex positions
        """
        thetas = np.mod(np.asarray(thetas).astype(float), 2*np.pi)
        pts = points_from_thetas(self.P, thetas)
        
        # Edge vectors
        edges = [pts[(i+1) % self.N] - pts[i] for i in range(self.N)]
        edge_lengths_sq = np.array([np.dot(v, v) for v in edges])
        
        # --- Constraint 1: Equal edge lengths ---
        # N-1 independent constraints: L²_i = L²_{i+1}
        r_len = [edge_lengths_sq[i] - edge_lengths_sq[(i+1) % self.N] 
                 for i in range(self.N - 1)]
        
        # Compute mean side length for diagonal constraints
        side_mean = np.mean(np.sqrt(edge_lengths_sq)) if np.mean(edge_lengths_sq) > 0 else 0.0
        
        # --- Constraint 2: Equal diagonal lengths ---
        # For regular N-gon, chord length for separation k is:
        # chord_k = side * sin(πk/N) / sin(π/N)
        r_diag = []
        if self.N > 2:
            denom = math.sin(math.pi / self.N)
            for k in range(2, (self.N // 2) + 1):
                # Compute actual chord distances for separation k
                actual_chords = np.array([
                    np.linalg.norm(pts[(i + k) % self.N] - pts[i]) 
                    for i in range(self.N)
                ])
                expected_chord = side_mean * (math.sin(math.pi * k / self.N) / denom)
                
                # Add deviations as residuals
                for val in (actual_chords - expected_chord):
                    r_diag.append(val)
        
        # --- Constraint 3: Equal interior angles ---
        # For regular polygon, dot product a·b is constant across vertices
        # where a = e[i-1], b = e[i] are incident edges at vertex i
        if self.N > 2:
            dot_products = []
            for i in range(self.N):
                a = edges[(i - 1) % self.N]
                b = edges[i]
                dot_products.append(np.dot(a, b))
            
            # N-1 independent constraints: dot_i = dot_{i+1}
            for i in range(self.N - 1):
                r_diag.append(dot_products[i] - dot_products[i + 1])
        
        # Stack all residuals
        residuals = np.hstack((r_len, r_diag))
        return residuals, pts
    
    def _jacobian_fd(self, fun, x, eps=1e-6):
        """
        Compute Jacobian using finite differences.
        
        Parameters
        ----------
        fun : callable
            Function returning (residuals, points)
        x : ndarray
            Parameter vector
        eps : float
            Finite difference step size
        
        Returns
        -------
        J : ndarray, shape (m, n)
            Jacobian matrix
        f0 : ndarray, shape (m,)
            Residuals at x
        """
        x = np.asarray(x, dtype=float)
        f0, _ = fun(x)
        m, n = f0.size, x.size
        J = np.zeros((m, n))
        
        # Adaptive step sizes
        h = eps * np.maximum(1.0, np.abs(x))
        
        for j in range(n):
            xp = x.copy()
            xm = x.copy()
            xp[j] += h[j]
            xm[j] -= h[j]
            
            fp, _ = fun(xp)
            fm, _ = fun(xm)
            
            J[:, j] = (fp - fm) / (2 * h[j])
        
        return J, f0
    
    def _constraint_margin(self, thetas):
        """
        Check feasibility: minimum side length constraint.
        
        Parameters
        ----------
        thetas : ndarray
            Parameter values
        
        Returns
        -------
        float
            Margin: positive if feasible, negative if infeasible
        """
        pts = points_from_thetas(self.P, thetas)
        sides = np.array([
            np.linalg.norm(pts[(i+1) % self.N] - pts[i]) 
            for i in range(self.N)
        ])
        return (sides.min() if sides.size > 0 else 0.0) - self.min_side_threshold
    
    def _make_feasible_guess(self, guess):
        """
        Perturb an infeasible guess to make it feasible.
        
        Spreads vertices apart until minimum side constraint is satisfied.
        
        Parameters
        ----------
        guess : ndarray
            Initial parameter values
        
        Returns
        -------
        ndarray or None
            Feasible guess, or None if couldn't make feasible
        """
        g = np.mod(np.asarray(guess, dtype=float), 2*np.pi)
        
        for attempt in range(24):
            if self._constraint_margin(g) >= 0:
                return g
            
            # Progressively spread vertices apart
            step = 0.05 * (attempt + 1)
            offsets = step * np.linspace(0.0, 1.0, g.size, endpoint=False)
            g = np.mod(g + offsets, 2*np.pi)
        
        return None
    
    def _damped_gauss_newton(self, guess):
        """
        Solve using damped Gauss-Newton with line search.
        
        Parameters
        ----------
        guess : ndarray
            Initial parameter values (should be feasible)
        
        Returns
        -------
        success : bool
            Whether converged to tolerance
        history : list of dict
            Optimization history with 'theta', 'points', 'error' at each iteration
        """
        x = np.atleast_1d(np.asarray(guess, dtype=float).copy())
        lam = 1e-3  # Levenberg-Marquardt damping parameter
        history = []
        
        for it in range(self.max_iterations):
            # Wrapped residual function
            def wrapped(z):
                return self._residuals_and_points(z)
            
            # Compute Jacobian and residuals
            J, r = self._jacobian_fd(wrapped, x)
            
            # Check for numerical issues
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(r))):
                lam *= 10
                if lam > 1e12:
                    break
                continue
            
            rnorm = np.linalg.norm(r)
            _, pts = wrapped(x)
            
            # Store iteration
            history.append({
                'theta': x.copy(),
                'points': pts.copy(),
                'error': float(rnorm)
            })
            
            # Check convergence
            if rnorm < 1e-9:
                return True, history
            
            # Damped Gauss-Newton step
            A = J.T @ J + lam * np.eye(J.shape[1])
            g = J.T @ r
            
            try:
                dx = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            
            # Line search with feasibility check
            alpha = 1.0
            accepted = False
            for _ in range(12):
                x_trial = x + alpha * dx
                
                # Check feasibility
                if self._constraint_margin(x_trial) < 0:
                    alpha *= 0.5
                    continue
                
                # Check for descent
                r_trial, _ = wrapped(x_trial)
                if np.linalg.norm(r_trial) < rnorm:
                    x = x_trial
                    lam = max(lam / 10, 1e-12)
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                lam *= 10
        
        # Return best result even if not converged
        if history:
            return (history[-1]['error'] < 1e-9), history
        
        return False, []
    
    def _run_with_restarts(self, base_guess):
        """
        Run optimization with multiple random restarts.
        
        Parameters
        ----------
        base_guess : ndarray
            Base initial guess
        
        Returns
        -------
        best_history : list or None
            Best optimization history
        best_error : float
            Best final error achieved
        best_size : float
            Mean side length of best solution
        """
        dim = base_guess.size
        
        # Deterministic random seed based on guess
        seed = int(np.sum(np.sin(base_guess) * 1e6)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        
        # Generate perturbations: no perturbation + random perturbations
        perturbations = [np.zeros(dim)] + [
            self.restart_spread * rng.uniform(-1.0, 1.0, size=dim) 
            for _ in range(self.restart_attempts)
        ]
        
        best_hist, best_err, best_size = None, 1e12, 0.0
        
        for p in perturbations:
            guess = base_guess + p
            feasible = self._make_feasible_guess(guess)
            
            if feasible is None:
                continue
            
            ok, hist = self._damped_gauss_newton(feasible)
            
            if not hist:
                continue
            
            final_err = hist[-1]['error']
            final_pts = hist[-1]['points']
            avg_side = np.mean([
                np.linalg.norm(final_pts[i] - final_pts[(i+1) % self.N]) 
                for i in range(self.N)
            ])
            
            # Prefer lower error; break ties by larger size
            if final_err < best_err - 1e-8 or \
               (abs(final_err - best_err) < 1e-8 and avg_side > best_size):
                best_hist, best_err, best_size = hist, final_err, avg_side
        
        return best_hist, best_err, best_size
    
    def find_ngon(self, initial_guesses=None, num_random_inits=20):
        """
        Find inscribed regular N-gon on the curve.
        
        Tries multiple initial guesses (uniform, rotated, random, user-provided)
        with restarts, and returns the best solution found.
        
        Parameters
        ----------
        initial_guesses : list of array_like, optional
            User-provided initial guesses for parameter values
        num_random_inits : int
            Number of additional random initial guesses
        
        Returns
        -------
        thetas : ndarray, shape (N,) or None
            Sorted parameter values of N-gon vertices (None if no solution)
        error : float or None
            Final residual error (None if no solution)
        history : list or None
            Optimization history (None if no solution)
        """
        # Prepare initial guesses
        guesses = []
        
        # Add user-provided guesses
        if initial_guesses is not None:
            for g in initial_guesses:
                guesses.append(np.mod(np.asarray(g, dtype=float), 2*np.pi))
        
        # Add uniform distribution
        base_uniform = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        guesses.append(base_uniform)
        
        # Add rotated uniforms
        for offset in [0.1, 0.3, 0.6]:
            guesses.append(np.mod(base_uniform + offset, 2*np.pi))
        
        # Add random seeds
        rng = np.random.default_rng(12345)
        for _ in range(num_random_inits):
            arr = np.sort(rng.random(self.N) * 2 * np.pi)
            guesses.append(arr)
        
        # Run optimization with restarts for each guess
        best_hist, best_err, best_size = None, 1e12, 0.0
        
        for g in guesses:
            hist, err, size = self._run_with_restarts(g)
            
            if hist is None:
                continue
            
            # Prefer lower error; break ties by larger size
            if err < best_err - 1e-8 or \
               (abs(err - best_err) < 1e-8 and size > best_size):
                best_hist, best_err, best_size = hist, err, size
        
        # Return None if no solution found
        if best_hist is None:
            return None, None, None
        
        final_thetas = best_hist[-1]['theta']
        return np.sort(np.mod(final_thetas, 2*np.pi)), best_err, best_hist
    
    def find_square(self, mode='4D', initial_guess=None, min_side_fraction=None):
        """
        Convenience method for finding squares (N=4).
        
        Maintains backward compatibility with RobustSquareFinder API.
        
        Parameters
        ----------
        mode : str
            '4D' for full optimization, '1D' for symmetric square (legacy)
        initial_guess : array_like, optional
            Initial parameter values
        min_side_fraction : float, optional
            Override minimum side fraction
        
        Returns
        -------
        ndarray or None
            Sorted parameter values [t1, t2, t3, t4], or None if no solution
        """
        # Only allow N=4 for this method
        if self.N != 4:
            raise ValueError("find_square() only works for N=4. Use find_ngon() instead.")
        
        # Update minimum side fraction if provided
        if min_side_fraction is not None:
            self.min_side_fraction = float(min_side_fraction)
            self.min_side_threshold = self.min_side_fraction * self.curve.scale
        
        # Prepare guesses based on mode
        if mode == '4D':
            guesses = [
                [0, np.pi/2, np.pi, 3*np.pi/2],
                [np.pi/8, 5*np.pi/8, 9*np.pi/8, 13*np.pi/8],
                [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4],
                [np.pi/6, 2*np.pi/3, 7*np.pi/6, 5*np.pi/3],
                [0.1, 1.6, 3.1, 4.7],
                [0.3, 1.8, 3.3, 4.9],
                [0.5, 2.0, 3.5, 5.1]
            ]
            if initial_guess is not None:
                guesses.insert(0, list(np.asarray(initial_guess)))
        else:  # 1D mode: symmetric square (legacy support)
            # For 1D, we optimize a single base parameter and place vertices at π/2 intervals
            # This is handled by searching over base angles
            guesses = [[g, g + np.pi/2, g + np.pi, g + 3*np.pi/2] 
                      for g in np.linspace(0, np.pi/2, 16)]
            if initial_guess is not None:
                base = float(initial_guess[0])
                guesses.insert(0, [base, base + np.pi/2, base + np.pi, base + 3*np.pi/2])
        
        # Run optimization
        thetas, error, history = self.find_ngon(initial_guesses=guesses, num_random_inits=0)
        
        # Return None if error too large
        if thetas is None or error > 1e-8:
            return None
        
        return thetas
