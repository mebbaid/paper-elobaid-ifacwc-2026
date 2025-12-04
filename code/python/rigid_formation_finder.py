import numpy as np
import math
from functools import lru_cache


def points_from_thetas(P, thetas):
    thetas = np.mod(np.asarray(thetas), 2*np.pi)
    try:
        result = P(thetas)
        if isinstance(result, tuple) and len(result) == 2:
            return np.column_stack(result)
    except (TypeError, ValueError):
        pass
    return np.array([P(t) for t in thetas])


class RigidFormationFinder:
    def __init__(self, curve, num_vertices, 
                 min_side_fraction=0.02, 
                 restart_attempts=8, 
                 restart_spread=0.4,
                 max_iterations=200,
                 target_center=None,
                 early_stop_error=1e-10):
        self.curve = curve
        self.P = curve.P
        self.N = int(num_vertices)
        self.min_side_fraction = float(min_side_fraction)
        self.restart_attempts = int(restart_attempts)
        self.restart_spread = float(restart_spread)
        self.max_iterations = int(max_iterations)
        self.target_center = np.array(target_center) if target_center is not None else None
        self.early_stop_error = float(early_stop_error)
        
        self.min_side_threshold = self.min_side_fraction * self.curve.scale
        
        if self.N == 5:
            self.min_side_threshold = max(self.min_side_threshold, 1.0)
        
        self._precompute_ngon_constants()
        self._point_cache = {}
        self._deriv_cache = {}
    
    def _precompute_ngon_constants(self):
        self.sin_pi_N = math.sin(math.pi / self.N)
        self.chord_ratios = {}
        for k in range(2, (self.N // 2) + 1):
            self.chord_ratios[k] = math.sin(math.pi * k / self.N) / self.sin_pi_N
    
    def _clear_cache(self):
        self._point_cache.clear()
        self._deriv_cache.clear()
    
    def _get_point(self, t):
        t_key = round(t % (2*np.pi), 10)
        if t_key not in self._point_cache:
            pt = self.P(t)
            self._point_cache[t_key] = np.array([pt[0], pt[1]])
        return self._point_cache[t_key]
    
    def _get_points(self, thetas):
        thetas = np.mod(np.asarray(thetas), 2*np.pi)
        return np.array([self._get_point(t) for t in thetas])
    
    def _get_derivative(self, t, h=1e-6):
        t_key = round(t % (2*np.pi), 10)
        if t_key not in self._deriv_cache:
            t_mod = t % (2*np.pi)
            p_plus = self._get_point(t_mod + h)
            p_minus = self._get_point(t_mod - h)
            self._deriv_cache[t_key] = (p_plus - p_minus) / (2 * h)
        return self._deriv_cache[t_key]
    
    def _residuals_and_points(self, thetas):
        thetas = np.mod(np.asarray(thetas).astype(float), 2*np.pi)
        pts = self._get_points(thetas)
        pts_next = np.roll(pts, -1, axis=0)
        edges = pts_next - pts
        edge_lengths_sq = np.sum(edges * edges, axis=1)        
        r_len = edge_lengths_sq[:-1] - edge_lengths_sq[1:]        
        edge_lengths = np.sqrt(edge_lengths_sq)
        side_mean = np.mean(edge_lengths) if np.mean(edge_lengths_sq) > 0 else 0.0
        
        r_diag = []
        if self.N > 2:
            for k in range(2, (self.N // 2) + 1):
                indices = np.arange(self.N)
                pts_k = pts[(indices + k) % self.N]
                actual_chords = np.linalg.norm(pts_k - pts, axis=1)
                expected_chord = side_mean * self.chord_ratios[k]
                r_diag.extend(actual_chords - expected_chord)            
            edges_prev = np.roll(edges, 1, axis=0)
            dot_products = np.sum(edges_prev * edges, axis=1)
            r_diag.extend(dot_products[:-1] - dot_products[1:])
        
        residuals = np.concatenate([r_len, r_diag]) if r_diag else r_len
        return residuals, pts
    
    def _jacobian_fd(self, fun, x, eps=1e-6):
        x = np.asarray(x, dtype=float)
        f0, pts0 = fun(x)
        m, n = f0.size, x.size
        J = np.zeros((m, n))
        
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
        pts = self._get_points(thetas)
        pts_next = np.roll(pts, -1, axis=0)
        sides = np.linalg.norm(pts_next - pts, axis=1)
        return sides.min() - self.min_side_threshold
    
    def _make_feasible_guess(self, guess):
        g = np.mod(np.asarray(guess, dtype=float), 2*np.pi)
        
        for attempt in range(24):
            if self._constraint_margin(g) >= 0:
                return g
            
            step = 0.05 * (attempt + 1)
            offsets = step * np.linspace(0.0, 1.0, g.size, endpoint=False)
            g = np.mod(g + offsets, 2*np.pi)
        
        return None
    
    def _damped_gauss_newton(self, guess):
        x = np.atleast_1d(np.asarray(guess, dtype=float).copy())
        lam = 1e-3
        history = []        
        self._clear_cache()
        
        for it in range(self.max_iterations):
            J, r = self._jacobian_fd(self._residuals_and_points, x)
            
            if not (np.all(np.isfinite(J)) and np.all(np.isfinite(r))):
                lam *= 10
                if lam > 1e12:
                    break
                continue
            
            rnorm = np.linalg.norm(r)
            _, pts = self._residuals_and_points(x)
            
            history.append({
                'theta': x.copy(),
                'points': pts.copy(),
                'error': float(rnorm)
            })
            
            if rnorm < 1e-9:
                return True, history
            
            JtJ = J.T @ J
            A = JtJ + lam * np.eye(J.shape[1])
            g = J.T @ r
            
            try:
                dx = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            
            alpha = 1.0
            accepted = False
            for _ in range(12):
                x_trial = x + alpha * dx
                
                if self._constraint_margin(x_trial) < 0:
                    alpha *= 0.5
                    continue
                
                r_trial, _ = self._residuals_and_points(x_trial)
                r_trial_norm = np.linalg.norm(r_trial)
                if r_trial_norm < rnorm:
                    x = x_trial
                    lam = max(lam / 10, 1e-12)
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                lam *= 10
        
        if history:
            return (history[-1]['error'] < 1e-9), history
        
        return False, []
    
    def _run_with_restarts(self, base_guess):
        dim = base_guess.size
        
        seed = int(np.sum(np.sin(base_guess) * 1e6)) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        
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
            if final_err < self.early_stop_error:
                final_pts = hist[-1]['points']
                pts_next = np.roll(final_pts, -1, axis=0)
                avg_side = np.mean(np.linalg.norm(pts_next - final_pts, axis=1))
                return hist, final_err, avg_side
            
            final_pts = hist[-1]['points']
            pts_next = np.roll(final_pts, -1, axis=0)
            avg_side = np.mean(np.linalg.norm(pts_next - final_pts, axis=1))
            
            if final_err < best_err - 1e-8 or \
               (abs(final_err - best_err) < 1e-8 and avg_side > best_size):
                best_hist, best_err, best_size = hist, final_err, avg_side
        
        return best_hist, best_err, best_size
    
    def find_ngon(self, initial_guesses=None, num_random_inits=20):
        guesses = []
        
        if initial_guesses is not None:
            for g in initial_guesses:
                guesses.append(np.mod(np.asarray(g, dtype=float), 2*np.pi))
        
        base_uniform = np.linspace(0, 2*np.pi, self.N, endpoint=False)
        guesses.append(base_uniform)
        
        for offset in [0.1, 0.3, 0.6]:
            guesses.append(np.mod(base_uniform + offset, 2*np.pi))
        
        rng = np.random.default_rng(12345)
        for _ in range(num_random_inits):
            arr = np.sort(rng.random(self.N) * 2 * np.pi)
            guesses.append(arr)
        
        valid_solutions = []
        
        for g in guesses:
            hist, err, size = self._run_with_restarts(g)
            
            if hist is None:
                continue
            
            if err < 1e-7:
                final_pts = hist[-1]['points']
                center = np.mean(final_pts, axis=0)
                valid_solutions.append({
                    'hist': hist,
                    'error': err,
                    'size': size,
                    'center': center,
                    'theta': hist[-1]['theta']
                })
                
                if err < self.early_stop_error:
                    break

        if not valid_solutions:
            return None, None, None
            
        if self.target_center is not None:
            def dist_key(sol):
                return np.linalg.norm(sol['center'] - self.target_center)
            
            best_sol = min(valid_solutions, key=dist_key)
        else:
            best_sol = min(valid_solutions, key=lambda s: (s['error'], -s['size']))
            
        return np.sort(np.mod(best_sol['theta'], 2*np.pi)), best_sol['error'], best_sol['hist']
    
    def find_square(self, mode='4D', initial_guess=None, min_side_fraction=None):
        if self.N != 4:
            raise ValueError("find_square() only works for N=4. Use find_ngon() instead.")
        
        if min_side_fraction is not None:
            self.min_side_fraction = float(min_side_fraction)
            self.min_side_threshold = self.min_side_fraction * self.curve.scale
        
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
        else:
            guesses = [[g, g + np.pi/2, g + np.pi, g + 3*np.pi/2] 
                      for g in np.linspace(0, np.pi/2, 16)]
            if initial_guess is not None:
                base = float(initial_guess[0])
                guesses.insert(0, [base, base + np.pi/2, base + np.pi, base + 3*np.pi/2])
        
        thetas, error, history = self.find_ngon(initial_guesses=guesses, num_random_inits=0)
        
        if thetas is None or error > 1e-8:
            return None
        
        return thetas
