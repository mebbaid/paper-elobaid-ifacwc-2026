"""
Utility functions for formation control and curve analysis.
"""

import numpy as np


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


def compute_winding_number(curve, point):
    """
    Compute winding number of a closed curve around a point.
    
    Uses angle accumulation method on the lookup table.
    
    Parameters
    ----------
    curve : Curve
        Closed parametric curve
    point : array_like, shape (2,)
        Query point
    
    Returns
    -------
    float
        Winding number (positive for counterclockwise)
    """
    pts = curve.lookup_points
    point = np.asarray(point)
    
    # Vectors from point to curve samples
    vecs = pts - point
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    
    # Unwrap angles to compute total rotation
    angle_diffs = np.diff(angles)
    angle_diffs = np.mod(angle_diffs + np.pi, 2*np.pi) - np.pi
    total_rotation = np.sum(angle_diffs)
    
    # Winding number
    winding = total_rotation / (2 * np.pi)
    return winding


def compute_formation_quality(points):
    """
    Compute quality metrics for a polygon formation.
    
    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Vertex positions
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'side_lengths': array of side lengths
        - 'side_mean': mean side length
        - 'side_std': standard deviation of side lengths
        - 'side_cv': coefficient of variation (std/mean)
        - 'area': polygon area (signed)
        - 'perimeter': total perimeter
    """
    N = len(points)
    
    # Side lengths
    sides = np.array([
        np.linalg.norm(points[(i+1) % N] - points[i]) 
        for i in range(N)
    ])
    
    side_mean = np.mean(sides)
    side_std = np.std(sides)
    side_cv = side_std / side_mean if side_mean > 1e-9 else np.inf
    
    # Area using shoelace formula
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
    # Perimeter
    perimeter = np.sum(sides)
    
    return {
        'side_lengths': sides,
        'side_mean': side_mean,
        'side_std': side_std,
        'side_cv': side_cv,
        'area': area,
        'perimeter': perimeter
    }


def normalize_angle(theta):
    """
    Normalize angle to (-π, π].
    
    Parameters
    ----------
    theta : float or ndarray
        Angle(s) in radians
    
    Returns
    -------
    float or ndarray
        Normalized angle(s)
    """
    return np.mod(theta + np.pi, 2*np.pi) - np.pi


def wrap_to_2pi(theta):
    """
    Wrap angle to [0, 2π).
    
    Parameters
    ----------
    theta : float or ndarray
        Angle(s) in radians
    
    Returns
    -------
    float or ndarray
        Wrapped angle(s)
    """
    return np.mod(theta, 2*np.pi)
