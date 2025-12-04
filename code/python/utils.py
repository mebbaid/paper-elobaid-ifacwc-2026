import numpy as np


def points_from_thetas(P, thetas):
    thetas = np.mod(np.asarray(thetas), 2*np.pi)
    return np.array([P(t) for t in thetas])


def compute_winding_number(curve, point):
    pts = curve.lookup_points
    point = np.asarray(point)
    
    vecs = pts - point
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    
    angle_diffs = np.diff(angles)
    angle_diffs = np.mod(angle_diffs + np.pi, 2*np.pi) - np.pi
    total_rotation = np.sum(angle_diffs)
    
    winding = total_rotation / (2 * np.pi)
    return winding


def compute_formation_quality(points):
    N = len(points)
    
    sides = np.array([
        np.linalg.norm(points[(i+1) % N] - points[i]) 
        for i in range(N)
    ])
    
    side_mean = np.mean(sides)
    side_std = np.std(sides)
    side_cv = side_std / side_mean if side_mean > 1e-9 else np.inf
    
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    
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
    return np.mod(theta + np.pi, 2*np.pi) - np.pi


def wrap_to_2pi(theta):
    return np.mod(theta, 2*np.pi)
