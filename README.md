## Rigid Formation Control on Parametric Curves

Comprehensive implementation and documentation for finding and controlling rigid polygon formations on parametric curves while insuring curve "almost invariance". This repository contains the codebase used for demonstrations in formation control (IFAC 2026 paper).

---

## Quick start

Prerequisites: Python 3.8+ and numpy. This project uses the `pixi` tool for dependency management and convenient run scripts (optional).

Installation (pixi):

```bash
# Install pixi (optional)
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install
```

Run examples:

```bash
# Find polygons and generate example visualizations
pixi run finder-example

# Run multi-agent formation control simulation with animation
pixi run simulatation-example

# Run individual examples
pixi run example-triangle
pixi run example-square
pixi run example-pentagon
```

If you don't use `pixi`, you can run the Python scripts directly:

```bash
python formation-finder-example.py
python multi_agent_simulation.py
```

---

## Project structure

```
.
├── curve.py                      # Parametric curve class with projection/derivatives/curvature
├── rigid_formation_finder.py     # General N-gon finder on curves (Gauss-Newton/LM)
├── unicycle.py                   # Unicycle kinematic model
├── controllers.py                # SetpointLiftedTFL, Pose, Hybrid controllers
├── utils.py                      # Utility helpers (points_from_thetas, metrics)
├── example_usage.py              # Examples that find polygons on sample curves
├── multi_agent_simulation.py     # Multi-agent simulation + animation (TFL + blending)
├── REFACTORED_README.md          # Detailed developer documentation (this repo)
└── pixi.toml                     # Optional: pixi runtime and task definitions
```

---

## Core components & usage

This section summarizes the main modules and short usage snippets.

### curve.py — Curve class
Wraps a parametric planar curve P: [0, 2π] → R² and provides:
- Fast closest-point projection (lookup table + local optimization)
- Derivatives (adaptive finite differences)
- Curvature computation and scaling utilities

Example:

```python
from curve import Curve
import numpy as np

def deltoid(t):
    return (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t))

curve = Curve(deltoid, num_points=2000)
point = np.array([1.0, 0.5])
closest_t = curve.get_closest_point(point)
dp_dt, d2p_dt2 = curve.derivatives(closest_t)
kappa = curve.curvature(closest_t)
```

### rigid_formation_finder.py — RigidFormationFinder
Finds inscribed regular N-gons on general curves.

Key features:
- Damped Gauss-Newton solver with Levenberg-Marquardt regularization
- Multiple random restarts and feasibility projection
- Constraints for equal side lengths, diagonals, and interior angles

Basic usage:

```python
from curve import Curve
from rigid_formation_finder import RigidFormationFinder

curve = Curve(deltoid)
finder = RigidFormationFinder(curve, num_vertices=4)
thetas, error, history = finder.find_ngon()
if thetas is not None:
    print('Found square at parameters:', thetas)
    print(f'Residual error: {error:.3e}')
```

Backward compatibility: `find_square()` remains available for older scripts.


### controllers.py — Controllers
Provides three controllers used in experiments:

- SetpointLiftedTFLController: transverse feedback linearization with an internal dynamic for path following
- PoseController: pose stabilization to a target vertex
- HybridController: smooth blending between TFL (path following) and Pose (vertex stabilization), with collision avoidance and stuck recovery

Example of blending logic (conceptual):

σ = 0 → pure TFL (path following)
σ = 1 → pure Pose (vertex stabilization)
σ ∈ (0,1) → smooth blend based on revolutions and distance to the assigned vertex

Collision avoidance is implemented via the standard repulsive forces between agents and gain-scheduling based on σ.

### utils.py — Helpers
Utility functions such as `points_from_thetas(curve.P, thetas)` and `compute_formation_quality(points)` are available to evaluate side lengths, area, and coefficients of variation of the found formation.

---

## Extra notes testing

Unit test suggestions (quick checks):

```python
# Circle derivatives sanity check
curve = Curve(lambda t: (np.cos(t), np.sin(t)))
dp, d2p = curve.derivatives(0.0)
assert np.allclose(dp, [0, 1], atol=1e-3)

# Finder on perfect circle (square should be exact)
finder = RigidFormationFinder(curve, num_vertices=4)
thetas, error, _ = finder.find_ngon()
assert error < 1e-8
```

Development helper commands (pixi):

```bash
# Install dev deps
pixi install --feature dev

# Run tests (if present)
pixi run test

# Format and lint
pixi run format
pixi run lint
```

## License

BSD3 License 

---

## Authors & contact

Original implementation: Mohamed Elobaid <mohamed.elobaid@kaust.edu.sa>

