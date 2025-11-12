"""
Example: Finding regular polygons on various parametric curves.

Demonstrates the use of the refactored RigidFormationFinder class
to find inscribed triangles, squares, and pentagons.
"""

import numpy as np
import matplotlib.pyplot as plt
from curve import Curve
from rigid_formation_finder import RigidFormationFinder, points_from_thetas
import math


def sigma(u, L=1.0):
    """Cubic Hermite ramp 0->L with zero endpoint slopes."""
    return L * (3*u**2 - 2*u**3)


def gamma_c1_gear(s):
    num_teeth = 6
    num_segments = 2 * num_teeth
    segment_angle = 2 * np.pi / num_segments
    R_outer, R_inner = 2.3, 1.7
    s_norm = s % (2*np.pi)
    segment_idx = int(s_norm // segment_angle)
    s_start = segment_idx * segment_angle
    u = (s_norm - s_start) / segment_angle
    u_smooth = sigma(u)
    s_smooth = s_start + u_smooth * segment_angle
    radius = R_outer if segment_idx % 2 == 0 else R_inner
    return radius * np.cos(s_smooth), radius * np.sin(s_smooth)


def deltoid(t):
    """Deltoid curve (3-cusped hypocycloid)"""
    return (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t))


def rose(t, k=3, a=1.8):
    """Rose curve with k petals"""
    r = a * np.cos(k * t)
    return (r * np.cos(t), r * np.sin(t))


def lissajous(t, a=3, b=4, delta=0):
    """Lissajous curve"""
    return (2.0 * np.sin(a * t + np.pi/2), 2.0 * np.sin(b * t + delta))


def cassini_oval(t, A=3.0, B=3.15):
    """Cassini oval"""
    with np.errstate(invalid='ignore'):
        i = B**4 - A**4 * np.sin(2*t)**2
        r2 = A**2 * np.cos(2*t) + np.sqrt(np.clip(i, 0.0, np.inf))
        r = np.sqrt(np.clip(r2, 0.0, np.inf))
    return (r * np.cos(t), r * np.sin(t))


def make_cassini(A, B):
    def c(t):
        with np.errstate(invalid='ignore'):
            i = B**4 - A**4 * np.sin(2*t)**2
            r2 = A**2 * np.cos(2*t) + np.sqrt(np.clip(i, 0.0, np.inf))
            r = np.sqrt(np.clip(r2, 0.0, np.inf))
        return r*np.cos(t), r*np.sin(t)
    return c


def make_fourier_curve(a, b):
    # a and b are lists of (k, coeff) pairs
    def f(t):
        x = sum(c * np.cos(k * t) for k, c in a)
        y = sum(c * np.sin(k * t) for k, c in b)
        return x, y
    return f


def make_spirograph(R, r, d):
    # Spirograph parametrization using an integer-period multiplier so the
    # curve closes when R and r are integer-related. Use float arithmetic.
    p = 2 * np.pi * r / math.gcd(int(R), int(r))
    def s(t):
        s_t = t * p / (2 * np.pi)
        x = (R - r) * np.cos(s_t) + d * np.cos((R - r) * s_t / r)
        y = (R - r) * np.sin(s_t) - d * np.sin((R - r) * s_t / r)
        return x, y
    return s


# remove simple make_gear helper in favor of gamma_c1_gear (C1 parametrization)


def plot_polygon_on_curve(ax, curve, thetas, title='', color='C0'):
    """
    Plot a polygon on a curve.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    curve : Curve
        Parametric curve
    thetas : ndarray
        Parameter values for polygon vertices
    title : str
        Plot title
    color : str
        Polygon color
    """
    # Plot curve
    t = np.linspace(0, 2*np.pi, 800)
    curve_pts = np.array([curve.P(t_val) for t_val in t])
    ax.plot(curve_pts[:, 0], curve_pts[:, 1], 
            ls='--', color='0.6', lw=1.5, alpha=0.8, label='Curve')
    
    # Plot polygon
    pts = points_from_thetas(curve.P, thetas)
    closed = np.vstack([pts, pts[0]])
    
    ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.15)
    ax.plot(closed[:, 0], closed[:, 1], '-', color=color, lw=2.5, alpha=0.9, label=f'N={len(thetas)}')
    ax.scatter(pts[:, 0], pts[:, 1], s=60, color=color, edgecolor='k', linewidth=0.8, zorder=10)
    
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def main():
    """Run examples of finding N-gons on various curves."""
    
    # Example 1: Square on deltoid
    print("=" * 60)
    print("Example 1: Finding a square on the deltoid curve")
    print("=" * 60)
    
    curve_deltoid = Curve(deltoid, num_points=2000)

    # Find triangle, square, pentagon, and hexagon on deltoid
    print("=" * 60)
    print("Example: Finding triangle, square, pentagon, and hexagon on the deltoid curve")
    print("=" * 60)

    finder_tri = RigidFormationFinder(curve_deltoid, num_vertices=3, min_side_fraction=0.03)
    thetas_tri, err_tri, _ = finder_tri.find_ngon()

    finder_sq = RigidFormationFinder(curve_deltoid, num_vertices=4, min_side_fraction=0.03)
    thetas_sq, err_sq, _ = finder_sq.find_ngon()

    finder_p5 = RigidFormationFinder(curve_deltoid, num_vertices=5, min_side_fraction=0.03)
    thetas_p5, err_p5, _ = finder_p5.find_ngon()

    finder_hex = RigidFormationFinder(curve_deltoid, num_vertices=6, min_side_fraction=0.03)
    thetas_hex, err_hex, _ = finder_hex.find_ngon()

    def _print_result(name, thetas, err):
        if thetas is not None:
            pts = points_from_thetas(deltoid, thetas)
            sides = [np.linalg.norm(pts[(i+1)%len(thetas)] - pts[i]) for i in range(len(thetas))]
            print(f"✓ {name} found, err={err:.3e}")
            print(f"  Side mean: {np.mean(sides):.4f}, std: {np.std(sides):.4e}")
        else:
            print(f"✗ {name} not found")

    _print_result('Triangle on Deltoid', thetas_tri, err_tri)
    _print_result('Square on Deltoid', thetas_sq, err_sq)
    _print_result('Pentagon on Deltoid', thetas_p5, err_p5)
    _print_result('Hexagon on Deltoid', thetas_hex, err_hex)
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Triangle on deltoid
    if thetas_tri is not None:
        plot_polygon_on_curve(axes[0, 0], curve_deltoid, thetas_tri, 
                             title='Triangle on Deltoid', color='C0')
    
    # Plot 2: Square on deltoid
    if thetas_sq is not None:
        plot_polygon_on_curve(axes[0, 1], curve_deltoid, thetas_sq, 
                             title='Square on Deltoid', color='C1')
    
    # Plot 3: Pentagon on deltoid
    if thetas_p5 is not None:
        plot_polygon_on_curve(axes[1, 0], curve_deltoid, thetas_p5, 
                             title='Pentagon on Deltoid', color='C2')
    
    # Plot 4: Hexagon on deltoid
    if thetas_hex is not None:
        plot_polygon_on_curve(axes[1, 1], curve_deltoid, thetas_hex, 
                             title='Hexagon on Deltoid', color='C3')
    
    # Save the deltoid 2x2 figure separately
    try:
        fig.tight_layout()
        fig.savefig('deltoid_polygons.png', dpi=150, bbox_inches='tight')
        print("Saved deltoid figure to 'deltoid_polygons.png'")
    except Exception as e:
        print(f"Warning: failed to save deltoid figure: {e}")
    
    # --- Additional curves: build a list of curve definitions and plot them ---
    curve_definitions = [
        {'func': make_cassini(3, 3.15), 'title': "Cassini (b=1.05a)", 'mode': '1D'},
        {'func': make_cassini(3, 3), 'title': "Lemniscate (b=a)", 'mode': '4D', 'min_side_fraction': 0.02,
         'initial_guesses': [[-0.4, 0.4, np.pi-0.4, np.pi+0.4]]},
        {'func': lambda t, a=4, b=1: (a*np.cos(t), b*np.sin(t)), 'title': "Ellipse (a=4,b=1)", 'mode': '4D'},
        {'func': lambda t, k=5: (np.cos(k*t)*np.cos(t), np.cos(k*t)*np.sin(t)), 'title': "Rose (k=5)", 'mode': '4D'},
        {'func': lambda t: (np.sin(3*t + np.pi/2), np.sin(2*t)), 'title': "Lissajous (3:2)", 'mode': '4D'},
        {'func': lambda t: (np.sin(5*t), np.sin(4*t)), 'title': "Lissajous (5:4)", 'mode': '4D'},
        {'func': lambda t: (3*np.sign(np.cos(t))*np.abs(np.cos(t))**0.25, 3*np.sign(np.sin(t))*np.abs(np.sin(t))**0.25), 'title': "Superellipse (n=4)", 'mode': '4D'},
        {'func': lambda t: ((2 + 0.4*np.cos(7*t))*np.cos(t), (2 + 0.4*np.cos(7*t))*np.sin(t)), 'title': "7-Pointed Star", 'mode': '4D'},
        {'func': lambda t: ((2 + 0.5*np.sin(2*t))*np.cos(t), (1.5 + 0.3*np.sin(2*t))*np.sin(t)), 'title': "Peanut Shape", 'mode': '4D'},
        {'func': lambda t: (3*np.cos(t) - np.cos(3*t), 3*np.sin(t) - np.sin(3*t)), 'title': "Nephroid", 'mode': '4D'},
        # MODIFIED: Deltoid now uses the robust lifted controller
        {'func': lambda t: (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t)), 'title': "Deltoid", 'mode': '4D', 'controller': 'lifted_setpoint'},
        # MODIFIED: Gear now uses C¹ parametrization and the robust lifted controller
        {'func': gamma_c1_gear, 'title': "Gear (6 teeth, C¹)", 'mode': '4D', 'controller': 'lifted_setpoint'},
        {'func': lambda t: (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) + np.sin(2*t)), 'title': "Hypotrochoid", 'mode': '4D'},
        {'func': lambda t: (2*np.sin(t) * np.sin(2*t + 0.5), 2*np.cos(t) * np.cos(3*t)), 'title': "Harmonograph", 'mode': '4D'},
        {'func': make_spirograph(7, 3, 6), 'title': "Spirograph (7,3,6)", 'mode': '4D'},
        {'func': make_fourier_curve([(0,0),(1,2.5),(5,0.4)],[(0,0),(1,2.3),(5,0.5)]), 'title': "Fourier (1,5)", 'mode': '4D'},
    ]

    # Prepare plotting grid for additional curves
    n = len(curve_definitions)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes2 = np.array(axes2).reshape(-1)

    for idx, cdef in enumerate(curve_definitions):
        ax = axes2[idx]
        try:
            curve_obj = Curve(cdef['func'], num_points=2000)
            min_sf = cdef.get('min_side_fraction', 0.03)
            finder = RigidFormationFinder(curve_obj, num_vertices=4, min_side_fraction=min_sf)
            init = cdef.get('initial_guesses')
            if init is not None:
                thetas_res, err_res, _ = finder.find_ngon(initial_guesses=init)
            else:
                thetas_res, err_res, _ = finder.find_ngon()

            title = cdef.get('title', f'Curve {idx}')
            if thetas_res is not None:
                plot_polygon_on_curve(ax, curve_obj, thetas_res, title=title)
                ax.text(0.01, 0.01, f'err={err_res:.1e}', transform=ax.transAxes, fontsize=8, va='bottom')
                print(f"✓ [{title}] found square, err={err_res:.3e}")
            else:
                # Plot curve only and mark failure
                t = np.linspace(0, 2*np.pi, 800)
                pts = np.array([cdef['func'](tt) for tt in t])
                ax.plot(pts[:,0], pts[:,1], ls='--', color='0.6')
                ax.set_title(title + ' — no square')
                ax.axis('off')
                print(f"✗ [{title}] no square found")
        except Exception as e:
            ax.set_title(cdef.get('title', 'error'))
            ax.text(0.5, 0.5, 'error', ha='center')
            ax.axis('off')
            print(f"! [{cdef.get('title','?')}] error during processing: {e}")

    # Hide unused axes
    for j in range(n, rows*cols):
        axes2[j].axis('off')

    try:
        fig2.tight_layout()
        fig2.savefig('example_formations.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'example_formations.png'")
    except Exception as e:
        print(f"Warning: failed to save example formations figure: {e}")

    # Show all figures
    plt.show()


if __name__ == '__main__':
    main()
