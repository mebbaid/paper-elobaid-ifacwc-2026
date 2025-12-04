import numpy as np
import matplotlib.pyplot as plt
from curve import Curve
from rigid_formation_finder import RigidFormationFinder, points_from_thetas
import math


def sigma(u, L=1.0):
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
    x = 2*np.cos(t) + np.cos(2*t)
    y = 2*np.sin(t) - np.sin(2*t)
    return x, y


def deltoid_derivatives(t):
    dx = -2*np.sin(t) - 2*np.sin(2*t)
    dy =  2*np.cos(t) - 2*np.cos(2*t)
    return np.array([dx, dy])


def deltoid_second_derivatives(t):
    d2x = -2*np.cos(t) - 4*np.cos(2*t)
    d2y = -2*np.sin(t) + 4*np.sin(2*t)
    return np.array([d2x, d2y])


def rose(t, k=3, a=1.8):
    r = a * np.cos(k * t)
    return (r * np.cos(t), r * np.sin(t))


def lissajous(t, a=3, b=4, delta=0):
    return (2.0 * np.sin(a * t + np.pi/2), 2.0 * np.sin(b * t + delta))


def cassini_oval(t, A=3.0, B=3.15):
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
    def f(t):
        x = sum(c * np.cos(k * t) for k, c in a)
        y = sum(c * np.sin(k * t) for k, c in b)
        return x, y
    return f


def make_spirograph(R, r, d):
    p = 2 * np.pi * r / math.gcd(int(R), int(r))
    def s(t):
        s_t = t * p / (2 * np.pi)
        x = (R - r) * np.cos(s_t) + d * np.cos((R - r) * s_t / r)
        y = (R - r) * np.sin(s_t) - d * np.sin((R - r) * s_t / r)
        return x, y
    return s

def plot_polygon_on_curve(ax, curve, thetas, title='', color='C0'):
    t = np.linspace(0, 2*np.pi, 800)
    curve_pts = np.array([curve.P(t_val) for t_val in t])
    ax.plot(curve_pts[:, 0], curve_pts[:, 1], 
            ls='--', color='0.6', lw=1.5, alpha=0.8, label='Curve')
    
    pts = points_from_thetas(curve.P, thetas)
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    pts_sorted = pts[sorted_idx]
    
    closed = np.vstack([pts_sorted, pts_sorted[0]])
    
    ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=0.15)
    ax.plot(closed[:, 0], closed[:, 1], '-', color=color, lw=2.5, alpha=0.9, label=f'N={len(thetas)}')
    ax.scatter(pts_sorted[:, 0], pts_sorted[:, 1], s=60, color=color, edgecolor='k', linewidth=0.8, zorder=10)
    
    ax.plot(centroid[0], centroid[1], 'rx', markersize=8, markeredgewidth=2, zorder=11)
    
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def ellipse(t, a=4, b=1):
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y


def ellipse_derivatives(t, a=4, b=1):
    dx = -a * np.sin(t)
    dy =  b * np.cos(t)
    return np.array([dx, dy])


def ellipse_second_derivatives(t, a=4, b=1):
    d2x = -a * np.cos(t)
    d2y = -b * np.sin(t)
    return np.array([d2x, d2y])


def lissajous_3_2(t):
    x = np.sin(3*t + np.pi/2)
    y = np.sin(2*t)
    return x, y


def lissajous_3_2_derivatives(t):
    dx = 3 * np.cos(3*t + np.pi/2)
    dy = 2 * np.cos(2*t)
    return np.array([dx, dy])


def lissajous_3_2_second_derivatives(t):
    d2x = -9 * np.sin(3*t + np.pi/2)
    d2y = -4 * np.sin(2*t)
    return np.array([d2x, d2y])


def main():
    # --- build a list of curve definitions and plot them ---
    curve_definitions = [
        {'func': make_cassini(3, 3.15), 'title': "Cassini (b=1.05a)", 'mode': '1D'},
        {'func': make_cassini(3, 3), 'title': "Lemniscate (b=a)", 'mode': '4D', 'min_side_fraction': 0.02,
         'initial_guesses': [
             [-0.5, 0.5, np.pi-0.5, np.pi+0.5],
             [-0.6, 0.6, np.pi-0.6, np.pi+0.6],
             [-0.3, 0.3, np.pi-0.3, np.pi+0.3],
             [-0.7, 0.7, np.pi-0.7, np.pi+0.7],
         ]},
        {'func': lambda t, a=4, b=1: (a*np.cos(t), b*np.sin(t)), 'title': "Ellipse (a=4,b=1)", 'mode': '4D'},
        {'func': lambda t, k=5: (np.cos(k*t)*np.cos(t), np.cos(k*t)*np.sin(t)), 'title': "Rose (k=5)", 'mode': '4D'},
        {'func': lambda t: (np.sin(3*t + np.pi/2), np.sin(2*t)), 'title': "Lissajous (3:2)", 'mode': '4D'},
        {'func': lambda t: (np.sin(5*t), np.sin(4*t)), 'title': "Lissajous (5:4)", 'mode': '4D'},
        {'func': lambda t: (3*np.sign(np.cos(t))*np.abs(np.cos(t))**0.25, 3*np.sign(np.sin(t))*np.abs(np.sin(t))**0.25), 'title': "Superellipse (n=4)", 'mode': '4D'},
        {'func': lambda t: ((2 + 0.4*np.cos(7*t))*np.cos(t), (2 + 0.4*np.cos(7*t))*np.sin(t)), 'title': "7-Pointed Star", 'mode': '4D'},
        {'func': lambda t: ((2 + 0.5*np.sin(2*t))*np.cos(t), (1.5 + 0.3*np.sin(2*t))*np.sin(t)), 'title': "Peanut Shape", 'mode': '4D'},
        {'func': lambda t: (3*np.cos(t) - np.cos(3*t), 3*np.sin(t) - np.sin(3*t)), 'title': "Nephroid", 'mode': '4D'},
        {'func': lambda t: (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t)), 'title': "Deltoid", 'mode': '4D', 'controller': 'lifted_setpoint'},
        {'func': gamma_c1_gear, 'title': "Gear (6 teeth, C¹)", 'mode': '4D', 'controller': 'lifted_setpoint'},
        {'func': lambda t: (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) + np.sin(2*t)), 'title': "Hypotrochoid", 'mode': '4D'},
        {'func': lambda t: (2*np.sin(t) * np.sin(2*t + 0.5), 2*np.cos(t) * np.cos(3*t)), 'title': "Harmonograph", 'mode': '4D'},
        {'func': make_spirograph(7, 3, 6), 'title': "Spirograph (7,3,6)", 'mode': '4D'},
        {'func': make_fourier_curve([(0,0),(1,2.5),(5,0.4)],[(0,0),(1,2.3),(5,0.5)]), 'title': "Fourier (1,5)", 'mode': '4D'},
    ]

    n = len(curve_definitions)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes2 = np.array(axes2).reshape(-1)

    for idx, cdef in enumerate(curve_definitions):
        ax = axes2[idx]
        try:
            dP = None
            d2P = None
            title = cdef.get('title')
            # Note: finite differences are enough for this family of examples
            # if title == 'Deltoid':
            #     dP = deltoid_derivatives
            #     d2P = deltoid_second_derivatives
            # elif title == 'Ellipse (a=4,b=1)':
            #     dP = lambda t: ellipse_derivatives(t, a=4, b=1)
            #     d2P = lambda t: ellipse_second_derivatives(t, a=4, b=1)
            # elif title == 'Lissajous (3:2)':
            #     dP = lissajous_3_2_derivatives
            #     d2P = lissajous_3_2_second_derivatives
            curve_obj = Curve(cdef['func'], num_points=2000)
            min_sf = cdef.get('min_side_fraction', 0.03)
            finder = RigidFormationFinder(curve_obj, num_vertices=4, min_side_fraction=min_sf)
            init = cdef.get('initial_guesses')
            num_rand = cdef.get('num_random_inits', 20)
            if init is not None:
                thetas_res, err_res, _ = finder.find_ngon(initial_guesses=init, num_random_inits=num_rand)
            else:
                thetas_res, err_res, _ = finder.find_ngon(num_random_inits=num_rand)

            title = cdef.get('title', f'Curve {idx}')
            if thetas_res is not None:
                plot_polygon_on_curve(ax, curve_obj, thetas_res, title=title)
                ax.text(0.01, 0.01, f'err={err_res:.1e}', transform=ax.transAxes, fontsize=8, va='bottom')
                print(f"✓ [{title}] found square, err={err_res:.3e}")
            else:
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

    for j in range(n, rows*cols):
        axes2[j].axis('off')

    try:
        fig2.tight_layout()
        fig2.savefig('example_formations.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to 'example_formations.png'")
    except Exception as e:
        print(f"Warning: failed to save example formations figure: {e}")

    plt.show()


if __name__ == '__main__':
    main()
