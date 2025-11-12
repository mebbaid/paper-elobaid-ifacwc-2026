"""
Multi-Agent Coordinated Path Following to Form a Square.

This example demonstrates:
- Finding inscribed squares on various parametric curves
- Multi-agent coordination using hybrid TFL + Pose control
- Smooth blending from path following to formation stabilization
- Collision avoidance among agents
- Real-time animation with trail visualization

Uses the refactored modules: curve.py, rigid_formation_finder.py, 
unicycle.py, controllers.py
"""

import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from curve import Curve
from rigid_formation_finder import RigidFormationFinder, points_from_thetas
from unicycle import Unicycle
from controllers import SetpointLiftedTFLController, PoseController, HybridController


# =============================================================================
# Curve Definitions
# =============================================================================

def deltoid(t):
    """Deltoid curve (3-cusped hypocycloid)"""
    return (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t))


def rose(t, k=3, a=1.8):
    """Rose curve with k petals"""
    r = a * np.cos(k * t)
    return (r * np.cos(t), r * np.sin(t))


def cassini_oval(t, A=3.0, B=3.15):
    """Cassini oval"""
    with np.errstate(invalid='ignore'):
        i = B**4 - A**4 * np.sin(2*t)**2
        r2 = A**2 * np.cos(2*t) + np.sqrt(np.clip(i, 0.0, np.inf))
        r = np.sqrt(np.clip(r2, 0.0, np.inf))
    return (r * np.cos(t), r * np.sin(t))


def lissajous(t, a=3, b=4, delta=0):
    """Lissajous curve (a:b)"""
    return (2.0 * np.sin(a * t + np.pi/2), 2.0 * np.sin(b * t + delta))


def gear(s, num_teeth=6, R_outer=2.3, R_inner=1.7):
    """Smooth gear-like curve with C¹ continuity"""
    num_segments = 2 * num_teeth
    segment_angle = 2 * np.pi / num_segments
    s_norm = s % (2*np.pi)
    segment_idx = int(s_norm // segment_angle)
    s_start = segment_idx * segment_angle
    u = (s_norm - s_start) / segment_angle
    
    # Cubic smoothstep
    def smoothstep(t):
        return 3*t**2 - 2*t**3
    
    u_smooth = smoothstep(u)
    s_smooth = s_start + u_smooth * segment_angle
    radius = R_outer if segment_idx % 2 == 0 else R_inner
    return (radius * np.cos(s_smooth), radius * np.sin(s_smooth))


# =============================================================================
# Multi-Agent Visualizer Class
# =============================================================================

class MultiAgentVisualizer:
    """
    Visualizer for multi-agent formation control with animations.
    
    Features:
    - Real-time position updates
    - Colored trails showing agent history
    - Formation edges connecting agents
    - Blending parameter (sigma) display
    - Revolution counting
    """
    
    def __init__(self, curve, ax, target_thetas, curve_title="Curve"):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        curve : Curve
            Parametric curve object
        ax : matplotlib.axes.Axes
            Axes to plot on
        target_thetas : ndarray or None
            Target vertex parameters for square formation
        curve_title : str
            Title for the plot
        """
        self.curve = curve
        self.ax = ax
        self.target_thetas = target_thetas
        self.curve_title = curve_title
        self.simulation_history = []  # list of (states, sigmas)
        self.time_history = []        # list of time stamps
        self.rms_error_history = []   # list of RMS position error wrt vertices
        self.min_dist_history = []    # list of min inter-agent distances
        # Extended metrics (Option B)
        self.side_cov_history = []        # coefficient of variation of side lengths
        self.distance_margin_history = [] # min_distance - d_safe
        self.curve_adherence_history = [] # mean distance of agents from curve projection
        self.sigma_mean_history = []      # mean sigma over agents
        self.sigma_std_history = []       # std sigma over agents
        
        # Agent and controller lists
        self.agents = []
        self.controllers = []
        
        # Visualization parameters
        self.trail_len = 200  # Number of frames to show in trails
        
        # Use perceptual colormap for agents
        self.agent_colors = list(plt.cm.viridis(np.linspace(0.15, 0.85, 4)))
        
        # Initialize agents and controllers if target square exists
        if target_thetas is None:
            print(f"  No target square found for {curve_title}")
            return
        
        self._initialize_agents_and_controllers()
    
    def _initialize_agents_and_controllers(self):
        """Initialize 4 agents near target vertices with controllers."""
        target_points = points_from_thetas(self.curve.P, self.target_thetas)
        
        # Start agents at a distance from vertices (with some noise)
        noise_radius = 0.18 * self.curve.scale
        
        for i in range(4):
            px, py = target_points[i]
            
            # Add radial offset with random angle
            angle = np.random.uniform(0, 2*np.pi)
            x = px + noise_radius * np.cos(angle)
            y = py + noise_radius * np.sin(angle)
            theta = np.random.uniform(-np.pi, np.pi)
            
            # Create agent
            agent = Unicycle(x, y, theta, v=0.0)
            agent.assigned_vertex = np.array([px, py])
            
            # Initialize path parameter tracking
            agent.prev_s = self.curve.get_closest_point(np.array([x, y]))
            agent.cum_s = agent.prev_s
            agent.s_revs = agent.cum_s / (2*np.pi)
            agent.prev_pos = np.array([x, y], dtype=float)
            
            self.agents.append(agent)
            
            # Create controllers
            # TFL: cruise along curve with desired progress velocity
            s_dot_des = 0.1  # desired parameter velocity
            vz_cmd = 2.0 * s_dot_des  # K_LIFT * s_dot
            tfl = SetpointLiftedTFLController(
                self.curve, 
                target_path_param=None,  # No fixed target, cruise mode
                K_LIFT=2.0,
                vz_cmd=vz_cmd,
                v_min=0.05,
                k_heading_fb=3.0,   # approximate mapping to (Kp,Kd)
                kv_speed_fb=3.0
            )
            
            # Pose: stabilize to assigned vertex
            pose = PoseController(kp_pos=3.0, kv=4.0, k_theta=5.0, v_max=0.6)
            
            # Hybrid: blend TFL -> Pose
            hybrid = HybridController(
                self.curve, tfl, pose,
                N_target=1.0,            # Blend target revolutions per spec
                d_switch_frac=0.15,      # Relaxed distance threshold for blending
                safety_radius_frac=0.06,
                avoid_on_frac=0.10
            )
            
            self.controllers.append(hybrid)
        
        # Give all agents access to each other for collision avoidance
        for agent in self.agents:
            agent._all_agents = self.agents
    
    def run_simulation(self, duration=120.0, dt=0.02):
        """
        Run the simulation for specified duration.
        
        Parameters
        ----------
        duration : float
            Simulation time in seconds
        dt : float
            Time step
        """
        if not self.agents:
            return
        
        num_steps = int(duration / dt)
        
        t_now = 0.0
        # Logging setup for cumulative s
        log_interval = 20.0
        steps_per_log = max(1, int(round(log_interval / dt)))
        for step in range(num_steps):
            current_states = [agent.state.copy() for agent in self.agents]
            
            # Update revolution counting and blending parameter for each agent
            for i, agent in enumerate(self.agents):
                x, y = agent.state[0], agent.state[1]
                pos = np.array([x, y])

                # Get current path parameter around current position
                s_curr = self.curve.get_closest_point(pos)

                # Estimate continuous parameter increment using projected displacement on tangent at prev_s
                if agent.prev_s is not None and agent.prev_pos is not None:
                    dp_ds, _ = self.curve.derivatives(agent.prev_s)
                    dp_norm_sq = float(dp_ds[0]**2 + dp_ds[1]**2)
                    if dp_norm_sq > 1e-10:
                        delta_p = pos - agent.prev_pos
                        ds_est = float(np.dot(delta_p, dp_ds) / dp_norm_sq)
                        # Clamp to avoid jumps due to crossings
                        ds_est = float(np.clip(ds_est, -0.2, 0.2))
                    else:
                        ds_est = 0.0
                    agent.cum_s += ds_est
                else:
                    # Fallback to wrapped difference
                    ds = s_curr - (agent.prev_s or s_curr)
                    if ds < -np.pi:
                        ds += 2*np.pi
                    elif ds > np.pi:
                        ds -= 2*np.pi
                    agent.cum_s += ds

                agent.prev_s = s_curr
                agent.prev_pos = pos
                agent.s_revs = agent.cum_s / (2*np.pi)
                
                # Compute blending parameter
                if agent.assigned_vertex is not None:
                    dist_to_vertex = np.linalg.norm(
                        agent.state[:2] - agent.assigned_vertex
                    )
                    sigma = self.controllers[i].blending_sigma(
                        agent.s_revs, dist_to_vertex
                    )
                    agent.sigma = sigma
                else:
                    agent.sigma = 0.0
            
            # Compute controls for all agents
            controls = []
            for i, agent in enumerate(self.agents):
                u, sigma = self.controllers[i].compute_controls(agent, dt)
                controls.append(u)
            
            # Update all agents
            for agent, u in zip(self.agents, controls):
                agent.update(u, dt)
            
            # Store history and metrics
            sigmas = [agent.sigma for agent in self.agents]
            self.simulation_history.append((current_states, sigmas))
            self.time_history.append(t_now)
            sigma_mean = float(np.mean(sigmas))
            sigma_std = float(np.std(sigmas))
            self.sigma_mean_history.append(sigma_mean)
            self.sigma_std_history.append(sigma_std)

            # Metrics: RMS error to vertices and min inter-agent distance
            positions = np.array([s[:2] for s in current_states])
            vertices = np.array([a.assigned_vertex for a in self.agents])
            if vertices.shape == positions.shape:
                errs = np.linalg.norm(positions - vertices, axis=1)
                rms_err = float(np.sqrt(np.mean(errs**2)))
            else:
                rms_err = np.nan
            self.rms_error_history.append(rms_err)

            # Min distance among agents
            if len(positions) >= 2:
                dmin = np.inf
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dij = np.linalg.norm(positions[i] - positions[j])
                        if dij < dmin:
                            dmin = dij
                self.min_dist_history.append(float(dmin))
            else:
                self.min_dist_history.append(np.nan)

            # Side length coefficient of variation (order uses agent index sequence)
            if positions.shape[0] == 4:
                sides = [np.linalg.norm(positions[(i+1) % 4] - positions[i]) for i in range(4)]
                m_side = np.mean(sides)
                if m_side > 1e-9:
                    cov = float(np.std(sides) / m_side)
                else:
                    cov = np.nan
                self.side_cov_history.append(cov)
            else:
                self.side_cov_history.append(np.nan)

            # Curve adherence: mean distance to closest projected point on curve
            adher_dists = []
            for p in positions:
                s_proj = self.curve.get_closest_point(p)
                proj = np.array(self.curve.P(s_proj))
                adher_dists.append(np.linalg.norm(p - proj))
            self.curve_adherence_history.append(float(np.mean(adher_dists)))

            # Distance margin relative to safety threshold (use configured fraction 0.06)
            d_safe = 0.06 * self.curve.scale
            if len(positions) >= 2 and not np.isnan(self.min_dist_history[-1]):
                self.distance_margin_history.append(self.min_dist_history[-1] - d_safe)
            else:
                self.distance_margin_history.append(np.nan)

            # Periodic log of cumulative revolutions and sigma
            if step % steps_per_log == 0:
                s_revs_list = [f"{a.s_revs:.2f}" for a in self.agents]
                sig_list = [f"{a.sigma:.2f}" for a in self.agents]
                print(f"[{self.curve_title} t={t_now:6.1f}s] s_revs=" + ", ".join(s_revs_list) +
                      " | sigma=" + ", ".join(sig_list))

            t_now += dt
    
    def setup_plot(self, title=""):
        """
        Setup the plot with curve, target square, and agent markers.
        
        Parameters
        ----------
        title : str
            Plot title
        """
        self.ax.set_title(title, fontsize=14)
        
        if self.target_thetas is None:
            self.ax.text(0.5, 0.5, 'No solution', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            return
        
        # Draw curve
        t = np.linspace(0, 2*np.pi, 400)
        x, y = np.vectorize(self.curve.P)(t)
        self.ax.plot(x, y, color='#2ca02c', linestyle='--', 
                    lw=1.0, alpha=0.9, solid_capstyle='round')
        
        # Draw target square
        target_points = points_from_thetas(self.curve.P, self.target_thetas)
        centroid = np.mean(target_points, axis=0)
        angles = np.arctan2(target_points[:,1] - centroid[1], 
                           target_points[:,0] - centroid[0])
        sorted_points = target_points[np.argsort(angles)]
        
        for i in range(4):
            p1 = sorted_points[i]
            p2 = sorted_points[(i+1) % 4]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        'k-', lw=2.0, alpha=0.3)
            self.ax.plot(p1[0], p1[1], 'ko', markersize=8, alpha=0.4)
        
        # Agent markers (will be updated in animation)
        self.agent_plots = []
        for i in range(4):
            plot, = self.ax.plot([], [], 'o', markersize=12, 
                               color=self.agent_colors[i],
                               markeredgecolor='k', markeredgewidth=1.5,
                               zorder=10)
            self.agent_plots.append(plot)
        
        # Trail collections (with fading effect)
        self.trail_collections = []
        for i in range(4):
            lc = LineCollection([], linewidths=2.5, 
                              colors=self.agent_colors[i],
                              alpha=0.6, zorder=5)
            self.ax.add_collection(lc)
            self.trail_collections.append(lc)
        
        # Formation edges (connecting agents)
        self.edge_plot, = self.ax.plot([], [], '--', linewidth=1.5, 
                                       color='blue', alpha=0.6, zorder=4)
        
        # Sigma text displays
        self.sigma_texts = [
            self.ax.text(0, 0, "", fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.75, 
                                boxstyle='round,pad=0.2'))
            for _ in range(4)
        ]
        
        # Set limits
        self.ax.set_xlim(self.curve.xlim[0]*1.2, self.curve.xlim[1]*1.2)
        self.ax.set_ylim(self.curve.ylim[0]*1.2, self.curve.ylim[1]*1.2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
    
    def update_frame(self, frame):
        """
        Update animation frame.
        
        Parameters
        ----------
        frame : int
            Frame number
        
        Returns
        -------
        list
            List of updated artists
        """
        if not self.simulation_history or frame >= len(self.simulation_history):
            return []
        
        states, sigmas = self.simulation_history[frame]
        points = np.array([s[:2] for s in states])
        
        # Trail start index
        start = max(0, frame - self.trail_len)
        
        # Update agent positions and trails
        for i, plot in enumerate(self.agent_plots):
            # Current position
            plot.set_data([points[i, 0]], [points[i, 1]])
            
            # Trail with fading effect
            if frame > 0:
                trail_states = [self.simulation_history[f][0][i][:2] 
                              for f in range(start, frame + 1)]
                trail_points = np.array(trail_states)
                
                # Create line segments
                segments = []
                for j in range(len(trail_points) - 1):
                    segments.append([trail_points[j], trail_points[j+1]])
                
                if segments:
                    # Fade out older trail segments
                    n_seg = len(segments)
                    alphas = np.linspace(0.1, 0.8, n_seg)
                    
                    self.trail_collections[i].set_segments(segments)
                    self.trail_collections[i].set_alpha(alphas[-1] if n_seg > 0 else 0.6)
            
            # Sigma text
            sigma_val = sigmas[i]
            offset = 0.15 * self.curve.scale
            self.sigma_texts[i].set_position((points[i, 0] + offset, 
                                             points[i, 1] + offset))
            self.sigma_texts[i].set_text(f'σ={sigma_val:.2f}')
            
            # Color text based on sigma (green when settled)
            if sigma_val > 0.9:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('#90EE90')
            elif sigma_val > 0.5:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('#FFFACD')
            else:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('white')
        
        # Update formation edges
        closed_points = np.vstack([points, points[0]])
        self.edge_plot.set_data(closed_points[:, 0], closed_points[:, 1])
        
        # Return updated artists
        artists = []
        artists.extend(self.agent_plots)
        artists.extend(self.trail_collections)
        artists.append(self.edge_plot)
        artists.extend(self.sigma_texts)
        
        return artists

    # ---- Snapshot rendering (static) ----
    def render_snapshot(self, ax, frame, title=""):
        """Render a static snapshot at a given frame index onto ax."""
        ax.clear()
        ax.set_title(title, fontsize=12)

        # Draw curve
        t = np.linspace(0, 2*np.pi, 400)
        x, y = np.vectorize(self.curve.P)(t)
        ax.plot(x, y, color='#2ca02c', linestyle='--', lw=1.0, alpha=0.9)

        # Draw target square
        if self.target_thetas is not None:
            target_points = points_from_thetas(self.curve.P, self.target_thetas)
            closed = np.vstack([target_points, target_points[0]])
            ax.plot(closed[:,0], closed[:,1], 'k-', lw=1.5, alpha=0.4)
            ax.scatter(target_points[:,0], target_points[:,1], c='k', s=20, alpha=0.6)

        # Draw trails up to frame and current agent positions
        if not self.simulation_history:
            return
        frame = max(0, min(frame, len(self.simulation_history)-1))
        states, sigmas = self.simulation_history[frame]
        points = np.array([s[:2] for s in states])

        # Trails: plot a faint path for each agent up to frame
        for i in range(len(self.agents)):
            trail_states = [self.simulation_history[f][0][i][:2] for f in range(0, frame+1)]
            trail_points = np.array(trail_states)
            ax.plot(trail_points[:,0], trail_points[:,1], '-', color=self.agent_colors[i], alpha=0.5, lw=1.5)

        # Current positions
        for i in range(len(self.agents)):
            ax.plot(points[i,0], points[i,1], 'o', color=self.agent_colors[i], markeredgecolor='k', markersize=8)
            ax.text(points[i,0], points[i,1], f"  σ={sigmas[i]:.2f}", fontsize=8)

        # Formation edges (current)
        closed_pts = np.vstack([points, points[0]])
        ax.plot(closed_pts[:,0], closed_pts[:,1], '--', color='blue', alpha=0.6)

        ax.set_xlim(self.curve.xlim[0]*1.2, self.curve.xlim[1]*1.2)
        ax.set_ylim(self.curve.ylim[0]*1.2, self.curve.ylim[1]*1.2)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """
    Main function: setup curves, find squares, run simulations, animate.
    """
    # Deterministic seeding for reproducibility of initial conditions
    np.random.seed(42)
    
    # Define curves to simulate
    curve_definitions = [
        {
            'func': deltoid,
            'title': 'Deltoid',
            'mode': '4D',
            'min_side_fraction': 0.05
        },
        {
            'func': lambda t: lissajous(t, a=3, b=2, delta=0.0),
            'title': 'Lissajous (3:2)',
            'mode': '4D',
            'min_side_fraction': 0.05
        },
    ]
    
    # Create figure with subplots
    n_curves = len(curve_definitions)
    fig, axes = plt.subplots(1, n_curves, figsize=(6*n_curves, 6))
    if n_curves == 1:
        axes = [axes]
    
    visualizers = []
    
    print("\n" + "="*70)
    print("Multi-Agent Formation Control Simulation")
    print("="*70)
    
    for i, definition in enumerate(curve_definitions):
        print(f"\n[{i+1}/{n_curves}] Processing: {definition['title']}")
        
        # Create curve
        curve = Curve(definition['func'], num_points=2000)
        print(f"  Curve scale: {curve.scale:.3f}")
        
        # Find square
        finder = RigidFormationFinder(
            curve, 
            num_vertices=4,
            min_side_fraction=definition.get('min_side_fraction', 0.05)
        )
        
        initial_guess = definition.get('guess')
        if initial_guess:
            target_thetas, error, _ = finder.find_ngon(
                initial_guesses=[initial_guess]
            )
        else:
            target_thetas = finder.find_square(mode=definition.get('mode', '4D'))
            error = 0.0  # find_square doesn't return error
        
        if target_thetas is not None:
            pts = points_from_thetas(curve.P, target_thetas)
            sides = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
            print(f"  ✓ Square found")
            print(f"    Side lengths: {[f'{s:.4f}' for s in sides]}")
            print(f"    Mean: {np.mean(sides):.4f}, Std: {np.std(sides):.4e}")
        else:
            print(f"  ✗ No square found")
        
        # Create visualizer and run simulation
        viz = MultiAgentVisualizer(
            curve, axes[i], target_thetas, 
            curve_title=definition['title']
        )
        
        if target_thetas is not None:
            print(f"  Running simulation...")
            # Increased duration for better convergence (Option B enhancement)
            viz.run_simulation(duration=180.0, dt=0.03)
            print(f"  Simulation complete: {len(viz.simulation_history)} frames")
            # Final diagnostics per agent
            final_states, final_sigmas = viz.simulation_history[-1]
            final_positions = np.array([s[:2] for s in final_states])
            print("    Final agent diagnostics:")
            for ai, (pos, sigma_val) in enumerate(zip(final_positions, final_sigmas), start=1):
                dist_vert = np.linalg.norm(pos - viz.agents[ai-1].assigned_vertex)
                print(f"      Agent {ai}: dist_to_vertex={dist_vert:.4f} sigma={sigma_val:.3f}")
        
        viz.setup_plot(title=definition['title'])
        visualizers.append(viz)
    
    # After simulations, produce snapshot figure and metrics for second curve
    # Identify visualizers for convenience
    if len(visualizers) == 2:
        viz_deltoid = visualizers[0]
        viz_liss = visualizers[1]
    else:
        viz_deltoid = visualizers[0]
        viz_liss = visualizers[-1]

    # --- Snapshot Figure ---
    # Times: 0s, 20s, and end-of-simulation per curve
    def time_to_frame(viz, t_target):
        if not viz.time_history:
            return 0
        arr = np.array(viz.time_history)
        return int(np.argmin(np.abs(arr - t_target)))

    fig_snap, axes_snap = plt.subplots(2, 3, figsize=(14, 7))

    # Row 1: Deltoid
    t_end_deltoid = viz_deltoid.time_history[-1] if viz_deltoid.time_history else 0.0
    times_deltoid = [0.0, 20.0, t_end_deltoid]
    for j, t_snap in enumerate(times_deltoid):
        f_d = time_to_frame(viz_deltoid, t_snap)
        viz_deltoid.render_snapshot(axes_snap[0, j], f_d, title=f"Deltoid t={t_snap:.0f}s")

    # Row 2: Lissajous
    t_end_liss = viz_liss.time_history[-1] if viz_liss.time_history else 0.0
    times_liss = [0.0, 20.0, t_end_liss]
    for j, t_snap in enumerate(times_liss):
        f_l = time_to_frame(viz_liss, t_snap)
        viz_liss.render_snapshot(axes_snap[1, j], f_l, title=f"Lissajous t={t_snap:.0f}s")

    fig_snap.suptitle('Simulation Snapshots: Deltoid (top) and Lissajous (bottom)', fontsize=14)
    fig_snap.tight_layout(rect=[0,0,1,0.96])
    fig_snap.savefig('simulation_snapshots.png', dpi=180, bbox_inches='tight')
    print("Saved snapshot figure to 'simulation_snapshots.png'")

    # --- Metrics Figure (for Lissajous curve) ---
    fig_metrics, axes_metrics = plt.subplots(3, 2, figsize=(14, 12))
    t_arr = np.array(viz_liss.time_history)

    def smooth(y, window=15):
        y = np.asarray(y)
        if y.size < window:
            return y
        w = np.ones(window) / window
        return np.convolve(y, w, mode='same')

    if len(t_arr) > 0:
        # (a) RMS formation error (semilog + smoothing)
        rms_raw = np.array(viz_liss.rms_error_history)
        rms_safe = np.maximum(rms_raw, 1e-4)
        rms_sm = smooth(rms_safe, window=25)
        axes_metrics[0,0].plot(t_arr, rms_safe, color='#1f77b4', lw=0.8, alpha=0.4, label='raw')
        axes_metrics[0,0].plot(t_arr, rms_sm, color='#1f77b4', lw=2.2, label='smoothed')
        axes_metrics[0,0].set_yscale('log')
        axes_metrics[0,0].set_ylabel('RMS error (m, log)')
        axes_metrics[0,0].set_title('(a) RMS Formation Error (log scale)')
        axes_metrics[0,0].grid(alpha=0.3, which='both')
        axes_metrics[0,0].legend(fontsize=8)

        # (b) Sigma statistics (mean ± std) with phase shading & faint individual lines
        sig_hist = np.array([h[1] for h in viz_liss.simulation_history])  # (frames, agents)
        mean_sig = np.array(viz_liss.sigma_mean_history)
        std_sig = np.array(viz_liss.sigma_std_history)
        for k in range(sig_hist.shape[1]):
            axes_metrics[0,1].plot(t_arr, sig_hist[:, k], lw=0.8, alpha=0.35, label=f'A{k+1}')
        axes_metrics[0,1].plot(t_arr, mean_sig, color='k', lw=2.0, label='mean σ')
        axes_metrics[0,1].fill_between(t_arr, mean_sig - std_sig, mean_sig + std_sig,
                                        color='gray', alpha=0.25, label='±1σ band')
        axes_metrics[0,1].set_ylim(-0.05, 1.05)
        axes_metrics[0,1].set_ylabel('σ')
        axes_metrics[0,1].set_title('(b) Blending σ Mean ± Std')
        axes_metrics[0,1].grid(alpha=0.3)

        # Phase shading: path-follow (<0.5), transition (0.5-0.9), stabilized (>0.9)
        def first_idx(y, thresh):
            idxs = np.where(y >= thresh)[0]
            return int(idxs[0]) if idxs.size else None
        idx_half = first_idx(mean_sig, 0.5)
        idx_full = first_idx(mean_sig, 0.9)
        t0 = t_arr[0]
        t_end = t_arr[-1]
        if idx_half is not None:
            axes_metrics[0,1].axvspan(t0, t_arr[idx_half], color='#ADD8E6', alpha=0.15, label='path-follow phase')
        if idx_half is not None and idx_full is not None:
            axes_metrics[0,1].axvspan(t_arr[idx_half], t_arr[idx_full], color='#FFFACD', alpha=0.20, label='transition')
        if idx_full is not None:
            axes_metrics[0,1].axvspan(t_arr[idx_full], t_end, color='#90EE90', alpha=0.18, label='stabilized')
        # Overlay thresholds
        axes_metrics[0,1].axhline(0.5, color='C1', ls='--', lw=1.0, alpha=0.6)
        axes_metrics[0,1].axhline(0.9, color='C2', ls='--', lw=1.0, alpha=0.6)
        axes_metrics[0,1].legend(fontsize=7, ncol=2, framealpha=0.6)

        # (c) Side length CoV
        axes_metrics[1,0].plot(t_arr, viz_liss.side_cov_history, color='#9467bd', lw=1.8)
        axes_metrics[1,0].set_ylabel('Side CoV')
        axes_metrics[1,0].set_title('(c) Side Length Coefficient of Variation')
        axes_metrics[1,0].grid(alpha=0.3)

        # (d) Minimum inter-agent distance
        min_d = np.array(viz_liss.min_dist_history)
        d_safe = 0.06 * viz_liss.curve.scale
        axes_metrics[1,1].plot(t_arr, min_d, color='g', lw=1.8, label='min distance')
        axes_metrics[1,1].axhline(d_safe, color='r', ls='--', lw=1.2, label='d_safe')
        axes_metrics[1,1].set_ylabel('Distance (m)')
        axes_metrics[1,1].set_title('(d) Minimum Inter-Agent Distance')
        axes_metrics[1,1].grid(alpha=0.3)
        axes_metrics[1,1].legend(fontsize=8, framealpha=0.5)

        # (e) Distance margin
        axes_metrics[2,0].plot(t_arr, viz_liss.distance_margin_history, color='#ff7f0e', lw=1.8)
        axes_metrics[2,0].axhline(0.0, color='k', ls='--', lw=1.0)
        axes_metrics[2,0].set_ylabel('Margin (m)')
        axes_metrics[2,0].set_xlabel('Time (s)')
        axes_metrics[2,0].set_title('(e) Distance Margin (min d - d_safe)')
        axes_metrics[2,0].grid(alpha=0.3)

        # (f) Curve adherence (mean projection error)
        axes_metrics[2,1].plot(t_arr, viz_liss.curve_adherence_history, color='#d62728', lw=1.8)
        axes_metrics[2,1].set_ylabel('Mean dist (m)')
        axes_metrics[2,1].set_xlabel('Time (s)')
        axes_metrics[2,1].set_title('(f) Curve Adherence Error')
        axes_metrics[2,1].grid(alpha=0.3)

    fig_metrics.suptitle('Lissajous Performance Metrics (Option B)', fontsize=14)
    fig_metrics.tight_layout(rect=[0,0,1,0.97])
    fig_metrics.savefig('simulation_plots.png', dpi=180, bbox_inches='tight')
    print("Saved enhanced metrics figure to 'simulation_plots.png'")

    print("\n" + "="*70)
    print("Creating animation...")
    print("="*70)
    
    # Find maximum number of frames
    max_frames = max(
        (len(v.simulation_history) for v in visualizers 
         if v.simulation_history),
        default=0
    )
    
    print(f"Total frames: {max_frames}")
    
    # Animation update function
    def update(frame):
        artists = []
        for viz in visualizers:
            artists.extend(viz.update_frame(frame))
        return artists
    
    # Create animation
    ani = FuncAnimation(
        fig, update, frames=range(max_frames),
        blit=False, interval=40, repeat=True
    )
    
    # Overall title
    fig.suptitle(
        'Multi-Agent Formation Control: Hybrid TFL + Pose Stabilization',
        fontsize=16, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save animation
    print("\nSaving animation...")
    try:
        # Save as GIF (optimized)
        frame_step = 2  # Skip every other frame for smaller file
        reduced_frames = range(0, max_frames, frame_step)
        ani_reduced = FuncAnimation(
            fig, update, frames=reduced_frames,
            blit=False, interval=40, repeat=True
        )
        ani_reduced.save(
            'multi_agent_formation.gif', 
            writer='pillow', 
            fps=25, 
            dpi=100,
            savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1}
        )
        print("✓ Saved animation to 'multi_agent_formation.gif'")
    except Exception as e:
        print(f"⚠ Could not save animation: {e}")
    
    print("\nDisplaying animation...")
    plt.show()


if __name__ == '__main__':
    main()
