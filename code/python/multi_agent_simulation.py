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


def deltoid(t):
    return (2*np.cos(t) + np.cos(2*t), 2*np.sin(t) - np.sin(2*t))


def rose(t, k=3, a=1.8):
    r = a * np.cos(k * t)
    return (r * np.cos(t), r * np.sin(t))


def cassini_oval(t, A=3.0, B=3.15):
    with np.errstate(invalid='ignore'):
        i = B**4 - A**4 * np.sin(2*t)**2
        r2 = A**2 * np.cos(2*t) + np.sqrt(np.clip(i, 0.0, np.inf))
        r = np.sqrt(np.clip(r2, 0.0, np.inf))
    return (r * np.cos(t), r * np.sin(t))


def lissajous(t, a=3, b=2, delta=0):
    return (np.sin(a * t + np.pi/2), np.sin(b * t + delta))


def gear(s, num_teeth=6, R_outer=2.3, R_inner=1.7):
    num_segments = 2 * num_teeth
    segment_angle = 2 * np.pi / num_segments
    s_norm = s % (2*np.pi)
    segment_idx = int(s_norm // segment_angle)
    s_start = segment_idx * segment_angle
    u = (s_norm - s_start) / segment_angle
    
    def smoothstep(t):
        return 3*t**2 - 2*t**3
    
    u_smooth = smoothstep(u)
    s_smooth = s_start + u_smooth * segment_angle
    radius = R_outer if segment_idx % 2 == 0 else R_inner
    return (radius * np.cos(s_smooth), radius * np.sin(s_smooth))


class MultiAgentVisualizer:    
    def __init__(self, curve, ax, target_thetas, curve_title="Curve", target_center=None):
        self.curve = curve
        self.ax = ax
        self.target_thetas = target_thetas
        self.curve_title = curve_title
        self.target_center = target_center
        self.simulation_history = []
        self.time_history = []
        self.rms_error_history = []
        self.min_dist_history = []
        self.side_cov_history = []
        self.distance_margin_history = []
        self.curve_adherence_history = []
        self.sigma_mean_history = []
        self.sigma_std_history = []
        
        self.agents = []
        self.controllers = []
        self.trail_len = 200
        self.agent_colors = list(plt.cm.viridis(np.linspace(0.15, 0.85, 4)))
        

        if target_thetas is None:
            print(f"  No target square found for {curve_title}")
            return
        
        self._initialize_agents_and_controllers()
    
    def _initialize_agents_and_controllers(self):
        target_points = points_from_thetas(self.curve.P, self.target_thetas)
        noise_radius = 0.18 * self.curve.scale
        
        for i in range(4):
            px, py = target_points[i]
            
            angle = np.random.uniform(0, 2*np.pi)
            x = px + noise_radius * np.cos(angle)
            y = py + noise_radius * np.sin(angle)
            theta = np.random.uniform(-np.pi, np.pi)
            
            agent = Unicycle(x, y, theta, v=0.0)
            agent.assigned_vertex = np.array([px, py])
            
            agent.prev_s = self.curve.get_closest_point(np.array([x, y]))
            agent.cum_s = agent.prev_s
            agent.s_revs = agent.cum_s / (2*np.pi)
            agent.prev_pos = np.array([x, y], dtype=float)
            
            self.agents.append(agent)
            
            s_dot_des = 0.1
            vz_cmd = 2.0 * s_dot_des
            tfl = SetpointLiftedTFLController(
                self.curve, 
                target_path_param=None,
                K_LIFT=2.0,
                vz_cmd=vz_cmd,
                v_min=0.05,
                k_heading_fb=3.0,
                kv_speed_fb=3.0
            )
            
            pose = PoseController(kp_pos=3.0, kv=4.0, k_theta=5.0, v_max=0.6)
            
            hybrid = HybridController(
                self.curve, tfl, pose,
                N_target=1.0,
                d_switch_frac=0.18,
                safety_radius_frac=0.07,
                avoid_on_frac=0.13,
                avoid_gain=0.75,
                v_avoid_frac=0.4,
                k_omega_avoid=3.0,
                k_v_avoid=3.0
            )
            
            self.controllers.append(hybrid)
        

        for agent in self.agents:
            agent._all_agents = self.agents
    
    def run_simulation(self, duration=120.0, dt=0.02):
        if not self.agents:
            return
        
        num_steps = int(duration / dt)
        t_now = 0.0
        log_interval = 20.0
        steps_per_log = max(1, int(round(log_interval / dt)))
        
        for step in range(num_steps):
            current_states = [agent.state.copy() for agent in self.agents]
            

            for i, agent in enumerate(self.agents):
                x, y = agent.state[0], agent.state[1]
                pos = np.array([x, y])

                s_curr = self.curve.get_closest_point(pos)

                if agent.prev_s is not None and agent.prev_pos is not None:
                    dp_ds, _ = self.curve.derivatives(agent.prev_s)
                    dp_norm_sq = float(dp_ds[0]**2 + dp_ds[1]**2)
                    if dp_norm_sq > 1e-10:
                        delta_p = pos - agent.prev_pos
                        ds_est = float(np.dot(delta_p, dp_ds) / dp_norm_sq)
                        ds_est = float(np.clip(ds_est, -0.2, 0.2))
                    else:
                        ds_est = 0.0
                    agent.cum_s += ds_est
                else:
                    ds = s_curr - (agent.prev_s or s_curr)
                    if ds < -np.pi:
                        ds += 2*np.pi
                    elif ds > np.pi:
                        ds -= 2*np.pi
                    agent.cum_s += ds

                agent.prev_s = s_curr
                agent.prev_pos = pos
                agent.s_revs = agent.cum_s / (2*np.pi)
                

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
            
            controls = []
            for i, agent in enumerate(self.agents):
                u, sigma = self.controllers[i].compute_controls(agent, dt)
                controls.append(u)
            
            for agent, u in zip(self.agents, controls):
                agent.update(u, dt)
            

            sigmas = [agent.sigma for agent in self.agents]
            self.simulation_history.append((current_states, sigmas))
            self.time_history.append(t_now)
            sigma_mean = float(np.mean(sigmas))
            sigma_std = float(np.std(sigmas))
            self.sigma_mean_history.append(sigma_mean)
            self.sigma_std_history.append(sigma_std)

            positions = np.array([s[:2] for s in current_states])
            vertices = np.array([a.assigned_vertex for a in self.agents])
            if vertices.shape == positions.shape:
                errs = np.linalg.norm(positions - vertices, axis=1)
                rms_err = float(np.sqrt(np.mean(errs**2)))
            else:
                rms_err = np.nan
            self.rms_error_history.append(rms_err)


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

            adher_dists = []
            for p in positions:
                s_proj = self.curve.get_closest_point(p)
                proj = np.array(self.curve.P(s_proj))
                adher_dists.append(np.linalg.norm(p - proj))
            self.curve_adherence_history.append(float(np.mean(adher_dists)))

            d_safe = 0.03 * self.curve.scale
            if len(positions) >= 2 and not np.isnan(self.min_dist_history[-1]):
                self.distance_margin_history.append(self.min_dist_history[-1] - d_safe)
            else:
                self.distance_margin_history.append(np.nan)


            if step % steps_per_log == 0:
                s_revs_list = [f"{a.s_revs:.2f}" for a in self.agents]
                sig_list = [f"{a.sigma:.2f}" for a in self.agents]
                print(f"[{self.curve_title} t={t_now:6.1f}s] s_revs=" + ", ".join(s_revs_list) +
                      " | sigma=" + ", ".join(sig_list))

            t_now += dt
    
    def setup_plot(self, title=""):
        self.ax.set_title(title, fontsize=14)
        
        if self.target_thetas is None:
            self.ax.text(0.5, 0.5, 'No solution', 
                        transform=self.ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            return
        

        t = np.linspace(0, 2*np.pi, 400)
        x, y = np.vectorize(self.curve.P)(t)
        self.ax.plot(x, y, color='#2ca02c', linestyle='--', 
                    lw=1.0, alpha=0.9, solid_capstyle='round')
        
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
        
        if self.target_center is not None:
            self.ax.plot(self.target_center[0], self.target_center[1], 
                        'rx', markersize=10, markeredgewidth=2.5, label='POI')
            self.ax.legend(loc='upper right', fontsize=8)

        self.agent_plots = []
        for i in range(4):
            plot, = self.ax.plot([], [], 'o', markersize=12, 
                               color=self.agent_colors[i],
                               markeredgecolor='k', markeredgewidth=1.5,
                               zorder=10)
            self.agent_plots.append(plot)
        
        self.trail_collections = []
        for i in range(4):
            lc = LineCollection([], linewidths=2.5, 
                              colors=self.agent_colors[i],
                              alpha=0.6, zorder=5)
            self.ax.add_collection(lc)
            self.trail_collections.append(lc)
        
        self.edge_plot, = self.ax.plot([], [], '--', linewidth=1.5, 
                                       color='blue', alpha=0.6, zorder=4)
        
        self.sigma_texts = [
            self.ax.text(0, 0, "", fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.75, 
                                boxstyle='round,pad=0.2'))
            for _ in range(4)
        ]
        

        self.ax.set_xlim(self.curve.xlim[0]*1.2, self.curve.xlim[1]*1.2)
        self.ax.set_ylim(self.curve.ylim[0]*1.2, self.curve.ylim[1]*1.2)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
    
    def update_frame(self, frame):
        if not self.simulation_history or frame >= len(self.simulation_history):
            return []
        
        states, sigmas = self.simulation_history[frame]
        points = np.array([s[:2] for s in states])
        start = max(0, frame - self.trail_len)
        

        for i, plot in enumerate(self.agent_plots):
            plot.set_data([points[i, 0]], [points[i, 1]])
            
            if frame > 0:
                trail_states = [self.simulation_history[f][0][i][:2] 
                              for f in range(start, frame + 1)]
                trail_points = np.array(trail_states)
                
                segments = []
                for j in range(len(trail_points) - 1):
                    segments.append([trail_points[j], trail_points[j+1]])
                
                if segments:
                    n_seg = len(segments)
                    alphas = np.linspace(0.1, 0.8, n_seg)
                    
                    self.trail_collections[i].set_segments(segments)
                    self.trail_collections[i].set_alpha(alphas[-1] if n_seg > 0 else 0.6)
            

            sigma_val = sigmas[i]
            offset = 0.15 * self.curve.scale
            self.sigma_texts[i].set_position((points[i, 0] + offset, 
                                             points[i, 1] + offset))
            self.sigma_texts[i].set_text(f'σ={sigma_val:.2f}')
            
            if sigma_val > 0.9:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('#90EE90')
            elif sigma_val > 0.5:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('#FFFACD')
            else:
                self.sigma_texts[i].get_bbox_patch().set_facecolor('white')
        
        closed_points = np.vstack([points, points[0]])
        self.edge_plot.set_data(closed_points[:, 0], closed_points[:, 1])
        
        artists = []
        artists.extend(self.agent_plots)
        artists.extend(self.trail_collections)
        artists.append(self.edge_plot)
        artists.extend(self.sigma_texts)
        
        return artists

    def render_snapshot(self, ax, frame, title=""):

        ax.clear()
        ax.set_title(title, fontsize=12)

        t = np.linspace(0, 2*np.pi, 400)
        x, y = np.vectorize(self.curve.P)(t)
        ax.plot(x, y, color='#2ca02c', linestyle='--', lw=1.0, alpha=0.9)

        if self.target_thetas is not None:
            target_points = points_from_thetas(self.curve.P, self.target_thetas)
            closed = np.vstack([target_points, target_points[0]])
            ax.plot(closed[:,0], closed[:,1], 'k-', lw=1.5, alpha=0.4)
            ax.scatter(target_points[:,0], target_points[:,1], c='k', s=20, alpha=0.6)

        if self.target_center is not None:
            ax.plot(self.target_center[0], self.target_center[1], 
                    'rx', markersize=10, markeredgewidth=2.5)

        if not self.simulation_history:
            return
        frame = max(0, min(frame, len(self.simulation_history)-1))
        states, sigmas = self.simulation_history[frame]
        points = np.array([s[:2] for s in states])

        for i in range(len(self.agents)):
            trail_states = [self.simulation_history[f][0][i][:2] for f in range(0, frame+1)]
            trail_points = np.array(trail_states)
            ax.plot(trail_points[:,0], trail_points[:,1], '-', color=self.agent_colors[i], alpha=0.5, lw=1.5)

        for i in range(len(self.agents)):
            ax.plot(points[i,0], points[i,1], 'o', color=self.agent_colors[i], markeredgecolor='k', markersize=8)
            ax.text(points[i,0], points[i,1], f"  σ={sigmas[i]:.2f}", fontsize=8)

        closed_pts = np.vstack([points, points[0]])
        ax.plot(closed_pts[:,0], closed_pts[:,1], '--', color='blue', alpha=0.6)

        ax.set_xlim(self.curve.xlim[0]*1.2, self.curve.xlim[1]*1.2)
        ax.set_ylim(self.curve.ylim[0]*1.2, self.curve.ylim[1]*1.2)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([]); ax.set_yticks([])


def main():
    np.random.seed(123)
    

    curve_definitions = [
        {
            'func': deltoid,
            'title': 'Deltoid',
            'mode': '4D',
            'min_side_fraction': 0.05,
            'target_center': [0.0, 0.0]
        },
        {
            'func': lambda t: lissajous(t, a=3, b=2, delta=0.0),
            'title': 'Lissajous (3:2)',
            'mode': '4D',
            'min_side_fraction': 0.05,
            'target_center': [0.5, 0.5]
        },
    ]
    
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
        
        curve = Curve(definition['func'], num_points=2000)
        print(f"  Curve scale: {curve.scale:.3f}")
        
        target_center = definition.get('target_center')
        finder = RigidFormationFinder(
            curve, 
            num_vertices=4,
            min_side_fraction=definition.get('min_side_fraction', 0.05),
            target_center=target_center
        )
        
        initial_guess = definition.get('guess')
        if initial_guess:
            target_thetas, error, _ = finder.find_ngon(
                initial_guesses=[initial_guess]
            )
        else:
            target_thetas, error, _ = finder.find_ngon(num_random_inits=50)
        
        if target_thetas is not None:
            pts = points_from_thetas(curve.P, target_thetas)
            sides = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]
            center = np.mean(pts, axis=0)
            print(f"Square found")
            print(f"Center: {center}")
            print(f"Side lengths: {[f'{s:.4f}' for s in sides]}")
            print(f"Mean: {np.mean(sides):.4f}, Std: {np.std(sides):.4e}")
        else:
            print(f"  ✗ No square found")
        
        viz = MultiAgentVisualizer(
            curve, axes[i], target_thetas, 
            curve_title=definition['title'],
            target_center=target_center
        )
        
        if target_thetas is not None:
            print(f"  Running simulation...")
            viz.run_simulation(duration=180.0, dt=0.03)
            print(f"  Simulation complete: {len(viz.simulation_history)} frames")

            final_states, final_sigmas = viz.simulation_history[-1]
            final_positions = np.array([s[:2] for s in final_states])
            print("    Final agent diagnostics:")
            for ai, (pos, sigma_val) in enumerate(zip(final_positions, final_sigmas), start=1):
                dist_vert = np.linalg.norm(pos - viz.agents[ai-1].assigned_vertex)
                print(f"      Agent {ai}: dist_to_vertex={dist_vert:.4f} sigma={sigma_val:.3f}")
        
        viz.setup_plot(title=definition['title'])
        visualizers.append(viz)
    
    if len(visualizers) == 2:
        viz_deltoid = visualizers[0]
        viz_liss = visualizers[1]
    else:
        viz_deltoid = visualizers[0]
        viz_liss = visualizers[-1]


    def time_to_frame(viz, t_target):
        if not viz.time_history:
            return 0
        arr = np.array(viz.time_history)
        return int(np.argmin(np.abs(arr - t_target)))

    fig_snap, axes_snap = plt.subplots(2, 3, figsize=(14, 7))

    t_end_deltoid = viz_deltoid.time_history[-1] if viz_deltoid.time_history else 0.0
    times_deltoid = [0.0, 20.0, t_end_deltoid]
    for j, t_snap in enumerate(times_deltoid):
        f_d = time_to_frame(viz_deltoid, t_snap)
        viz_deltoid.render_snapshot(axes_snap[0, j], f_d, title=f"Deltoid t={t_snap:.0f}s")

    t_end_liss = viz_liss.time_history[-1] if viz_liss.time_history else 0.0
    times_liss = [0.0, 20.0, t_end_liss]
    for j, t_snap in enumerate(times_liss):
        f_l = time_to_frame(viz_liss, t_snap)
        viz_liss.render_snapshot(axes_snap[1, j], f_l, title=f"Lissajous t={t_snap:.0f}s")

    fig_snap.suptitle('Simulation Snapshots: Deltoid (top) and Lissajous (bottom)', fontsize=14)
    fig_snap.tight_layout(rect=[0,0,1,0.96])
    fig_snap.savefig('simulation_snapshots.png', dpi=180, bbox_inches='tight')
    print("Saved snapshot figure to 'simulation_snapshots.png'")

    fig_metrics, axes_metrics = plt.subplots(2, 1, figsize=(10, 8))
    t_arr = np.array(viz_liss.time_history)

    def smooth(y, window=15):
        y = np.asarray(y)
        if y.size < window:
            return y
        w = np.ones(window) / window
        return np.convolve(y, w, mode='same')

    if len(t_arr) > 0:
        min_d = np.array(viz_liss.min_dist_history)
        d_safe = 0.02 * viz_liss.curve.scale
        axes_metrics[0].plot(t_arr, min_d, color='#2ca02c', lw=2.5, label='Minimum distance')
        axes_metrics[0].axhline(d_safe, color='#d62728', ls='--', lw=2.0, label=r'$d_{\mathrm{safe}}$')
        axes_metrics[0].set_ylabel('Distance (m)', fontsize=14, fontweight='bold')
        axes_metrics[0].set_title('(a) Minimum Inter-Agent Distance', fontsize=16, fontweight='bold')
        axes_metrics[0].grid(alpha=0.3, linewidth=1.2)
        axes_metrics[0].legend(fontsize=12, framealpha=0.9, loc='best')
        axes_metrics[0].tick_params(axis='both', labelsize=12)
        
        violation_mask = min_d < d_safe
        if np.any(violation_mask):
            axes_metrics[0].fill_between(t_arr, 0, d_safe, where=violation_mask, 
                                         color='red', alpha=0.15, label='Safety violations')


        axes_metrics[1].plot(t_arr, viz_liss.curve_adherence_history, color='#1f77b4', lw=2.5)
        axes_metrics[1].set_ylabel('Mean distance (m)', fontsize=14, fontweight='bold')
        axes_metrics[1].set_xlabel('Time (s)', fontsize=14, fontweight='bold')
        axes_metrics[1].set_title('(b) Curve Adherence Error', fontsize=16, fontweight='bold')
        axes_metrics[1].grid(alpha=0.3, linewidth=1.2)
        axes_metrics[1].tick_params(axis='both', labelsize=12)

    fig_metrics.suptitle('Lissajous Formation Control: Key Performance Metrics', fontsize=18, fontweight='bold')
    fig_metrics.tight_layout(rect=[0, 0, 1, 0.96])
    fig_metrics.savefig('simulation_plots.png', dpi=200, bbox_inches='tight')
    print("Saved simplified metrics figure to 'simulation_plots.png'")

    print("\n" + "="*70)
    print("Creating animation...")
    print("="*70)
    
    max_frames = max(
        (len(v.simulation_history) for v in visualizers 
         if v.simulation_history),
        default=0
    )
    
    print(f"Total frames: {max_frames}")
    
    def update(frame):
        artists = []
        for viz in visualizers:
            artists.extend(viz.update_frame(frame))
        return artists
    
    ani = FuncAnimation(
        fig, update, frames=range(max_frames),
        blit=False, interval=40, repeat=True
    )
    
    fig.suptitle(
        'Multi-Agent Formation Control: Hybrid TFL + Pose Stabilization',
        fontsize=16, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    

    print("\nSaving animation...")
    try:
        frame_step = 2
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
