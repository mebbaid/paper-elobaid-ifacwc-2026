"""
Controllers for path following and formation stabilization.

Includes:
- SetpointLiftedTFLController: Transverse feedback linearization with internal dynamics
- PoseController: Point stabilization for vertex positioning
- HybridController: Smooth blending between TFL and pose control
"""

import numpy as np
import math


def smoothstep(u):
    """
    Cubic smoothstep function with C¹ continuity.
    
    Parameters
    ----------
    u : float
        Input value (clipped to [0, 1])
    
    Returns
    -------
    float
        Smooth interpolation: 0 at u=0, 1 at u=1, with zero derivatives at endpoints
    """
    u = np.clip(u, 0.0, 1.0)
    return 3 * u**2 - 2 * u**3


class SetpointLiftedTFLController:
    """
    Setpoint-tracking lifted transverse feedback linearization (TFL) controller.
    
    Augments the unicycle state with an internal variable z that tracks progress
    along the curve. The controller:
    1. Tracks a lifted reference trajectory (h₁(z), h₂(z), z)
    2. Allows commanded z-velocity for path progression
    3. Uses PD control in the lifted coordinates
    
    References:
    - Nielsen et al., "Path following using transverse feedback linearization"
    """
    
    def __init__(self, curve, target_path_param=None, K_LIFT=2.0, vz_cmd=0.0,
                 v_min=-0.6, k_heading_fb=3.0, kv_speed_fb=2.0):
        """
        Initialize TFL controller.
        
        Parameters
        ----------
        curve : Curve
            Parametric curve to follow
        target_path_param : float, optional
            Target path parameter (if fixed setpoint desired)
        K_LIFT : float
            Lifting gain: z = K_LIFT * s (where s is path parameter)
        vz_cmd : float
            Commanded z-velocity (drives progress along curve)
        v_min : float
            Minimum fallback velocity when v ≈ 0
        k_heading_fb : float
            Heading feedback gain for fallback mode
        kv_speed_fb : float
            Speed feedback gain for fallback mode
        """
        self.curve = curve
        self.target_lambda = target_path_param
        self.K_LIFT = K_LIFT
        self.vz_cmd = float(vz_cmd)
        self.v_min = float(v_min)
        self.k_heading_fb = float(k_heading_fb)
        self.kv_speed_fb = float(kv_speed_fb)
        
        # Target z (if setpoint tracking)
        self.z_target = self.K_LIFT * self.target_lambda if self.target_lambda is not None else 0.0
        
        # PD gains for lifted coordinates
        kp1, kp2, kp_z = 15.0, 15.0, 3.0
        kd1, kd2, kd_z = 10.0, 10.0, 3.0
        self.Kp = np.diag([kp1, kp2, kp_z])
        self.Kd = np.diag([kd1, kd2, kd_z])
        
        # Internal state: z and v_z
        self.z = 0.0
        self.v_z = 0.0
        
        if self.target_lambda is not None:
            self.z = self.target_lambda * self.K_LIFT
        
        # Saturation limits for internal dynamics
        self.max_az = 20.0
        self.max_vz = 10.0
    
    def compute_controls(self, state, dt):
        """
        Compute control inputs [a, ω].
        
        Parameters
        ----------
        state : ndarray, shape (4,)
            Current state [x, y, θ, v]
        dt : float
            Time step
        
        Returns
        -------
        ndarray, shape (2,)
            Control inputs [a, ω]
        """
        x, y, theta, v = state
        
        # Find closest point on curve (for transverse error)
        current_s = self.curve.get_closest_point(np.array([x, y]))
        target_pos = np.array(self.curve.P(current_s))
        
        # Curve derivatives at current_s
        (dx_ds, dy_ds), (d2x_ds2, d2y_ds2) = self.curve.derivatives(current_s)
        
        # Convert to z-derivatives: dh/dz = (dh/ds) / K_LIFT
        dh1_dz = dx_ds / self.K_LIFT
        dh2_dz = dy_ds / self.K_LIFT
        d2h1_dz2 = d2x_ds2 / (self.K_LIFT**2)
        d2h2_dz2 = d2y_ds2 / (self.K_LIFT**2)
        
        # Advance z_target if cruise velocity commanded
        if abs(self.vz_cmd) > 1e-6:
            self.z_target += self.vz_cmd * dt
        
        # Error vectors in lifted coordinates
        y_vec = np.array([
            x - target_pos[0],
            y - target_pos[1],
            self.z - self.z_target
        ])
        
        ydot_vec = np.array([
            v * np.cos(theta) - dh1_dz * self.v_z,
            v * np.sin(theta) - dh2_dz * self.v_z,
            self.v_z
        ])
        
        # Nonlinear compensation term
        F_vec = np.array([
            -d2h1_dz2 * (self.v_z**2),
            -d2h2_dz2 * (self.v_z**2),
            0.0
        ])
        
        # Decoupling matrix
        D_mat = np.array([
            [np.cos(theta), -v * np.sin(theta), -dh1_dz],
            [np.sin(theta),  v * np.cos(theta), -dh2_dz],
            [0.0,            0.0,                1.0]
        ])
        
        # Virtual control
        nu_vec = -self.Kp @ y_vec - self.Kd @ ydot_vec
        
        # --- Fallback for low velocity ---
        v_eps = 1e-2
        if abs(v) < v_eps:
            # Align with curve tangent and accelerate
            direction_sign = 1.0 if self.vz_cmd >= 0.0 else -1.0
            desired_heading = math.atan2(direction_sign * dy_ds, direction_sign * dx_ds)
            heading_error = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
            
            v_des = self.v_min
            a_safe = np.clip(self.kv_speed_fb * (v_des - v), -self.max_vz, self.max_vz)
            omega_safe = np.clip(self.k_heading_fb * heading_error, -4.0, 4.0)
            
            # Still drive internal z forward if commanded
            if abs(self.vz_cmd) > 1e-6:
                a_z = self.vz_cmd - self.v_z
                a_z = np.clip(a_z, -self.max_az, self.max_az)
                self.v_z = np.clip(self.v_z + a_z * dt, -self.max_vz, self.max_vz)
                self.z += self.v_z * dt
            
            return np.array([a_safe, omega_safe])
        
        # --- Normal mode: invert decoupling matrix ---
        try:
            u_aug = np.linalg.solve(D_mat, (nu_vec - F_vec))
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0])
        
        a_aug, omega_aug, a_z = u_aug
        
        # Saturate internal acceleration and update internal state
        a_z = np.clip(a_z, -self.max_az, self.max_az)
        self.v_z = np.clip(self.v_z + a_z * dt, -self.max_vz, self.max_vz)
        self.z += self.v_z * dt
        
        # Clip final outputs to reasonable bounds
        a_aug = float(np.clip(a_aug, -10.0, 10.0))
        omega_aug = float(np.clip(omega_aug, -4.0, 4.0))
        
        return np.array([a_aug, omega_aug])


class PoseController:
    """
    Simple pose stabilization controller.
    
    Drives the robot to a target position [x*, y*] using:
    - Proportional control on position error
    - Heading alignment toward target
    - Velocity damping
    """
    
    def __init__(self, kp_pos=3.0, kv=3.0, k_theta=4.0, v_max=1.0, allow_reverse_heading=True):
        """
        Initialize pose controller.
        
        Parameters
        ----------
        kp_pos : float
            Position error gain
        kv : float
            Velocity error gain
        k_theta : float
            Heading error gain
        v_max : float
            Maximum approach velocity
        allow_reverse_heading : bool
            If True, allow stabilizing to the vertex with heading flipped by 180° when beneficial
        """
        self.kp_pos = kp_pos
        self.kv = kv
        self.k_theta = k_theta
        self.v_max = v_max
        # When True, the controller can stabilize to the vertex while facing either direction
        # (desired heading or desired heading + pi), choosing the one requiring less rotation,
        # and will drive backwards if needed to approach the vertex.
        self.allow_reverse_heading = bool(allow_reverse_heading)
        self.max_acc = 15.0
        self.max_omega = 10.0
    
    def compute_controls(self, state, target_pos, dt, sigma=0.0):
        """
        Compute control inputs to reach target position.
        
        Parameters
        ----------
        state : ndarray, shape (4,)
            Current state [x, y, θ, v]
        target_pos : array_like, shape (2,)
            Target position [x*, y*]
        dt : float
            Time step (unused, kept for API compatibility)
        sigma : float
            Blending parameter (used for gain scheduling: stronger control as σ→1)
        
        Returns
        -------
        ndarray, shape (2,)
            Control inputs [a, ω]
        """
        x, y, theta, v = state
        ex = target_pos[0] - x
        ey = target_pos[1] - y
        dist = np.hypot(ex, ey)
        
        # Desired heading toward target
        desired_heading = math.atan2(ey, ex)
        # Forward option: face the target
        e_fwd = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
        # Reverse option: face away from the target (180° rotated)
        e_rev = np.mod(desired_heading + np.pi - theta + np.pi, 2*np.pi) - np.pi
        # Gate reverse usage: only when sufficiently blended (sigma>0.5) and close to vertex
        use_reverse = False
        if self.allow_reverse_heading and sigma > 0.5 and dist < 0.8 * self.v_max:
            # Require meaningful reduction in absolute rotation (> 0.1 rad margin)
            if abs(e_rev) + 0.10 < abs(e_fwd):
                use_reverse = True
        heading_error = e_rev if use_reverse else e_fwd
        
        # Gain scheduling with blending parameter
        sigma = float(np.clip(sigma, 0.0, 1.0))
        kp_eff = self.kp_pos * (1.0 + 2.5 * sigma)
        kv_eff = self.kv * (1.0 + 1.5 * sigma)
        k_theta_eff = self.k_theta * (1.0 + 3.0 * sigma)
        v_max_eff = self.v_max
        
        # Desired signed speed (forward by default). If using reverse heading, allow backward motion.
        v_des_base = v_max_eff * np.tanh(kp_eff * dist)
        v_des = v_des_base * np.cos(heading_error)
        if use_reverse:
            v_des = -v_des  # invert direction when reversed
        
        # PD acceleration
        a = kv_eff * (v_des - v)
        
        # Angular velocity to align heading
        omega = k_theta_eff * heading_error
        
        # Saturate
        a = float(np.clip(a, -self.max_acc, self.max_acc))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        
        return np.array([a, omega])


class HybridController:
    """
    Hybrid controller blending TFL and pose control.
    
    Uses smooth cubic blending based on:
    - Number of revolutions around the curve (σ_N)
    - Distance to assigned vertex (σ_D)
    - Combined blending: σ = σ_N * σ_D
    
    Also includes collision avoidance among agents.
    """
    
    def __init__(self, curve, tfl_controller, pose_controller,
                 N_target=2.0, d_switch_frac=0.12, d_margin_frac=0.3,
                 safety_radius_frac=0.06, avoid_on_frac=0.10,
                 avoid_gain=1.0, v_avoid_frac=0.5,
                 k_omega_avoid=4.0, k_v_avoid=4.0,
                 # Forward progress helpers
                 min_cruise_speed_frac=0.05, sigma_cruise_off=0.3, k_cruise=2.0):
        """
        Initialize hybrid controller.
        
        Parameters
        ----------
        curve : Curve
            Parametric curve
        tfl_controller : SetpointLiftedTFLController
            TFL controller instance
        pose_controller : PoseController
            Pose controller instance
        N_target : float
            Number of target revolutions before full blending
        d_switch_frac : float
            Distance threshold (as fraction of curve scale) for blending
        d_margin_frac : float
            Margin for smoothstep normalization
        safety_radius_frac : float
            Hard safety radius for collision avoidance (fraction of curve scale)
        avoid_on_frac : float
            Distance where avoidance begins (fraction of curve scale)
        avoid_gain : float
            Avoidance force gain
        v_avoid_frac : float
            Avoidance velocity as fraction of max velocity
        k_omega_avoid : float
            Angular velocity gain for avoidance
        k_v_avoid : float
            Linear velocity gain for avoidance
        """
        self.curve = curve
        self.tfl = tfl_controller
        self.pose = pose_controller
        self.N_target = float(N_target)
        
        # Blending parameters
        self.d_switch = d_switch_frac * curve.scale
        self.d_margin = d_margin_frac * curve.scale
        
        # Collision avoidance parameters
        self.safety_radius = safety_radius_frac * curve.scale
        self.avoid_on_dist = avoid_on_frac * curve.scale
        self.avoid_gain = float(avoid_gain)
        self.v_avoid = float(v_avoid_frac)
        self.k_omega_avoid = float(k_omega_avoid)
        self.k_v_avoid = float(k_v_avoid)
        # Forward progress helpers
        self.min_cruise_speed_frac = float(min_cruise_speed_frac)
        self.sigma_cruise_off = float(sigma_cruise_off)
        self.k_cruise = float(k_cruise)
    
    def blending_sigma(self, revs, dist):
        """
        Compute blending parameter σ ∈ [0, 1].
        
        σ = 0: TFL control (path following)
        σ = 1: Pose control (vertex stabilization)
        
        Parameters
        ----------
        revs : float
            Number of completed revolutions
        dist : float
            Distance to assigned vertex
        
        Returns
        -------
        float
            Blending parameter σ ∈ [0, 1]
        """
        # Revolution-based component
        r = np.clip(revs / self.N_target, 0.0, 1.0)
        sN = smoothstep(r)
        
        # Distance-based component
        d_norm = np.clip(dist / self.d_switch, 0.0, 1.0)
        sD = 1.0 - smoothstep(d_norm)
        
        return float(sN * sD)
    
    def compute_controls(self, agent, dt):
        """
        Compute hybrid control with avoidance.
        
        Parameters
        ----------
        agent : Unicycle
            Agent to control (must have assigned_vertex and _all_agents attributes)
        dt : float
            Time step
        
        Returns
        -------
        u_final : ndarray, shape (2,)
            Final control [a, ω]
        sigma : float
            Current blending parameter
        """
        state = agent.state.copy()
        x, y = state[0], state[1]
        v_cur = state[3]
        
        # --- Stage 1: Nominal controllers ---
        u_tfl = self.tfl.compute_controls(state, dt)
        
        if agent.assigned_vertex is None:
            # No target vertex: default to TFL only
            return u_tfl, 0.0
        
        sigma = agent.sigma
        u_pose = self.pose.compute_controls(state, agent.assigned_vertex, dt, sigma=sigma)
        u_nominal = (1.0 - sigma) * u_tfl + sigma * u_pose

        # Ensure forward progress in early phase: maintain a minimum cruise speed
        if sigma < self.sigma_cruise_off:
            v_des_min = self.min_cruise_speed_frac * agent.max_velocity
            a_boost = self.k_cruise * (v_des_min - v_cur)
            if a_boost > 0:  # only boost if below desired minimum
                u_nominal = u_nominal.copy()
                u_nominal[0] = np.clip(u_nominal[0] + a_boost,
                                       agent.min_acceleration, agent.max_acceleration)
        
        # --- Stage 2: Avoidance logic ---
        agents_list = getattr(agent, "_all_agents", None)
        if agents_list is None:
            return u_nominal, sigma
        
        u_avoid, alpha = self._compute_avoidance(agent, agents_list, dt)
        
        # --- Stage 3: Final blend ---
        u_final = (1.0 - alpha) * u_nominal + alpha * u_avoid
        return u_final, sigma
    
    def _compute_avoidance(self, agent, agents, dt):
        """
        Compute collision avoidance control.
        
        Parameters
        ----------
        agent : Unicycle
            Current agent
        agents : list of Unicycle
            All agents (including current)
        dt : float
            Time step
        
        Returns
        -------
        u_avoid : ndarray, shape (2,)
            Avoidance control
        alpha : float
            Avoidance blending weight
        """
        p_i = agent.state[:2]
        theta = agent.state[2]
        v_cur = agent.state[3]
        
        # Blending parameters for avoidance activation
        sigma_shield = 0.6
        delta_sigma = 0.2
        sigma_accept = 0.6
        delta_accept = 0.2
        
        def smoothstep_clip(u):
            u = float(np.clip(u, 0.0, 1.0))
            return 3 * u * u - 2 * u * u * u
        
        def a_of_sigma(sigma_i):
            """Avoidance activation: ramp up as agent settles (low when sigma is small)."""
            # Active primarily when sigma exceeds acceptance threshold; near zero otherwise
            return smoothstep_clip((sigma_i - sigma_accept) / max(1e-6, delta_accept))
        
        # Accumulate repulsive forces
        f = np.zeros(2)
        alpha_local = 0.0
        
        for other in agents:
            if other is agent:
                continue
            
            p_j = other.state[:2]
            diff = p_i - p_j
            dist = np.linalg.norm(diff)
            
            if dist <= 1e-8:
                continue
            
            # Check if this agent should avoid (based on its sigma)
            sigma_i = agent.sigma
            ai = a_of_sigma(sigma_i)
            
            # Compute blending for this pair
            t = np.clip(
                (self.avoid_on_dist - dist) / max(1e-6, (self.avoid_on_dist - self.safety_radius)),
                0.0, 1.0
            )
            pair_alpha = smoothstep_clip(t)
            
            if dist < self.avoid_on_dist:
                # Repulsive force
                force_mag = self.avoid_gain * ai * pair_alpha / max(dist, 1e-3)
                f += force_mag * (diff / dist)
                alpha_local = max(alpha_local, ai * pair_alpha)
        
        # If no avoidance needed
        norm_f = np.linalg.norm(f)
        if alpha_local < 0.01 or norm_f < 1e-6:
            return np.array([0.0, 0.0]), 0.0
        
        # Desired avoidance direction
        v_des_avoid = self.v_avoid * agent.max_velocity
        desired_dir_vec = f / norm_f
        
        # Handle stuck agents: rotate perpendicular
        is_stuck = alpha_local > 0.2 and norm_f < 0.1 and abs(v_cur) < 0.05
        if is_stuck:
            desired_dir_vec = np.array([-math.sin(theta), math.cos(theta)])
        
        # Compute desired heading and control
        desired_heading = math.atan2(desired_dir_vec[1], desired_dir_vec[0])
        heading_error = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
        
        v_des_signed = v_des_avoid * np.cos(heading_error)
        
        a_avoid = self.k_v_avoid * (v_des_signed - v_cur)
        omega_avoid = self.k_omega_avoid * heading_error
        
        # Saturate
        a_avoid = np.clip(a_avoid, agent.min_acceleration, agent.max_acceleration)
        omega_avoid = np.clip(omega_avoid, -agent.max_angular_velocity, agent.max_angular_velocity)
        
        alpha_eff = float(np.clip(alpha_local, 0.0, 1.0))
        
        return np.array([a_avoid, omega_avoid]), alpha_eff
