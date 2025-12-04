import numpy as np
import math


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3 * u**2 - 2 * u**3


class SetpointLiftedTFLController:
    
    def __init__(self, curve, target_path_param=None, K_LIFT=2.0, vz_cmd=0.0,
                 v_min=-0.6, k_heading_fb=3.0, kv_speed_fb=2.0):
        self.curve = curve
        self.target_lambda = target_path_param
        self.K_LIFT = K_LIFT
        self.vz_cmd = float(vz_cmd)
        self.v_min = float(v_min)
        self.k_heading_fb = float(k_heading_fb)
        self.kv_speed_fb = float(kv_speed_fb)
        
        self.z_target = self.K_LIFT * self.target_lambda if self.target_lambda is not None else 0.0
        
        kp1, kp2, kp_z = 15.0, 15.0, 3.0
        kd1, kd2, kd_z = 10.0, 10.0, 3.0
        self.Kp = np.diag([kp1, kp2, kp_z])
        self.Kd = np.diag([kd1, kd2, kd_z])
        
        self.z = 0.0
        self.v_z = 0.0
        self.initialized = False
        
        if self.target_lambda is not None:
            self.z = self.target_lambda * self.K_LIFT
            self.initialized = True
        
        self.max_az = 20.0
        self.max_vz = 10.0
    
    def compute_controls(self, state, dt):
        x, y, theta, v = state
        
        vz_cmd_eff = self.vz_cmd

        current_s = self.curve.get_closest_point(np.array([x, y]))
        target_pos = np.array(self.curve.P(current_s))
        
        (dx_ds, dy_ds), (d2x_ds2, d2y_ds2) = self.curve.derivatives(current_s)
        
        dh1_dz = dx_ds / self.K_LIFT
        dh2_dz = dy_ds / self.K_LIFT
        d2h1_dz2 = d2x_ds2 / (self.K_LIFT**2)
        d2h2_dz2 = d2y_ds2 / (self.K_LIFT**2)
        
        if abs(self.vz_cmd) > 1e-6:
            self.z_target += vz_cmd_eff * dt
        
        y_vec = np.array([
            x - target_pos[0],
            y - target_pos[1],
            self.z - self.z_target
        ])
        
        ydot_vec = np.array([
            v * np.cos(theta) - dh1_dz * self.v_z,
            v * np.sin(theta) - dh2_dz * self.v_z,
            self.v_z - vz_cmd_eff
        ])
        
        F_vec = np.array([
            -d2h1_dz2 * (self.v_z**2),
            -d2h2_dz2 * (self.v_z**2),
            0.0
        ])
        
        D_mat = np.array([
            [np.cos(theta), -v * np.sin(theta), -dh1_dz],
            [np.sin(theta),  v * np.cos(theta), -dh2_dz],
            [0.0,            0.0,                1.0]
        ])
        
        nu_vec = -self.Kp @ y_vec - self.Kd @ ydot_vec
        
        v_eps = 1e-2
        if abs(v) < v_eps:
            direction_sign = 1.0 if self.vz_cmd >= 0.0 else -1.0
            desired_heading = math.atan2(direction_sign * dy_ds, direction_sign * dx_ds)
            heading_error = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
            
            v_des = self.v_min
            a_safe = np.clip(self.kv_speed_fb * (v_des - v), -self.max_vz, self.max_vz)
            omega_safe = np.clip(self.k_heading_fb * heading_error, -4.0, 4.0)
            
            if abs(self.vz_cmd) > 1e-6:
                a_z = self.vz_cmd - self.v_z
                a_z = np.clip(a_z, -self.max_az, self.max_az)
                self.v_z = np.clip(self.v_z + a_z * dt, -self.max_vz, self.max_vz)
                self.z += self.v_z * dt
            
            return np.array([a_safe, omega_safe])
        
        try:
            u_aug = np.linalg.solve(D_mat, (nu_vec - F_vec))
        except np.linalg.LinAlgError:
            return np.array([0.0, 0.0])
        
        a_aug, omega_aug, a_z = u_aug
        
        a_z = np.clip(a_z, -self.max_az, self.max_az)
        self.v_z = np.clip(self.v_z + a_z * dt, -self.max_vz, self.max_vz)
        self.z += self.v_z * dt
        
        a_aug = float(np.clip(a_aug, -10.0, 10.0))
        omega_aug = float(np.clip(omega_aug, -4.0, 4.0))
        
        return np.array([a_aug, omega_aug])


class PoseController:    
    def __init__(self, kp_pos=3.0, kv=3.0, k_theta=4.0, v_max=1.0, allow_reverse_heading=True):
        self.kp_pos = kp_pos
        self.kv = kv
        self.k_theta = k_theta
        self.v_max = v_max
        self.allow_reverse_heading = bool(allow_reverse_heading)
        self.max_acc = 15.0
        self.max_omega = 10.0
    
    def compute_controls(self, state, target_pos, dt, sigma=0.0):
        x, y, theta, v = state
        ex = target_pos[0] - x
        ey = target_pos[1] - y
        dist = np.hypot(ex, ey)
        
        desired_heading = math.atan2(ey, ex)
        e_fwd = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
        e_rev = np.mod(desired_heading + np.pi - theta + np.pi, 2*np.pi) - np.pi
        
        use_reverse = False
        if self.allow_reverse_heading and sigma > 0.5 and dist < 0.8 * self.v_max:
            if abs(e_rev) + 0.10 < abs(e_fwd):
                use_reverse = True
        heading_error = e_rev if use_reverse else e_fwd
        
        sigma = float(np.clip(sigma, 0.0, 1.0))
        kp_eff = self.kp_pos * (1.0 + 2.5 * sigma)
        kv_eff = self.kv * (1.0 + 1.5 * sigma)
        k_theta_eff = self.k_theta * (1.0 + 3.0 * sigma)
        v_max_eff = self.v_max
        
        v_des_base = v_max_eff * np.tanh(kp_eff * dist)
        v_des = v_des_base * np.cos(heading_error)
        if use_reverse:
            v_des = -v_des
        
        a = kv_eff * (v_des - v)
        omega = k_theta_eff * heading_error
        
        a = float(np.clip(a, -self.max_acc, self.max_acc))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        
        return np.array([a, omega])


class HybridController:    
    def __init__(self, curve, tfl_controller, pose_controller,
                 N_target=2.0, d_switch_frac=0.12, d_margin_frac=0.3,
                 safety_radius_frac=0.06, avoid_on_frac=0.10,
                 avoid_gain=1.0, v_avoid_frac=0.5,
                 k_omega_avoid=4.0, k_v_avoid=4.0,
                 min_cruise_speed_frac=0.05, sigma_cruise_off=0.3, k_cruise=2.0):
        self.curve = curve
        self.tfl = tfl_controller
        self.pose = pose_controller
        self.N_target = float(N_target)
        
        self.d_switch = d_switch_frac * curve.scale
        self.d_margin = d_margin_frac * curve.scale
        
        self.safety_radius = safety_radius_frac * curve.scale
        self.avoid_on_dist = avoid_on_frac * curve.scale
        self.avoid_gain = float(avoid_gain)
        self.v_avoid = float(v_avoid_frac)
        self.k_omega_avoid = float(k_omega_avoid)
        self.k_v_avoid = float(k_v_avoid)
        
        self.min_cruise_speed_frac = float(min_cruise_speed_frac)
        self.sigma_cruise_off = float(sigma_cruise_off)
        self.k_cruise = float(k_cruise)
    
    def blending_sigma(self, revs, dist):
        r = np.clip(revs / self.N_target, 0.0, 1.0)
        sN = smoothstep(r)
        
        d_norm = np.clip(dist / self.d_switch, 0.0, 1.0)
        sD = 1.0 - smoothstep(d_norm)
        
        sigma_min = min(sN, sD)
        sigma_product = sN * sD
        
        gap = abs(sN - sD)
        blend_weight = smoothstep(np.clip(gap / 0.3, 0.0, 1.0))
        
        sigma = (1.0 - blend_weight) * sigma_product + blend_weight * sigma_min
        
        return float(np.clip(sigma, 0.0, 1.0))
    
    def compute_controls(self, agent, dt):
        state = agent.state.copy()
        x, y = state[0], state[1]
        v_cur = state[3]
        
        u_tfl = self.tfl.compute_controls(state, dt)
        
        if agent.assigned_vertex is None:
            return u_tfl, 0.0
        
        sigma = agent.sigma
        u_pose = self.pose.compute_controls(state, agent.assigned_vertex, dt, sigma=sigma)
        u_nominal = (1.0 - sigma) * u_tfl + sigma * u_pose

        if sigma < self.sigma_cruise_off:
            v_des_min = self.min_cruise_speed_frac * agent.max_velocity
            a_boost = self.k_cruise * (v_des_min - v_cur)
            if a_boost > 0:
                u_nominal = u_nominal.copy()
                u_nominal[0] = np.clip(u_nominal[0] + a_boost,
                                       agent.min_acceleration, agent.max_acceleration)
        
        agents_list = getattr(agent, "_all_agents", None)
        if agents_list is None:
            return u_nominal, sigma
        
        u_avoid, alpha = self._compute_avoidance(agent, agents_list, dt)
        
        u_final = (1.0 - alpha) * u_nominal + alpha * u_avoid
        return u_final, sigma
    
    def _compute_avoidance(self, agent, agents, dt):
        p_i = agent.state[:2]
        theta = agent.state[2]
        v_cur = agent.state[3]
        
        sigma_fade_start = 0.75
        delta_fade = 0.25
        base_activation = 0.6
        
        def smoothstep_clip(u):
            u = float(np.clip(u, 0.0, 1.0))
            return 3 * u * u - 2 * u * u * u
        
        def a_of_sigma(sigma_i):
            if sigma_i < sigma_fade_start:
                return base_activation
            else:
                u = (sigma_i - sigma_fade_start) / max(1e-6, delta_fade)
                return base_activation * (1.0 - smoothstep_clip(u))
        
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
            
            sigma_i = agent.sigma
            ai = a_of_sigma(sigma_i)
            sigma_j = other.sigma
            
            dist_to_vertex_j = np.linalg.norm(p_j - other.assigned_vertex)
            is_settled_j = (sigma_j > 0.85) or (dist_to_vertex_j < 0.25 * self.d_switch)

            effective_avoid_dist = self.avoid_on_dist
            if is_settled_j:
                effective_avoid_dist = 1.5 * self.safety_radius
            
            t = np.clip(
                (effective_avoid_dist - dist) / max(1e-6, (effective_avoid_dist - self.safety_radius)),
                0.0, 1.0
            )
            pair_alpha = smoothstep_clip(t)
            
            if dist < effective_avoid_dist:
                theta_i = agent.state[2]
                theta_j = other.state[2]
                heading_diff = abs(np.mod(theta_i - theta_j + np.pi, 2*np.pi) - np.pi)
                
                dir_factor_base = 0.3 + 0.7 * (heading_diff / np.pi)
                dir_factor = dir_factor_base

                shield_active = is_settled_j
                if shield_active:
                    shield_radius = 1.30 * self.safety_radius
                    if dist < shield_radius:
                        dir_factor = max(dir_factor, 0.85)
                        ai_eff = max(ai, 0.6)
                        tt = np.clip(
                            (shield_radius - dist) / max(1e-6, (shield_radius - self.safety_radius)),
                            0.0, 1.0
                        )
                        shield_alpha = smoothstep_clip(tt)
                        pair_alpha = max(pair_alpha, shield_alpha)
                        force_mag = 1.5 * self.avoid_gain * ai_eff * pair_alpha * dir_factor / max(dist, 1e-3)
                        f += force_mag * (diff / dist)
                        alpha_local = max(alpha_local, ai_eff * pair_alpha * dir_factor)
                        continue

                force_mag = self.avoid_gain * ai * pair_alpha * dir_factor / max(dist, 1e-3)
                f += force_mag * (diff / dist)
                alpha_local = max(alpha_local, ai * pair_alpha * dir_factor)
        
        norm_f = np.linalg.norm(f)
        if alpha_local < 0.01 or norm_f < 1e-6:
            return np.array([0.0, 0.0]), 0.0
        
        v_des_avoid = self.v_avoid * agent.max_velocity
        desired_dir_vec = f / norm_f
        
        is_stuck = alpha_local > 0.2 and norm_f < 0.1 and abs(v_cur) < 0.05
        if is_stuck:
            desired_dir_vec = np.array([-math.sin(theta), math.cos(theta)])
        
        desired_heading = math.atan2(desired_dir_vec[1], desired_dir_vec[0])
        heading_error = np.mod(desired_heading - theta + np.pi, 2*np.pi) - np.pi
        
        v_des_signed = v_des_avoid * np.cos(heading_error)
        
        a_avoid = self.k_v_avoid * (v_des_signed - v_cur)
        omega_avoid = self.k_omega_avoid * heading_error
        
        a_avoid = np.clip(a_avoid, agent.min_acceleration, agent.max_acceleration)
        omega_avoid = np.clip(omega_avoid, -agent.max_angular_velocity, agent.max_angular_velocity)
        
        alpha_eff = float(np.clip(alpha_local, 0.0, 1.0))
        
        return np.array([a_avoid, omega_avoid]), alpha_eff