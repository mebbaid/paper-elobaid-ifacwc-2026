import numpy as np


class Unicycle:   
    def __init__(self, x, y, theta, v=0.0):
        self.state = np.array([x, y, theta, v], dtype=float)
        
        self.max_acceleration = 2.0
        self.min_acceleration = -2.0
        self.max_angular_velocity = 100 * np.pi
        self.max_velocity = 2.0
        
        self.prev_s = None
        self.cum_s = 0.0
        self.assigned_vertex = None
        self.s_revs = 0.0
        self.s_last = None
        self.sigma = 0.0
        self.last_sigma = 0.0
    
    def update(self, u, dt):
        x, y, theta, v = self.state
        u0, u1 = u
        
        u0_clipped = np.clip(u0, self.min_acceleration, self.max_acceleration)
        u1_clipped = np.clip(u1, -self.max_angular_velocity, self.max_angular_velocity)
        
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = u1_clipped
        v_dot = u0_clipped
        
        self.state += np.array([x_dot, y_dot, theta_dot, v_dot]) * dt
        self.state[3] = np.clip(self.state[3], -self.max_velocity, self.max_velocity)
        self.state[2] = np.mod(self.state[2] + np.pi, 2 * np.pi) - np.pi
    
    @property
    def position(self):
        return self.state[:2]
    
    @property
    def heading(self):
        return self.state[2]
    
    @property
    def velocity(self):
        return self.state[3]
