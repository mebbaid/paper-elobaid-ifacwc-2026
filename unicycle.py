"""
Unicycle kinematic model for differential-drive robots.
"""

import numpy as np


class Unicycle:
    """
    Kinematic unicycle model with velocity dynamics.
    
    State: [x, y, θ, v]
    - (x, y): position
    - θ: heading angle
    - v: forward velocity
    
    Control: [a, ω]
    - a: linear acceleration
    - ω: angular velocity
    """
    
    def __init__(self, x, y, theta, v=0.0):
        """
        Initialize unicycle state.
        
        Parameters
        ----------
        x, y : float
            Initial position
        theta : float
            Initial heading (radians)
        v : float
            Initial forward velocity
        """
        self.state = np.array([x, y, theta, v], dtype=float)
        
        # Physical limits
        self.max_acceleration = 2.0
        self.min_acceleration = -2.0
        self.max_angular_velocity = 100 * np.pi
        self.max_velocity = 2.0
        
        # Path following / formation control bookkeeping
        self.prev_s = None          # Previous path parameter
        self.cum_s = 0.0            # Cumulative (unwrapped) path parameter
        self.assigned_vertex = None # Target vertex [x, y] for formation
        self.s_revs = 0.0           # Number of revolutions (cum_s / 2π)
        self.s_last = None
        
        # Controller state sharing
        self.sigma = 0.0            # Blending parameter (0=TFL, 1=pose)
        self.last_sigma = 0.0
    
    def update(self, u, dt):
        """
        Update state using first-order Euler integration.
        
        Parameters
        ----------
        u : array_like, shape (2,)
            Control input [a, ω]
        dt : float
            Time step
        """
        x, y, theta, v = self.state
        u0, u1 = u
        
        # Saturate controls
        u0_clipped = np.clip(u0, self.min_acceleration, self.max_acceleration)
        u1_clipped = np.clip(u1, -self.max_angular_velocity, self.max_angular_velocity)
        
        # Kinematic equations
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = u1_clipped
        v_dot = u0_clipped
        
        # Integrate
        self.state += np.array([x_dot, y_dot, theta_dot, v_dot]) * dt
        
        # Velocity saturation
        self.state[3] = np.clip(self.state[3], -self.max_velocity, self.max_velocity)
        
        # Normalize theta to (-π, π]
        self.state[2] = np.mod(self.state[2] + np.pi, 2 * np.pi) - np.pi
    
    @property
    def position(self):
        """Get current position [x, y]"""
        return self.state[:2]
    
    @property
    def heading(self):
        """Get current heading θ"""
        return self.state[2]
    
    @property
    def velocity(self):
        """Get current forward velocity v"""
        return self.state[3]
