import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class SwingUpCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super(SwingUpCartPoleEnv, self).__init__()
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20.0 # PLUS PUISSANT (Avant 10.0)
        self.tau = 0.02 
        self.x_threshold = 5.0 # Augmenté (était 2.4)

        self.action_space = spaces.Discrete(2)

        # Obs: [x, x_dot, cos(theta), sin(theta), theta_dot]
        high = np.array([self.x_threshold * 2, np.finfo(np.float32).max, 1, 1, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        # --- CORRECTION TYPE ---
        # On sauvegarde en numpy array pour pouvoir le modifier via l'interface plus tard
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # --- NOUVELLE FONCTION DE RÉCOMPENSE (Optimisée V3.2 - HIGH POWER) ---
        
        # 1. Angle: 
        # Base: -1 (Bas) à +1 (Haut)
        reward_theta = np.cos(theta)
        
        # BONUS CRITIQUE : Si on est presque vertical, on donne un gros bonbon.
        # C'est ça qui va lui apprendre à MAINTENIR l'équilibre.
        if reward_theta > 0.95:
            reward_theta += 10.0 

        # 2. Position: Pénalité quadratique
        # On veut qu'il reste au centre (0), mais on tolère un peu de mouvement
        reward_x = -0.2 * (x**2)
        
        # 3. Vitesse Pendule: 
        # On punit la vitesse angulaire SEULEMENT si on est proche de l'équilibre
        # Pour forcer la stabilisation immobile.
        reward_vel = 0.0
        if np.cos(theta) > 0.8:
            reward_vel = -0.5 * (theta_dot**2) # Pénalité forte sur la rotation rapide en haut
        
        reward = float(reward_theta + reward_x + reward_vel)

        # DEBUG: Voir ce que le réseau voit
        # if np.abs(x) < 0.1: # Pour ne pas spammer, on affiche que au début
        #    print(f"Action: {action} | Obs: x={x:.2f}, dx={x_dot:.2f}, th={theta:.2f}, dth={theta_dot:.2f} | Rew: {reward:.2f}")

        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        
        # Pas de bonus discret (+1) pour éviter les sauts de gradient
        
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        low, high = -0.05, 0.05
        self.state = np.random.uniform(low=low, high=high, size=(4,))
        # self.state[2] += np.pi # Tête en bas (DÉSACTIVÉ POUR TEST ÉQUILIBRE)
        # On démarre tout droit (0 radian +/- bruit)
        
        self.state = self.state.astype(np.float32) # Force float32

        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        return obs, {}