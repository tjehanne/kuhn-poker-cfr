import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from scipy.integrate import odeint

class DoubleSwingUpCartPoleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self):
        super(DoubleSwingUpCartPoleEnv, self).__init__()
        
        # --- PARAMETRES PHYSIQUES ---
        self.g = 9.81
        self.m_cart = 1.0   # Masse chariot
        self.m1 = 0.5       # Masse pendule 1
        self.m2 = 0.5       # Masse pendule 2
        self.l1 = 0.5       # Longueur 1
        self.l2 = 0.5       # Longueur 2
        self.force_mag = 20.0
        self.dt = 0.02      # Pas de temps simulation
        self.x_threshold = 5.0

        # Actions: Force continue entre -1 et 1 (multipliée par force_mag)
        # C'est MIEUX que discret pour le double pendule
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Obs: [x, x_dot, cos1, sin1, th1_dot, cos2, sin2, th2_dot]
        # On normalise un peu les bornes pour aider (mais Box gère bien)
        high = np.inf * np.ones(8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None

    def deriv(self, state, t, f):
        # State = [x, x_dot, th1, th1_dot, th2, th2_dot]
        x, x_dot, th1, th1_dot, th2, th2_dot = state
        
        # Force d'entrée
        u = f
        
        # Raccourcis
        s1, c1 = math.sin(th1), math.cos(th1)
        s2, c2 = math.sin(th2), math.cos(th2)
        s12 = math.sin(th1 - th2)
        
        # Masses combinées
        m = self.m1 + self.m2
        M = self.m_cart + m

        # Équations du mouvement (Lagrangien simplifié pour double pendule sur chariot)
        # C'est un système A * acceleration = B
        # A est la matrice de masse (inertie)
        # B est le vecteur des forces (gravité, coriolis, input)
        
        # Matrice A (3x3) pour [x_acc, th1_acc, th2_acc]
        # x_acc termes: M, l1*m*c1, m2*l2*c2
        # th1_acc termes: l1*m*c1, l1*l1*m, l1*l2*m2*c12
        # th2_acc termes: m2*l2*c2, l1*l2*m2*c12, l2*l2*m2
        
        # Pour simplifier et éviter d'importer scipy.linalg ou numpy.linalg lourd dans la boucle step,
        # on écrit les équations explicitement ou on utilise numpy solve.
        
        A = np.array([
            [M,              self.l1*m*c1,       self.m2*self.l2*c2],
            [self.l1*m*c1,   self.l1**2*m,       self.l1*self.l2*self.m2*math.cos(th1-th2)],
            [self.m2*self.l2*c2, self.l1*self.l2*self.m2*math.cos(th1-th2), self.l2**2*self.m2]
        ])

        B = np.array([
            u + self.l1*m*s1*th1_dot**2 + self.m2*self.l2*s2*th2_dot**2, # Coriolis/Centrifuge sur x
            -self.l1*self.l2*self.m2*s12*th2_dot**2 + m*self.g*self.l1*s1, # Gravité + interaction sur th1
            self.l1*self.l2*self.m2*s12*th1_dot**2 + self.m2*self.g*self.l2*s2 # Gravité + interaction sur th2
        ])

        # Résolution A * acc = B
        # acc = [x_acc, th1_acc, th2_acc]
        acc = np.linalg.solve(A, B)
        
        return np.array([x_dot, acc[0], th1_dot, acc[1], th2_dot, acc[2]])

    def step(self, action):
        # Action est un tableau numpy, on prend la valeur scalaire
        force = float(action[0]) * self.force_mag
        
        # Intégration Euler semi-implicite (plus simple et rapide que RK4 pour RL)
        # Ou Euler simple si le pas de temps est petit.
        # Ici on fait une approximation simple pour la rapidité :
        
        # On récupère les dérivées à l'instant t
        derivs = self.deriv(self.state, 0, force)
        
        # Mise à jour état (Euler Integration)
        self.state = self.state + derivs * self.dt
        
        # Unpack
        x, x_dot, th1, th1_dot, th2, th2_dot = self.state

        # --- REWARD FUNCTION (Optimisée Swing-Up + Balance) ---
        
        # 1. Alignement vertical (Hauteur des pointes)
        # Partie "Guidage Global" : Encourage à monter le plus haut possible.
        # Va de -2 (tout en bas) à +2 (tout en haut).
        reward_height = np.cos(th1) + np.cos(th2)
        
        # Partie "Précision" : Uniquement quand on est proche du but.
        # C'est ça qui verrouille l'équilibre.
        cos_dist_1 = (1.0 - np.cos(th1)) 
        cos_dist_2 = (1.0 - np.cos(th2))
        reward_precision = np.exp(-2.0 * (cos_dist_1 + cos_dist_2))
        
        # 2. Pénalité de Vitesse (Ajustée)
        # On pénalise la vitesse, MAIS on doit permettre l'élan pour monter.
        # Donc on réduit un peu la pénalité par rapport à avant, ou on accepte que monter coûte un peu.
        penalty_spin = -0.05 * (th1_dot**2) - 0.1 * (th2_dot**2)
        
        # 3. Pénalité distance chariot et Vitesse chariot
        penalty_cart = -0.1 * (x**2) - 0.05 * (x_dot**2)
        
        # 4. Bonus d'équilibre parfait
        bonus_static = 0.0
        if cos_dist_1 < 0.1 and cos_dist_2 < 0.1:
            if abs(th1_dot) < 1.0 and abs(th2_dot) < 1.0:
                bonus_static = 5.0

        # Total : On combine le guidage (x2) et la précision (x10)
        # Si on est en bas : Height=-2, Prec=0 -> Reward faible mais on veut monter.
        # Si on est en haut : Height=2, Prec=1 -> Reward MAX.
        reward = float(2.0 * reward_height + 10.0 * reward_precision + penalty_spin + penalty_cart + bonus_static)

        # Conditions d'arrêt
        terminated = bool(x < -self.x_threshold or x > self.x_threshold)
        
        # Normalization implicite dans l'obs (cos/sin sont bornés)
        obs = np.array([
            x, x_dot, 
            np.cos(th1), np.sin(th1), th1_dot,
            np.cos(th2), np.sin(th2), th2_dot
        ], dtype=np.float32)

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- INITIALISATION HYBRIDE ---
        # 50% du temps : On commence en haut (Equilibre)
        # 50% du temps : On commence aléatoirement (Swing-Up)
        
        if np.random.rand() < 0.5:
            # Mode "Start at Top" (Apprentissage finesse)
            low, high = -0.05, 0.05
            self.state = np.random.uniform(low=low, high=high, size=(6,))
        else:
            # Mode "Swing Up" (Apprentissage force brute)
            # On initialise les angles n'importe où entre -PI et PI
            # Mais on garde x proche de 0
            self.state = np.zeros(6)
            self.state[0] = np.random.uniform(-1, 1) # x
            self.state[2] = np.random.uniform(-np.pi, np.pi) # th1
            self.state[4] = np.random.uniform(-np.pi, np.pi) # th2
            # Vitesses faibles au début
            self.state[1] = np.random.uniform(-0.1, 0.1)
            self.state[3] = np.random.uniform(-0.1, 0.1)
            self.state[5] = np.random.uniform(-0.1, 0.1)
        
        self.state = self.state.astype(np.float32)

        x, x_dot, th1, th1_dot, th2, th2_dot = self.state
        obs = np.array([
            x, x_dot, 
            np.cos(th1), np.sin(th1), th1_dot,
            np.cos(th2), np.sin(th2), th2_dot
        ], dtype=np.float32)
        
        return obs, {}