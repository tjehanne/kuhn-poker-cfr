import gymnasium as gym
from stable_baselines3 import PPO, DQN
import pygame
import numpy as np
import math

# --- CONFIGURATION ---
MODEL_PATH = "V2/models/PPO/100000.zip" # Ton modèle
ENV_NAME = "CartPole-v1"
ALGO_CLASS = PPO 

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 50, 50)
GREY  = (150, 150, 150)
BLUE  = (50, 50, 200)
GREEN = (50, 200, 50) # Couleur de l'élastique

# Dimensions écran
WIDTH, HEIGHT = 800, 400
SCALE = 100 

def draw_arrow(screen, color, start, end, width=3):
    start_int = (int(start[0]), int(start[1]))
    end_int = (int(end[0]), int(end[1]))
    pygame.draw.line(screen, color, start_int, end_int, width)
    
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    p1 = (int(end[0]+20*math.sin(math.radians(rotation))), int(end[1]+20*math.cos(math.radians(rotation))))
    p2 = (int(end[0]+20*math.sin(math.radians(rotation-120))), int(end[1]+20*math.cos(math.radians(rotation-120))))
    p3 = (int(end[0]+20*math.sin(math.radians(rotation+120))), int(end[1]+20*math.cos(math.radians(rotation+120))))
    pygame.draw.polygon(screen, color, (p1, p2, p3))

def run_visualizer():
    # 1. Init Env & Model
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env.unwrapped.theta_threshold_radians = 360 * 2 * math.pi / 360
    try:
        model = ALGO_CLASS.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Erreur: Modèle '{MODEL_PATH}' introuvable.")
        return

    # 2. Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Interactive RL: {ENV_NAME}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 24, bold=True)

    obs, _ = env.reset()
    running = True

    # Variables pour l'interaction souris
    is_dragging = False
    mouse_pos = (0, 0)

    while running:
        # --- GESTION DES EVENEMENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Clic souris : Activation
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Clic gauche
                    is_dragging = True
            
            # Relachement souris : Désactivation
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_dragging = False
            
            # Mouvement souris
            elif event.type == pygame.MOUSEMOTION:
                mouse_pos = event.pos

        # --- IA (Prediction) ---
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Récupération état physique
        cart_x = obs[0]
        pole_angle = obs[2]
        
        if terminated or truncated:
            obs, _ = env.reset()
            # On arrête de tirer si le pendule tombe
            is_dragging = False 

        # --- CALCULS GRAPHIQUES ---
        base_x = WIDTH // 2
        base_y = HEIGHT // 2 + 50
        
        screen_cart_x = base_x + (cart_x * SCALE)
        screen_cart_y = base_y
        
        pole_len = 120
        pole_end_x = screen_cart_x + pole_len * math.sin(pole_angle)
        pole_end_y = screen_cart_y - pole_len * math.cos(pole_angle)

        cx_int, cy_int = int(screen_cart_x), int(screen_cart_y)
        px_int, py_int = int(pole_end_x), int(pole_end_y)

        # --- INTERACTION PHYSIQUE (PERTURBATION) ---
        if is_dragging:
            mx, my = mouse_pos
            
            # 1. Calcul de la distance horizontale (la force)
            # Si la souris est à droite du pendule, dx est positif
            dx = mx - px_int
            
            # 2. Application de la force (Modification de la vitesse angulaire)
            # On accède directement aux variables internes de l'environnement (HACK)
            # state = [x, x_dot, theta, theta_dot]
            # On ajoute un petit delta à theta_dot (index 3)
            FORCE_MULTIPLIER = 0.002 # Ajuste cette valeur pour rendre "l'élastique" plus ou moins fort
            
            # On modifie l'état interne
            env.unwrapped.state[3] += dx * FORCE_MULTIPLIER
            
            # 3. Mise à jour de l'observation pour que l'IA "sente" le changement au prochain tour
            # (Optionnel car l'env le fera au prochain step, mais plus propre ainsi)
            obs[3] = env.unwrapped.state[3]

        # --- DESSIN ---
        screen.fill(WHITE)
        
        # Sol
        pygame.draw.line(screen, BLACK, (0, base_y + 15), (WIDTH, base_y + 15), 2)

        # Dessin Elastique (Si dragging)
        if is_dragging:
            # Ligne verte entre le bout du pendule et la souris
            pygame.draw.line(screen, GREEN, (px_int, py_int), mouse_pos, 2)
            # Petit rond à la souris
            pygame.draw.circle(screen, GREEN, mouse_pos, 5)

        # Chariot
        cart_width, cart_height = 80, 40
        rect_x, rect_y = int(cx_int - cart_width // 2), int(cy_int - cart_height // 2)
        cart_rect = pygame.Rect(rect_x, rect_y, cart_width, cart_height)
        
        pygame.draw.rect(screen, GREY, cart_rect)
        pygame.draw.rect(screen, BLACK, cart_rect, 2)
        pygame.draw.circle(screen, BLACK, (cx_int - 25, cy_int + 20), 10)
        pygame.draw.circle(screen, BLACK, (cx_int + 25, cy_int + 20), 10)

        # Tige
        pygame.draw.line(screen, BLACK, (cx_int, cy_int), (px_int, py_int), 6)
        
        # Masse (Rouge si normal, Vert si on tire dessus)
        bob_color = GREEN if is_dragging else RED
        pygame.draw.circle(screen, bob_color, (px_int, py_int), 15)

        # Force de l'agent
        force_len = 60
        if action == 1:
            draw_arrow(screen, BLUE, (cx_int, cy_int), (cx_int + force_len, cy_int))
        else:
            draw_arrow(screen, BLUE, (cx_int, cy_int), (cx_int - force_len, cy_int))

        # Texte
        title = title_font.render(f"PPO Agent - Perturbations", True, BLACK)
        screen.blit(title, (20, 20))
        
        if is_dragging:
            txt_mode = font.render("MODE: PERTURBATION EXTERNE", True, GREEN)
            screen.blit(txt_mode, (20, 50))
        
        angle_deg = math.degrees(pole_angle)
        txt_info = font.render(f"Angle: {angle_deg:.1f}° | Pos: {cart_x:.1f}", True, BLACK)
        screen.blit(txt_info, (20, 370))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()