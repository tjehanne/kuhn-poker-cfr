import gymnasium as gym
from stable_baselines3 import PPO, DQN
import pygame
import numpy as np
import math

# --- CONFIGURATION ---
MODEL_PATH = "models/PPO/100000.zip" # Vérifie que ce fichier existe bien
ENV_NAME = "CartPole-v1"
ALGO_CLASS = PPO 

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (200, 50, 50)
GREY  = (150, 150, 150)
BLUE  = (50, 50, 200)

# Dimensions écran
WIDTH, HEIGHT = 800, 400
SCALE = 100 

def draw_arrow(screen, color, start, end, width=3):
    # Conversion des points de départ/arrivée en entiers pour la ligne
    start_int = (int(start[0]), int(start[1]))
    end_int = (int(end[0]), int(end[1]))
    
    pygame.draw.line(screen, color, start_int, end_int, width)
    
    # Calculs pour la tête de la flèche
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    
    # Création des 3 points du triangle de la flèche (avec conversion int !)
    p1 = (
        int(end[0]+20*math.sin(math.radians(rotation))), 
        int(end[1]+20*math.cos(math.radians(rotation)))
    )
    p2 = (
        int(end[0]+20*math.sin(math.radians(rotation-120))), 
        int(end[1]+20*math.cos(math.radians(rotation-120)))
    )
    p3 = (
        int(end[0]+20*math.sin(math.radians(rotation+120))), 
        int(end[1]+20*math.cos(math.radians(rotation+120)))
    )
    
    pygame.draw.polygon(screen, color, (p1, p2, p3))

def run_visualizer():
    # 1. Init Env & Model
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    try:
        model = ALGO_CLASS.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Erreur: Modèle '{MODEL_PATH}' introuvable.")
        return

    # 2. Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Control: {ENV_NAME}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 24, bold=True)

    obs, _ = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- IA ---
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        cart_x = obs[0]
        pole_angle = obs[2]
        
        if terminated or truncated:
            obs, _ = env.reset()

        # --- DESSIN ---
        screen.fill(WHITE)
        base_x = WIDTH // 2
        base_y = HEIGHT // 2 + 50

        # Sol
        pygame.draw.line(screen, BLACK, (0, base_y + 15), (WIDTH, base_y + 15), 2)

        # Calculs positions (FLOATS)
        screen_cart_x = base_x + (cart_x * SCALE)
        screen_cart_y = base_y
        
        pole_len = 120
        pole_end_x = screen_cart_x + pole_len * math.sin(pole_angle)
        pole_end_y = screen_cart_y - pole_len * math.cos(pole_angle)

        # --- CONVERSION EN ENTIERS (CRUCIAL POUR PYGAME) ---
        cx_int = int(screen_cart_x)
        cy_int = int(screen_cart_y)
        px_int = int(pole_end_x)
        py_int = int(pole_end_y)

        # Dessin Chariot
        cart_width = 80
        cart_height = 40
        # On centre le rectangle par rapport au point central du chariot
        rect_x = int(cx_int - cart_width // 2)
        rect_y = int(cy_int - cart_height // 2)

        cart_rect = pygame.Rect(rect_x, rect_y, cart_width, cart_height)
        
        pygame.draw.rect(screen, GREY, cart_rect)
        pygame.draw.rect(screen, BLACK, cart_rect, 2)

        # Roues
        pygame.draw.circle(screen, BLACK, (cx_int - 25, cy_int + 20), 10)
        pygame.draw.circle(screen, BLACK, (cx_int + 25, cy_int + 20), 10)

        # Tige (Pendule)
        # On utilise les coordonnées converties en int
        pygame.draw.line(screen, BLACK, (cx_int, cy_int), (px_int, py_int), 6)
        
        # Masse (Bout rouge)
        pygame.draw.circle(screen, RED, (px_int, py_int), 15)

        # Force (Flèche)
        force_len = 60
        if action == 1: # Droite
            # On passe les tuples d'entiers ou de floats, 
            # mais ma fonction draw_arrow corrigée gère la conversion maintenant.
            draw_arrow(screen, BLUE, (cx_int, cy_int), (cx_int + force_len, cy_int))
        else: # Gauche
            draw_arrow(screen, BLUE, (cx_int, cy_int), (cx_int - force_len, cy_int))

        # Texte
        title = title_font.render(f"Control: {ENV_NAME}", True, BLACK)
        screen.blit(title, (20, 20))
        
        angle_deg = math.degrees(pole_angle)
        txt_angle = font.render(f"Angle: {angle_deg:.2f} deg", True, BLACK)
        txt_pos = font.render(f"Position: {cart_x:.2f} m", True, BLACK)
        screen.blit(txt_angle, (20, 60))
        screen.blit(txt_pos, (20, 80))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    env.close()

if __name__ == "__main__":
    run_visualizer()