import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
import os

# Création des dossiers pour sauvegarder les modèles
models_dir = "models/PPO"
log_dir = "logs"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# 1. Choix de l'environnement
# CartPole-v1 est DISCRET (Gauche/Droite). 
# DQN et PPO fonctionnent. SAC ne fonctionne pas nativement (il lui faut du continu).
env = gym.make("CartPole-v1") 

# 2. Choix de l'algorithme
# Pour changer d'algo, remplace PPO par DQN.
# Note : Pour utiliser SAC, il faudrait un environnement à actions continues 
# (ex: 'Pendulum-v1' ou un CustomCartPole).
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# 3. Entraînement
TIMESTEPS = 10000
print("Début de l'entraînement...")
# On entraîne par blocs pour voir la progression (optionnel, on peut mettre 100k direct)
for i in range(1, 11):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    print(f"Modèle sauvegardé à {TIMESTEPS*i} pas.")

print("Entraînement terminé.")
env.close()