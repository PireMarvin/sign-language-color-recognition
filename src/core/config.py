import os
import numpy as np

# --- PARAMÈTRES DES DONNÉES ---

# Liste des actions (On utilise np.array pour faciliter l'entraînement plus tard)
ACTIONS = np.array(["BLEU", "VERT", "JAUNE", "ROUGE", "ORANGE"])

# Nombre de vidéos à enregistrer par action
NO_SEQUENCES = 30

# Nombre d'images (frames) par vidéo
SEQUENCE_LENGTH = 30

# --- GESTION DES CHEMINS ---

# On remonte de 3 niveaux pour trouver la racine du projet (src/core/config.py -> src/core -> src -> racine)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Chemins STATIQUES (Photo)
DATA_STATIC = os.path.join(BASE_DIR, "data", "static", "dataset_couleurs.csv")
MODEL_STATIC = os.path.join(BASE_DIR, "models", "modele_couleurs.npz")

# Chemins DYNAMIQUES (Vidéo)
DATA_DYNAMIC = os.path.join(BASE_DIR, "data", "dynamic")
# On définit déjà où on sauvegardera le cerveau vidéo (extension .keras ou .h5)
MODEL_DYNAMIC = os.path.join(BASE_DIR, "models", "modele_lstm.keras")