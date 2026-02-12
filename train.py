import numpy as np
import pandas as pd
from neural_net import MLP

# --- CONFIGURATION ---
CSV_FILE = "dataset_couleurs.csv"
HIDDEN_NEURONS = 30  # 30 neurones pour gérer 5 formes, c'est suffisant
OUTPUT_CLASSES = 5  # Bleu, Vert, Jaune, Rouge, Orange
EPOCHS = 10000  # Nombre de répétitions
LEARNING_RATE = 0.1


def load_data(filename):
    print("⏳ Chargement des données...")
    try:
        # Lecture du CSV (header=None car il n'y a pas de titres de colonnes)
        df = pd.read_csv(filename, header=None)

        # On mélange les données (SHUFFLE) pour que le réseau n'apprenne pas l'ordre
        df = df.sample(frac=1).reset_index(drop=True)

        # Séparation : Colonne 0 = Label, Colonnes 1 à 42 = Features
        y_raw = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

        # One-Hot Encoding (Ex: 1 devient [0, 1, 0, 0, 0])
        y = np.eye(OUTPUT_CLASSES)[y_raw.astype(int)]

        return X, y
    except Exception as e:
        print(f"❌ Erreur : {e}")
        exit()


# 1. Préparation
X, y = load_data(CSV_FILE)
print(f"📊 Données chargées : {X.shape[0]} exemples pour {OUTPUT_CLASSES} couleurs.")

# 2. Création du cerveau
nn = MLP(input_size=42,
         hidden_size=HIDDEN_NEURONS,
         output_size=OUTPUT_CLASSES,
         learning_rate=LEARNING_RATE)

print(f"🚀 Début de l'entraînement sur {EPOCHS} époques...")

# 3. Entraînement
for i in range(EPOCHS):
    nn.train(X, y)

    # Affichage de la progression
    if i % 1000 == 0:
        loss = np.mean(np.square(y - nn.forward(X)))
        print(f"Epoch {i} \t Erreur : {loss:.6f}")

# 4. Sauvegarde finale
print("🏁 Entraînement terminé !")
nn.save_weights("modele_couleurs.npz")