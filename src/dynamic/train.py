import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- IMPORTS DU CŒUR ---
# On rajoute le chemin racine pour trouver src.core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, DATA_DYNAMIC, MODEL_DYNAMIC

# --- 1. CHARGEMENT DES DONNÉES ---
print("📂 Chargement des séquences vidéos...")

sequences, labels = [], []

# On crée un dictionnaire pour dire : "BLEU" = 0, "VERT" = 1, etc.
label_map = {label: num for num, label in enumerate(ACTIONS)}

for action in ACTIONS:
    print(f"   - Chargement de l'action : {action}")
    for sequence in range(NO_SEQUENCES):
        # On charge le fichier .npy (ex: data/dynamic/BLEU/0.npy)
        window = np.load(os.path.join(DATA_DYNAMIC, action, "{}.npy".format(sequence)))

        sequences.append(window)
        labels.append(label_map[action])

# Conversion en tableaux Numpy
X = np.array(sequences)
y = to_categorical(labels).astype(int)  # Convertit [0, 1] en [[1,0,0,0,0], [0,1,0,0,0]]

# Séparation : 95% pour l'entraînement, 5% pour tester si ça marche
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"📊 Données prêtes : {X.shape[0]} séquences chargées.")

# --- 2. CRÉATION DU MODÈLE LSTM ---
print("🧠 Construction du réseau de neurones...")

model = Sequential()

# Couche 1 : LSTM (reçoit les 30 frames de 42 points)
# return_sequences=True signifie qu'il passe l'info à la couche suivante
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 42)))

# Couche 2 : LSTM (reçoit l'info de la couche 1)
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Couche 3 : LSTM (dernière couche récurrente)
# return_sequences=False car on veut maintenant une décision finale, plus une suite
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Couche 4 : Dense (Analyse classique)
model.add(Dense(64, activation='relu'))

# Couche 5 : Dense (Sortie - 32 neurones pour stabiliser)
model.add(Dense(32, activation='relu'))

# Couche Finale : Softmax (Probabilités pour nos 5 couleurs)
model.add(Dense(len(ACTIONS), activation='softmax'))

# Compilation
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 3. ENTRAÎNEMENT ---
print("🚀 Début de l'entraînement ...")

# epochs = nombre de fois qu'il voit tout le dataset (200 à 1000 selon la difficulté)
model.fit(X_train, y_train, epochs=200, callbacks=[])

# --- 4. SAUVEGARDE ---
model.summary()
print("💾 Sauvegarde du modèle...")
model.save(MODEL_DYNAMIC)  # Sauvegarde en .keras ou .h5 selon ta config
print(f"✅ Terminé ! Modèle sauvegardé dans : {MODEL_DYNAMIC}")