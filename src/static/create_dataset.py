import cv2
import mediapipe as mp
import numpy as np
import csv
import os

from src.core.processing import normalize_landmarks

# --- CONFIGURATION ---
FILE_NAME = "../../data/static/dataset_couleurs.csv"  # On change le nom pour ne pas mélanger
CLASSES = {
    ord('b'): 0,  # Touche 'b' -> Bleu
    ord('v'): 1,  # Touche 'v' -> Vert
    ord('j'): 2,  # Touche 'j' -> Jaune
    ord('r'): 3,  # Touche 'r' -> Rouge
    ord('o'): 4   # Touche 'o' -> Orange
}

# --- INITIALISATION MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# --- PREPARATION DU FICHIER CSV ---
# Si le fichier n'existe pas, on le crée
if not os.path.exists(FILE_NAME):
    with open(FILE_NAME, 'w', newline='') as f:
        writer = csv.writer(f)
        # Pas d'en-tête nécessaire pour le réseau, mais utile pour nous si on veut vérifier
        # On laisse vide pour rester compatible format "raw"
        pass

cap = cv2.VideoCapture(0)
print(f"--- OUTIL DE CREATION DE DATASET ---")
print(f"Fichier de sortie : {FILE_NAME}")
print(f"Instructions : Fais un signe et appuie sur la touche correspondante (a, b, c).")
print(f"Appuie sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Miroir + Conversion couleur
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    data_saved = False  # Juste pour l'affichage visuel

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Attente d'une touche
            key = cv2.waitKey(1) & 0xFF

            if key in CLASSES:
                # C'est une touche connue (a, b ou c) -> On enregistre !
                class_id = CLASSES[key]

                # 1. Normalisation
                normalized_data = normalize_landmarks(hand_landmarks.landmark)

                # 2. Ajout de la classe en premier (Format: [label, x1, y1, ...])
                row = [class_id] + normalized_data

                # 3. Ecriture dans le CSV
                with open(FILE_NAME, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"Enregistré : Classe {chr(key).upper()} (Label {class_id})")
                data_saved = True

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    # Feedback visuel si sauvegardé
    if data_saved:
        cv2.putText(frame, "SAUVEGARDE !", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Dataset Creator', frame)

    # Nécessaire pour rafraîchir la fenêtre si on n'a pas appuyé sur une touche de classe
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()