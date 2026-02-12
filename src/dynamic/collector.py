import cv2
import numpy as np
import os
import time
import mediapipe as mp
import sys

# Astuce pour trouver les modules "core"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.config import ACTIONS, NO_SEQUENCES, SEQUENCE_LENGTH, DATA_DYNAMIC
from src.core.processing import normalize_landmarks

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --- CRÉATION DES DOSSIERS ---
for action in ACTIONS:
    try:
        os.makedirs(os.path.join(DATA_DYNAMIC, action))
    except FileExistsError:
        pass

cap = cv2.VideoCapture(0)

# --- BOUCLE PRINCIPALE ---
for action in ACTIONS:
    print(f"🎬 Préparation pour : {action}")

    # On boucle sur le nombre de vidéos (30 vidéos par action)
    for sequence in range(NO_SEQUENCES):

        # On boucle sur la longueur de la vidéo (30 frames par vidéo)
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret: break

            # Traitement Image
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Dessin
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- LOGIQUE D'AFFICHAGE ET D'ATTENTE ---

            # Si c'est la première frame de la vidéo (frame 0)
            if frame_num == 0:
                cv2.putText(image, 'PREPAREZ-VOUS', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Action: {action} | Video: {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # On affiche l'image et on attend 2 secondes pour que tu te places
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000)
            else:
                # Pendant l'enregistrement
                cv2.putText(image, f'Enregistrement... {frame_num}/{SEQUENCE_LENGTH}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

            # --- EXTRACTION ET SAUVEGARDE DES POINTS ---
            keypoints = np.zeros(42)  # Par défaut, des zéros si pas de main

            if results.multi_hand_landmarks:
                # On utilise ta fonction du CORE
                keypoints = normalize_landmarks(results.multi_hand_landmarks[0].landmark)

            # On définit le chemin de sauvegarde
            # Ex: data/dynamic/BLEU/0.npy, data/dynamic/BLEU/1.npy ...
            npy_path = os.path.join(DATA_DYNAMIC, action, str(sequence))

            # Si c'est la première frame, on crée le fichier, sinon on l'ouvre ?
            # NON : En LSTM, on doit sauvegarder TOUTE la séquence d'un coup.
            # Donc on stocke en mémoire vive d'abord.

            if frame_num == 0:
                sequence_data = []

            sequence_data.append(keypoints)

            # Si on est à la dernière frame (29), on sauvegarde le fichier complet
            if frame_num == SEQUENCE_LENGTH - 1:
                np.save(npy_path, np.array(sequence_data))

            # Quitter avec 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()