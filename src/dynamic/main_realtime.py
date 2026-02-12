import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import sys
import os

# --- SETUP CHEMINS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.config import ACTIONS, SEQUENCE_LENGTH, MODEL_DYNAMIC
from src.core.processing import normalize_landmarks
from src.dynamic.voice import speak  # Assure-toi que ce fichier existe bien dans src/dynamic/ ou src/core/

# --- 1. CHARGEMENT DU MODÈLE ---
print("🧠 Chargement du cerveau LSTM...")
try:
    model = tf.keras.models.load_model(MODEL_DYNAMIC)
except OSError:
    print(f"❌ Erreur : Impossible de trouver {MODEL_DYNAMIC}.")
    exit()

# --- 2. SETUP MEDIAPIPE OPTIMISÉ ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Baissé un peu pour la rapidité
    min_tracking_confidence=0.5,
    model_complexity=0  # 0 = Lite (Rapide), 1 = Full (Précis). Passe à 0 si ça lag encore.
)

# --- 3. VARIABLES DE GESTION ---
sequence = []
sentence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

# OPTIMISATION : Compteur pour ne pas prédire à chaque image
frame_counter = 0
PREDICTION_FREQ = 5  # On prédit seulement toutes les 5 images

print("🎥 Lancement de la caméra... Fais un signe !")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Préparation image
    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Dessin
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 1. On récupère les points
            keypoints = normalize_landmarks(hand_landmarks.landmark)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            # --- OPTIMISATION MAJEURE ---
            # On prédit SEULEMENT si on a assez de frames ET que c'est le bon moment (toutes les 5 frames)
            if len(sequence) == SEQUENCE_LENGTH and (frame_counter % PREDICTION_FREQ == 0):

                # Astuce : model(...) est beaucoup plus rapide que model.predict(...) dans une boucle
                input_data = np.expand_dims(sequence, axis=0)
                res = model(input_data, training=False)[0].numpy()

                best_class_id = np.argmax(res)
                confidence = res[best_class_id]

                # Stabilisation
                predictions.append(best_class_id)
                predictions = predictions[-10:]

                # On vérifie la stabilité (ici on vérifie sur les 10 dernières prédictions,
                # vu qu'on prédit moins souvent, ça couvre une période plus longue, c'est plus stable)
                if np.unique(predictions)[-1] == best_class_id:
                    if confidence > threshold:
                        current_word = ACTIONS[best_class_id]

                        # Affichage console léger
                        print(f"Détecté : {current_word} ({confidence:.2f})")

                        if len(sentence) > 0:
                            if current_word != sentence[-1]:
                                sentence = [current_word]
                                speak(current_word)
                        else:
                            sentence.append(current_word)
                            speak(current_word)

    # --- AFFICHAGE ---
    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
    if len(sentence) > 0:
        cv2.putText(image, sentence[0], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Traducteur LSF - Temps Reel', image)

    frame_counter += 1  # On incrémente le compteur

    if cv2.waitKey(1) & 0xFF == ord('q'):  # waitKey(1) est plus fluide que waitKey(10)
        break

cap.release()
cv2.destroyAllWindows()