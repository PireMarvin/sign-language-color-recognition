import cv2
import mediapipe as mp
import numpy as np
from neural_net import MLP

# --- CONFIGURATION ---
CLASSES = ["BLEU", "VERT", "JAUNE", "ROUGE", "ORANGE"]
MODEL_FILE = "modele_couleurs.npz"
HIDDEN_NEURONS = 30  # Doit être identique à l'entraînement !

# --- 1. CHARGEMENT DU CERVEAU ---
print("🧠 Chargement du réseau de neurones...")
# On recrée la structure vide
nn = MLP(input_size=42, hidden_size=HIDDEN_NEURONS, output_size=5)
# On injecte les poids appris
try:
    nn.load_weights(MODEL_FILE)
except FileNotFoundError:
    print(f"❌ Erreur : Le fichier {MODEL_FILE} est introuvable. As-tu lancé train.py ?")
    exit()


# --- 2. FONCTION DE NORMALISATION (Copier-coller de create_dataset) ---
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist
    max_value = np.max(np.abs(coords))
    if max_value > 0:
        coords = coords / max_value
    return coords.flatten().tolist()


# --- 3. INITIALISATION CAMÉRA & MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # On est plus exigeant pour éviter les faux positifs
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("🎥 Lancement de la détection temps réel... Appuie sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Miroir + Conversion RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Zone d'affichage du texte
    # On met un bandeau noir en haut pour écrire le résultat proprement
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # A. Dessiner le squelette
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # B. Préparer la donnée pour le réseau
            input_data = normalize_landmarks(hand_landmarks.landmark)
            input_data = np.array([input_data])  # Le réseau attend une liste de listes (Batch)

            # C. PRÉDICTION
            prediction = nn.forward(input_data)  # Ça sort un truc genre [0.01, 0.99, 0.0, ...]

            # --- AJOUT DIAGNOSTIC ---
            # Affiche les probas brutes pour voir ce qui se passe
            print(
                f"B:{prediction[0][0]:.2f} | V:{prediction[0][1]:.2f} | J:{prediction[0][2]:.2f} | R:{prediction[0][3]:.2f} | O:{prediction[0][4]:.2f}")
            # ------------------------

            class_id = np.argmax(prediction)  # On prend l'index du plus grand (ex: 1)
            confidence = prediction[0][class_id]  # On regarde la probabilité (ex: 0.99)

            # D. Affichage intelligent
            # On affiche seulement si l'IA est sûre à plus de 80% (0.8)
            if confidence > 0.8:
                color_name = CLASSES[class_id]
                text = f"{color_name} ({confidence * 100:.0f}%)"
                cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "???", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Projet RNA - Temps Reel', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()