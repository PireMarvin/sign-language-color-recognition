import cv2
import mediapipe as mp

# 1. Initialisation de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration :
# static_image_mode=False (car on est en vidéo)
# max_num_hands=1 (pour l'instant, on se concentre sur une main pour faire simple comme dans ton projet)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 2. Ouverture de la webcam (0 est généralement la cam par défaut)
cap = cv2.VideoCapture(0)

print("Appuie sur 'q' pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture caméra")
        break

    # 3. Prétraitement de l'image
    # OpenCV lit en BGR, MediaPipe a besoin de RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inverser l'image horizontalement pour un effet miroir (plus naturel)
    image_rgb = cv2.flip(image_rgb, 1)

    # Faire la détection
    results = hands.process(image_rgb)

    # On repasse en BGR pour l'affichage OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # 4. Extraction des données
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner le squelette sur l'image
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Afficher les coordonnées du bout de l'index (Point 8) pour tester
            # Les coordonnées sont normalisées entre 0 et 1 par rapport à la taille de l'image
            index_tip = hand_landmarks.landmark[8]
            print(f"Index Tip -> x: {index_tip.x:.2f}, y: {index_tip.y:.2f}")

    # 5. Affichage
    cv2.imshow('Projet RNA - Vision', image_bgr)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()