import numpy as np

# --- FONCTION DE NORMALISATION ---
def normalize_landmarks(landmarks):
    """
    Convertit la liste des points MediaPipe en une liste de 42 valeurs (x, y)
    centrées sur le poignet et normalisées par la taille de la main.
    """
    #1. Convertir en tableau numpy pour faciliter les calculs
    coords = np.array([[lm.x, lm.y] for lm in landmarks])

    #2. Centrer sur le poignet (Point0)
    wrist = coords[0]
    coords = coords - wrist #soustraction vectorielle

    #3. Mettre à l'échelle (invariance à la distance)
    #on trouve la distance maximal entre le poignet et n'importe quel point
    max_value = np.max(np.abs(coords))
    if max_value > 0:
        coord = coords / max_value

    #4. Aplatir en une liste simple [x1, y1, x2, y2, ...]
    return coords.flatten().tolist()