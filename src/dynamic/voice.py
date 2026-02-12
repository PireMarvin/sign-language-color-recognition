import pyttsx3
import threading

# Initialisation du moteur vocal
engine = pyttsx3.init()


def speak(text):
    """Fonction pour parler sans bloquer le programme"""

    def _run():
        try:
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            # Sécurité si le moteur est déjà en train de parler
            pass

    thread = threading.Thread(target=_run)
    thread.start()