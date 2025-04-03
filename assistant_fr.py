import os
import sys
import queue
import sounddevice as sd
import vosk
import json
import subprocess
import numpy as np
# import soundfile as sf # Soundfile might not be strictly needed
import io
import datetime
import time
import requests # For download helper
from tqdm import tqdm # For download helper progress bar
import zipfile # For extraction
import tarfile # For extraction
import traceback # For detailed error printing

# --- Configuration ---
# (Configuration section remains the same as your working version)
VOSK_MODEL_PATH = "vosk-model-small-fr-0.22/vosk-model-small-fr-0.22"
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
VOSK_MODEL_ZIP = "vosk-model-small-fr-0.22.zip"
PIPER_EXE_PATH = "piper_windows_amd64/piper/piper.exe"
PIPER_URL_LINUX = "https://github.com/rhasspy/piper/releases/download/2023.11.14-1/piper_linux_x86_64.tar.gz"
PIPER_URL_WINDOWS = "https://github.com/rhasspy/piper/releases/download/2023.11.14-1/piper_windows_x86_64.zip"
PIPER_DOWNLOAD_URL = PIPER_URL_WINDOWS
PIPER_ARCHIVE_NAME = os.path.basename(PIPER_DOWNLOAD_URL)
PIPER_VOICE_MODEL = "fr-fr-siwis-medium.onnx" # Or fr-FR- if you changed script instead of files
PIPER_VOICE_JSON = "fr-fr-siwis-medium.onnx.json" # Or fr-FR-
PIPER_VOICE_URL_ONNX = None
PIPER_VOICE_URL_JSON = None
INPUT_DEVICE = None
OUTPUT_DEVICE = None
SAMPLE_RATE = 16000
VOSK_SAMPLE_RATE = 16000
PIPER_SAMPLE_RATE = 22050
BLOCK_SIZE = 8000
ASSISTANT_NAME = "Assistant IT" # Changed name slightly

# --- Helper Functions (download_file, extract_archive) ---
# (These functions remain the same)
def download_file(url, filename):
    print(f"Downloading {os.path.basename(filename)} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(filename)}")
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR: Download incomplete.")
            return False
        print(f"{os.path.basename(filename)} downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {os.path.basename(filename)}: {e}")
        return False
    except Exception as e:
         print(f"An unexpected error occurred during download: {e}")
         return False

def extract_archive(archive_path, extract_to='.'):
    print(f"Extracting {archive_path}...")
    extracted_content_path = None
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                 zip_ref.extractall(extract_to)
                 potential_path = os.path.join(extract_to, os.path.splitext(os.path.basename(archive_path))[0])
                 if os.path.isdir(potential_path):
                     extracted_content_path = potential_path
                 else:
                     extracted_content_path = extract_to
        elif archive_path.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                 tar_ref.extractall(path=extract_to)
                 potential_path = os.path.join(extract_to, os.path.splitext(os.path.splitext(os.path.basename(archive_path))[0])[0])
                 if os.path.isdir(potential_path):
                     extracted_content_path = potential_path
                 else:
                     extracted_content_path = extract_to
        else:
            print(f"Unsupported archive format: {archive_path}")
            return None
        print(f"Extracted {archive_path} to '{extract_to}' (content likely within '{extracted_content_path or extract_to}')")
        return extracted_content_path or extract_to
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        print(f"Error extracting {archive_path}: {e}")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during extraction: {e}")
         return None

# --- Prerequisite Checks ---
# (This section remains the same)
print("--- Checking Prerequisites ---")
print(f"Current Working Directory: {os.getcwd()}")
if not os.path.isdir(VOSK_MODEL_PATH):
    print(f"Vosk model not found at '{VOSK_MODEL_PATH}'.")
    # (Add download/extract logic here if needed, simplified for brevity)
    sys.exit("Vosk model missing.")
else:
    print(f"Vosk model found at '{VOSK_MODEL_PATH}'.")
if not os.path.exists(PIPER_EXE_PATH):
    print(f"Piper executable not found at '{PIPER_EXE_PATH}'.")
     # (Add download/extract logic here if needed, simplified for brevity)
    sys.exit("Piper executable missing.")
else:
     print(f"Piper executable found at '{PIPER_EXE_PATH}'.")
print("--- Checking Piper Voice Files ---")
if os.path.exists(PIPER_VOICE_MODEL):
    print(f"Piper voice model found at '{PIPER_VOICE_MODEL}'.")
else:
    print(f"ERROR: Piper voice model (.onnx) *NOT FOUND* at '{PIPER_VOICE_MODEL}'.")
    sys.exit(f"Missing required file: {PIPER_VOICE_MODEL}")
if os.path.exists(PIPER_VOICE_JSON):
    print(f"Piper voice config found at '{PIPER_VOICE_JSON}'.")
else:
    print(f"WARNING: Piper voice config (.json) *NOT FOUND* at '{PIPER_VOICE_JSON}'.")

# --- Initialization ---
print("--- Initialization ---")
try:
    vosk.SetLogLevel(-1)
    stt_model = vosk.Model(VOSK_MODEL_PATH)
    stt_recognizer = vosk.KaldiRecognizer(stt_model, VOSK_SAMPLE_RATE)
    stt_recognizer.SetWords(False)
    print("Vosk STT initialized.")
except Exception as e:
    print(f"Error initializing Vosk STT: {e}")
    traceback.print_exc()
    sys.exit(1)

audio_queue = queue.Queue()

# --- TTS Function (speak) ---
# (This function remains the same)
def speak(text):
    print(f"{ASSISTANT_NAME}: {text}")
    if not os.path.exists(PIPER_EXE_PATH):
        print(f"Error: Piper executable not found at '{os.path.abspath(PIPER_EXE_PATH)}'.")
        return
    if not os.path.exists(PIPER_VOICE_MODEL):
        print(f"Error: Piper voice model not found at '{os.path.abspath(PIPER_VOICE_MODEL)}'.")
        return
    try:
        command = [PIPER_EXE_PATH, '--model', PIPER_VOICE_MODEL, '--output_raw']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = process.communicate(input=text.encode('utf-8'))
        stderr_output = stderr_data.decode('utf-8', errors='ignore')
        if process.returncode != 0:
            print(f"Piper Error Code: {process.returncode}\nPiper stderr:\n{stderr_output}")
            return
        if not stdout_data:
            print("Piper produced no audio output.")
            return
        audio_data = np.frombuffer(stdout_data, dtype=np.int16)
        if audio_data.size == 0:
             print("Warning: Received empty audio data from Piper.")
             return
        sd.play(audio_data, samplerate=PIPER_SAMPLE_RATE, device=OUTPUT_DEVICE)
        sd.wait()
    except Exception as e:
        print(f"Error during TTS processing or playback: {e}")
        traceback.print_exc()


# --- Audio Callback (audio_callback) ---
# (This function remains the same)
def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

# --- STT Function (listen) ---
# (This function remains the same)
def listen():
    print("\nListening...")
    with audio_queue.mutex:
        audio_queue.queue.clear()
    try:
        with sd.RawInputStream(samplerate=VOSK_SAMPLE_RATE, blocksize=BLOCK_SIZE, device=INPUT_DEVICE,
                               dtype='int16', channels=1, callback=audio_callback):
            while True:
                data = audio_queue.get()
                if data:
                    if stt_recognizer.AcceptWaveform(data):
                        result_json = stt_recognizer.Result()
                        result_dict = json.loads(result_json)
                        text = result_dict.get("text", "")
                        if text:
                            sys.stdout.write(" " * 60 + "\r")
                            sys.stdout.flush()
                            print(f"You: {text}")
                            return text.lower().strip()
                    else:
                        partial_result_json = stt_recognizer.PartialResult()
                        partial_result_dict = json.loads(partial_result_json)
                        partial_text = partial_result_dict.get("partial", "")
                        if partial_text:
                            sys.stdout.write(f"Partial: {partial_text}    \r")
                            sys.stdout.flush()
    except KeyboardInterrupt:
        sys.stdout.write(" " * 60 + "\r")
        sys.stdout.flush()
        print("\nStopped listening.")
        stt_recognizer.Reset()
        return "__keyboard_interrupt__"
    except Exception as e:
        sys.stdout.write(" " * 60 + "\r")
        sys.stdout.flush()
        print(f"Error during listening: {e}")
        traceback.print_exc()
        stt_recognizer.Reset()
        return None

# --- ############################################## ---
# --- Command Processing - MODIFIED FOR IT SUPPORT ---
# --- ############################################## ---
def process_command(command):
    """Processes the recognized command and generates an IT support response."""
    if command == "__keyboard_interrupt__":
        return "Au revoir !"

    response = f"Désolé, je ne suis pas sûr de comprendre le problème '{command}'. Pouvez-vous reformuler ?" # Default response
    if not command:
        return "Je n'ai rien entendu. Veuillez répéter votre problème."

    command = command.strip()

    # --- General Commands ---
    if "bonjour" in command or "salut" in command:
        response = "Bonjour ! Décrivez-moi votre problème technique."
    elif "quelle heure" in command or "l'heure" in command:
        now = datetime.datetime.now()
        response = f"Il est {now.hour} heures {now.minute}." # Keep some basic functions
    elif "quelle date" in command or "la date" in command:
        now = datetime.datetime.now()
        mois = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
        jour = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
        response = f"Nous sommes le {jour[now.weekday()]} {now.day} {mois[now.month - 1]} {now.year}."
    elif "qui es tu" in command or "comment tu t'appelles" in command:
         response = f"Je suis {ASSISTANT_NAME}, votre assistant de support technique local."
    elif command in ["merci", "c'est bon", "résolu", "ça marche"]:
         response = "Parfait ! N'hésitez pas si vous avez un autre problème."
         # Optionally return None here too if "merci" should end interaction
    elif command in ["arrête", "au revoir", "quitter", "stop"]:
        response = "Support terminé. Au revoir !"
        return None # Signal to exit the main loop

    # --- IT Support Keywords ---

    # == Printer Issues ==
    elif "imprimante" in command or "imprime pas" in command or "impression" in command:
        if "bloqué" in command or "erreur" in command or "marche pas" in command or "fonctionne pas" in command:
             response = ("Problème d'imprimante détecté. Voici quelques étapes :\n"
                        "1. Vérifiez que l'imprimante est allumée et bien branchée (USB et alimentation).\n"
                        "2. Assurez-vous qu'il y a du papier et de l'encre ou du toner.\n"
                        "3. Essayez de redémarrer l'imprimante et votre ordinateur.\n"
                        "4. Ouvrez la file d'attente d'impression sur votre PC et annulez les travaux bloqués.\n"
                        "5. Essayez d'imprimer une page de test depuis les paramètres Windows de l'imprimante.")
        else:
             response = "Vous avez un souci avec l'impression ? Pourriez-vous préciser ? Par exemple, l'imprimante ne répond pas, ou il y a une erreur ?"

    # == Network / Internet Issues ==
    elif "internet" in command or "wifi" in command or "wi-fi" in command or "connexion" in command or "réseau" in command:
        if "marche pas" in command or "fonctionne pas" in command or "pas de connexion" in command or "aucun accès" in command:
             response = ("Problème de connexion internet. Essayons ceci :\n"
                        "1. Vérifiez si d'autres appareils (téléphone, autre PC) ont accès à internet. Cela permet de savoir si le problème vient de votre PC ou du réseau.\n"
                        "2. Redémarrez votre modem et votre routeur. Débranchez-les pendant 30 secondes, puis rebranchez d'abord le modem, attendez qu'il soit stable, puis le routeur.\n"
                        "3. Redémarrez votre ordinateur.\n"
                        "4. Si vous êtes en Wifi, vérifiez que vous êtes connecté au bon réseau et que le signal est suffisant.\n"
                        "5. Si vous êtes par câble, vérifiez que le câble est bien branché des deux côtés.\n"
                        "Si le problème persiste après ces étapes, contactez votre fournisseur d'accès.")
        elif "lent" in command or "lente" in command:
             response = ("Connexion internet lente ? Voici quelques pistes :\n"
                        "1. Redémarrez votre modem, routeur et ordinateur.\n"
                        "2. Rapprochez-vous de votre routeur Wifi si possible.\n"
                        "3. Vérifiez si des téléchargements lourds ou des mises à jour sont en cours sur votre PC ou d'autres appareils.\n"
                        "4. Trop d'appareils connectés en même temps peuvent ralentir la connexion.")
        else:
             response = "Vous avez un problème de réseau ou d'internet ? Est-ce une absence de connexion, ou une lenteur ?"

    # == Computer Performance Issues ==
    elif ("ordinateur" in command or "pc" in command or "système" in command) and \
         ("lent" in command or "rame" in command or "bloqué" in command or "figé" in command):
        response = ("Ordinateur lent ou bloqué ? Essayons ces actions :\n"
                   "1. La première chose à faire : redémarrez complètement l'ordinateur.\n"
                   "2. Fermez toutes les applications que vous n'utilisez pas activement.\n"
                   "3. Vérifiez si votre disque dur n'est pas presque plein.\n"
                   "4. Assurez-vous que Windows et vos pilotes sont à jour.\n"
                   "5. Vous pouvez ouvrir le Gestionnaire des tâches (Ctrl + Maj + Echap) pour voir si un programme utilise anormalement beaucoup de ressources, mais je ne peux pas le faire pour vous.\n"
                   "6. Pensez à faire une analyse antivirus et anti-malware.")

    # == Software / Bureautique Issues ==
    elif "word" in command or "excel" in command or "outlook" in command or "powerpoint" in command or "office" in command:
         if "ouvre pas" in command or "ne répond pas" in command or "bloqué" in command or "erreur" in command:
              app_name = "Word" if "word" in command else "Excel" if "excel" in command else "Outlook" if "outlook" in command else "PowerPoint" if "powerpoint" in command else "une application Office"
              response = (f"Problème avec {app_name}. Voici des suggestions :\n"
                         f"1. Essayez de fermer complètement {app_name} (via le Gestionnaire des tâches si nécessaire) et de le rouvrir.\n"
                         "2. Redémarrez votre ordinateur.\n"
                         "3. Le problème se produit-il avec un seul fichier ou tous les fichiers de ce type ? Si c'est un seul fichier, il est peut-être corrompu.\n"
                         f"4. Essayez de lancer {app_name} en mode sans échec. Pour cela, cherchez '{app_name} /safe' dans la barre de recherche Windows.\n"
                         "5. Vous pouvez tenter de réparer l'installation d'Office depuis le Panneau de configuration, sous 'Programmes et fonctionnalités'.")
         else:
              response = "Vous rencontrez un souci avec une application Office ? Laquelle et que se passe-t-il exactement ?"

    # == Portability / Laptop specific (basic) ==
    elif "portable" in command or "batterie" in command:
        if "charge pas" in command:
             response = ("Problème de charge de la batterie du portable ?\n"
                        "1. Vérifiez que le chargeur est bien branché à la prise murale et au portable.\n"
                        "2. Essayez une autre prise murale si possible.\n"
                        "3. Vérifiez l'état du câble et du connecteur du chargeur (pas de dommage visible).\n"
                        "4. Redémarrez l'ordinateur portable.\n"
                        "5. Si possible, retirez la batterie (si elle est amovible), nettoyez les contacts, et remettez-la.")
        elif "tient pas" in command or "vide vite" in command:
             response = ("La batterie de votre portable se décharge vite ?\n"
                        "1. Réduisez la luminosité de l'écran.\n"
                        "2. Fermez les programmes gourmands en ressources que vous n'utilisez pas.\n"
                        "3. Déconnectez les périphériques USB non nécessaires.\n"
                        "4. Vérifiez les paramètres d'alimentation de Windows pour optimiser l'autonomie.\n"
                        "Il est normal que les batteries perdent de leur capacité avec le temps.")

    # == Password Issues (General Advice Only) ==
    elif "mot de passe" in command or "compte" in command:
         if "oublié" in command:
              response = ("Mot de passe oublié ? Malheureusement, je ne peux pas le récupérer pour vous. \n"
                         "Utilisez l'option 'Mot de passe oublié' ou 'Réinitialiser le mot de passe' sur le site web ou l'application concernée. \n"
                         "Vérifiez aussi que la touche Verr Maj (Caps Lock) n'est pas activée.")
         elif "bloqué" in command:
              response = ("Compte bloqué ? Cela arrive souvent après trop de tentatives de connexion échouées.\n"
                         "Attendez un peu (parfois 30 minutes ou une heure) avant de réessayer.\n"
                         "Sinon, utilisez l'option 'Mot de passe oublié' ou contactez le support du service concerné.")

    # --- Default fallback if specific IT keywords not matched ---
    # (The default response defined at the beginning of the function will be used)

    return response

# --- Main Loop ---
# (This section remains the same)
if __name__ == "__main__":
    try:
         sd.check_output_settings(device=OUTPUT_DEVICE, samplerate=PIPER_SAMPLE_RATE)
         print("Audio output device check successful.")
    except Exception as e:
         print(f"Warning: Audio output device check failed: {e}. Playback might have issues.")

    try:
        # Changed initial greeting
        speak(f"Bonjour ! Je suis {ASSISTANT_NAME}. Comment puis-je vous assister avec vos problèmes techniques aujourd'hui ?")
        while True:
            command = listen()
            if command is None:
                 speak("Désolé, une erreur s'est produite lors de l'écoute.")
                 time.sleep(2)
                 continue
            if command == "__keyboard_interrupt__":
                 speak("Support terminé. Au revoir !")
                 break
            if command:
                response = process_command(command)
                if response is None: # Exit signal
                    break
                elif response:
                    speak(response)
            else:
                 speak("Je n'ai pas bien entendu. Pouvez-vous décrire votre problème technique ?")

    except KeyboardInterrupt:
        print("\nArrêt de l'assistant demandé.")
    except Exception as e:
        print(f"\nUne erreur majeure et inattendue est survenue: {e}")
        traceback.print_exc()
    finally:
        print("Assistant terminé.")
        sd.stop()