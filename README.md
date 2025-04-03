# 🤖 French Voice Assistant with Vosk and Piper

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![Vosk](https://img.shields.io/badge/Vosk-STT-orange.svg)
![Piper](https://img.shields.io/badge/Piper-TTS-green.svg)

An offline French voice assistant that uses Vosk for speech recognition and Piper for text-to-speech. This assistant provides IT support responses in French.

## ✨ Features

- 🎤 Voice recognition with Vosk (French language model)
- 🔊 Natural-sounding speech synthesis with Piper
- 💻 IT support responses in French
- 🔒 Completely offline - no internet required after setup

## 📋 Requirements

- Python 3.6 or higher
- Vosk speech recognition library
- Piper TTS engine
- French language models for both Vosk and Piper

## 🚀 Setup

1. Clone this repository:
   ```
   git clone https://github.com/AminePro7/french-voice-assistant.git
   cd french-voice-assistant
   ```

2. Install Python dependencies:
   ```
   pip install vosk sounddevice numpy tqdm requests
   ```

3. Download required models:
   - Vosk French model: [vosk-model-small-fr-0.22](https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip)
   - Piper French voice: [fr-fr-siwis-medium](https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR/siwis/medium)
   - Piper executable: [Piper releases](https://github.com/rhasspy/piper/releases)

4. Place the models in the appropriate directories as specified in the script configuration.

## 🎮 Usage

Run the assistant with:
```
python assistant_fr.py
```

The assistant will initialize the speech recognition and text-to-speech systems, then wait for voice commands. Speak in French to interact with the assistant.

## 📝 Command Examples

- "Bonjour" - Greets the user
- "Quelle heure est-il?" - Tells the current time
- "Quelle est la date?" - Tells the current date
- "Qui es-tu?" - Explains what the assistant is

## 📚 Resources

- [Vosk Models](https://alphacephei.com/vosk/models)
- [Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/main/fr/fr_FR/siwis/medium)
- [Piper Releases](https://github.com/rhasspy/piper/releases)

## 👤 Author

Built by [AminePro7](https://github.com/AminePro7)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 