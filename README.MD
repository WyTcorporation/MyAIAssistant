Ubuntu 24 + python 3.12
sudo apt update -y && sudo apt upgrade -y
sudo apt install portaudio19-dev python3-dev
pip install -r requirements.txt

# 🧠 Desktop AI Assistant with Voice & Visual Input

> ⚡ Real-time voice-activated AI assistant with desktop screenshot context and TTS reply  
> 🇺🇸 English | 🇺🇦 [Ukrainian version below](#українська-версія)

## 🔍 Overview

This project implements an intelligent desktop assistant using:
- 🎙️ Voice recognition via Whisper
- 🖥️ Real-time desktop screenshot using `PIL` + `OpenCV`
- 🧠 GPT-4o response generation using LangChain
- 🔊 Streaming TTS reply using OpenAI's `tts-1` model

The assistant listens to your voice, sees your desktop, and responds with a short, witty spoken message based on both.

---

## 🚀 Features

- **Speech Recognition:** Converts your voice to text using Whisper (`speech_recognition`)
- **Visual Context:** Captures and encodes live screenshots (`PIL.ImageGrab` + `OpenCV`)
- **Multimodal Chat:** Combines text and image into GPT-4o via LangChain pipeline
- **Streaming TTS:** GPT response is spoken back using OpenAI's `tts-1` with streaming PCM output
- **Modular Design:** Extendable for automation, accessibility, and robotics use cases

---

## 🧩 Tech Stack

- Python 3.12
- OpenAI API (GPT-4o, TTS)
- LangChain (`langchain`, `langchain_openai`)
- OpenCV, Pillow, NumPy
- SpeechRecognition + PyAudio
- Threading and real-time capture

---

## 🛠 Installation

Make sure your system has:

```bash
    sudo apt update -y && sudo apt upgrade -y
    sudo apt install portaudio19-dev python3-dev
```

Then install dependencies:

```bash
  pip install -r requirements.txt
```

Create a .env file and add:

``` dotenv
  OPENAI_API_KEY=your_openai_api_key
```

## ▶️ Usage

Run the assistant:

## 📄 License
