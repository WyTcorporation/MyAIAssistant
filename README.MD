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

```bash
  python3 main.py
```

## 📄 License

MIT License — Free to use, modify and distribute.

# 🇺🇦 Українська версія

--- 

## 🔍 Огляд

Цей проєкт реалізує інтелектуального асистента для десктопу:
- 🎙️ Голосове розпізнавання через Whisper
- 🖥️ Скриншоти екрану в реальному часі (PIL + OpenCV)
- 🧠 Генерація відповідей GPT-4o через LangChain
- 🔊 Відповідь у голосовій формі через OpenAI tts-1 (потокове відтворення)

Асистент чує ваш голос, бачить ваш екран і відповідає стисло, дотепно та голосом — з урахуванням контексту.

## 🚀 Можливості
- Розпізнавання мови: Перетворює голос на текст (Whisper API)
- Візуальний контекст: Захоплення і кодування скриншотів (PIL.ImageGrab + OpenCV)
- Мультимодальний чат: Поєднання тексту і зображення через LangChain + GPT-4o
- Озвучення: Стрімінгова генерація аудіо через OpenAI tts-1
- Модульність: Легко розширюється під робототехніку, доступність та автоматизацію

---

## 🧩 Стек технологій
- Python 3.12
- OpenAI API (GPT-4o, TTS)
- LangChain (langchain, langchain_openai)
- OpenCV, Pillow, NumPy
-  SpeechRecognition + PyAudio

Потокове оброблення та багатопоточність

## 🛠 Встановлення
Підготуйте систему:
```bash
    sudo apt update -y && sudo apt upgrade -y
    sudo apt install portaudio19-dev python3-dev
```

Встановіть залежності:
```bash
  pip install -r requirements.txt
```

Створіть файл .env та додайте свій ключ API:
```dotenv
  OPENAI_API_KEY=your_openai_api_key
```
## ▶️ Запуск
Запустіть асистента:

```bash
  python3 main.py
```

## 📄 Ліцензія
MIT — вільне використання, модифікація та поширення.