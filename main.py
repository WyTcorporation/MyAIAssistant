import cv2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from speech_recognition import Microphone, Recognizer, UnknownValueError

from assistant import Assistant
from screenshots import DesktopScreenshot


load_dotenv()


def main():
    desktop_screenshot = DesktopScreenshot().start()
    model = ChatOpenAI(model="gpt-4o")
    assistant = Assistant(model)

    def audio_callback(recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            image = desktop_screenshot.read(encode=True)
            if image is None:
                print("Skipping response: screenshot not available.")
                return
            assistant.answer(prompt, image)
        except UnknownValueError:
            print("There was an error processing the audio.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    recognizer = Recognizer()
    microphone = Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback)

    while True:
        screenshot = desktop_screenshot.read()
        if screenshot is not None:
            cv2.imshow("Desktop", screenshot)
        if cv2.waitKey(1) in [27, ord("q")]:
            break

    desktop_screenshot.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)


if __name__ == "__main__":
    main()
