import logging
import cv2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from speech_recognition import Microphone, Recognizer, UnknownValueError
from openai import OpenAIError

from assistant import Assistant
from screenshots import DesktopScreenshot


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    desktop_screenshot = DesktopScreenshot().start()
    model = ChatOpenAI(model="gpt-4o")
    assistant = Assistant(model)

    def audio_callback(recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
            image = desktop_screenshot.read(encode=True)
            if image is None:
                logger.info("Skipping response: screenshot not available.")
                return
            assistant.answer(prompt, image)
        except UnknownValueError:
            logger.error("There was an error processing the audio.")
        except OpenAIError as e:
            logger.error(f"OpenAI error: {e}")
        except OSError as e:
            logger.error(f"I/O error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

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
