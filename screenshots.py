import base64
import time
from threading import Lock, Thread

import cv2
import numpy
from PIL import ImageGrab
from cv2 import imencode


class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = ImageGrab.grab()
            screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)

            with self.lock:
                self.screenshot = screenshot

            time.sleep(0.1)

    def read(self, encode: bool = False):
        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None

        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)

        return screenshot

    def stop(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join()
