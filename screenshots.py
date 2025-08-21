import base64
import time
from threading import Lock, Thread

import cv2
import numpy
from PIL import ImageGrab
from cv2 import imencode

from typing import Optional, Union


class DesktopScreenshot:
    """Capture and provide access to periodic desktop screenshots.

    Screenshots are captured in a background thread and can be retrieved as raw
    numpy arrays or base64-encoded JPEG bytes.
    """

    def __init__(self) -> None:
        """Initialize the screenshot store and synchronization primitives."""
        self.screenshot: Optional[numpy.ndarray] = None
        self.running = False
        self.lock = Lock()
        self.thread: Optional[Thread] = None

    def start(self) -> "DesktopScreenshot":
        """Start the background thread for capturing screenshots.

        Returns:
            The instance of :class:`DesktopScreenshot` for chaining.
        """

        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) -> None:
        """Capture screenshots repeatedly while the thread is running."""
        while self.running:
            screenshot = ImageGrab.grab()
            screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)

            with self.lock:
                self.screenshot = screenshot

            time.sleep(0.1)

    def read(self, encode: bool = False) -> Optional[Union[numpy.ndarray, bytes]]:
        """Retrieve the latest screenshot.

        Args:
            encode: Whether to return a base64-encoded JPEG of the screenshot.

        Returns:
            The screenshot as a numpy array, encoded bytes, or ``None`` if no
            screenshot has been captured yet.
        """

        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None

        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)

        return screenshot

    def stop(self) -> None:
        """Stop the screenshot capture thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
