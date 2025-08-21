import openai
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from pyaudio import PyAudio, paInt16
from langchain_openai import ChatOpenAI

from typing import Optional


class Assistant:
    """Generate conversational responses and speak them out loud.

    The assistant wraps a chat model with conversation history and a text-to-
    speech system to provide audible answers to user prompts.
    """

    def __init__(self, model: ChatOpenAI, voice: str) -> None:
        """Initialize the assistant with the given language model and voice.

        Args:
            model: Chat model used to generate responses.
            voice: Voice name used for text-to-speech.
        """

        self.chain = self._create_inference_chain(model)
        self.voice = voice

    def answer(self, prompt: str, image: Optional[bytes]) -> None:
        """Generate a spoken answer for the provided prompt and image.

        Args:
            prompt: User's text prompt.
            image: Optional base64-encoded screenshot related to the prompt.
        """

        if not prompt:
            return

        print("Prompt:", prompt)

        image_base64 = image.decode() if image is not None else ""

        response = self.chain.invoke(
            {
                "prompt": prompt,
                "image_base64": image_base64,
            },
            config={
                "configurable": {
                    "session_id": "unused"
                }
            },
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response: str) -> None:
        """Convert a text response to speech and play it back.

        Args:
            response: The text to synthesize and play.
        """

        pyaudio_instance = PyAudio()
        stream = None
        try:
            stream = pyaudio_instance.open(
                format=paInt16, channels=1, rate=24000, output=True
            )

            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=self.voice,
                response_format="pcm",
                input=response,
            ) as response_stream:
                for chunk in response_stream.iter_bytes(chunk_size=1024):
                    stream.write(chunk)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pyaudio_instance.terminate()

    def _create_inference_chain(self, model: ChatOpenAI) -> RunnableWithMessageHistory:
        """Create the runnable inference chain used to generate responses.

        Args:
            model: Chat model used to generate responses.

        Returns:
            A runnable that maintains chat history and invokes the model.
        """

        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )
