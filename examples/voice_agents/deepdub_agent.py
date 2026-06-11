import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import deepdub, silero

load_dotenv()

logger = logging.getLogger("deepdub-agent")
logger.setLevel(logging.INFO)

server = AgentServer()


class DeepdubAssistant(Agent):
    """A multilingual voice assistant using Deepdub TTS.

    Deepdub's streaming text-input API segments incremental text server side, and the
    synthesis locale can be switched between replies, which this agent exposes to the
    LLM as a tool so the voice follows the conversation language.
    """

    def __init__(self, tts: deepdub.TTS) -> None:
        self._deepdub_tts = tts
        super().__init__(
            instructions=(
                "You are a friendly voice assistant speaking with a Deepdub voice. "
                "Keep replies short, natural, and conversational. "
                "You can speak many languages. When the user speaks a different language "
                "or asks you to switch, FIRST call set_speech_locale with the matching "
                "locale, then reply in that language."
            )
        )

    @function_tool
    async def set_speech_locale(self, locale: str) -> str:
        """Switch the locale of the speech synthesizer voice.

        Call this before answering when the conversation language changes, so the
        voice pronounces the reply correctly.

        Args:
            locale: BCP-47 locale code, e.g. "en-US", "fr-FR", "de-DE", "es-ES",
                "pt-BR", "he-IL", "ja-JP".
        """
        self._deepdub_tts.update_options(locale=locale)
        logger.info("speech locale switched to %s", locale)
        return f"Speech locale is now {locale}. Reply in this language."


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    tts = (
        deepdub.TTS(voice_prompt_id=voice_prompt_id)
        if (voice_prompt_id := os.environ.get("DEEPDUB_VOICE_PROMPT_ID"))
        else deepdub.TTS()
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=inference.STT(model="assemblyai/universal-streaming-multilingual"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=tts,
    )
    await session.start(agent=DeepdubAssistant(tts), room=ctx.room)
    await session.generate_reply(instructions="Greet the user warmly and briefly.")


if __name__ == "__main__":
    cli.run_app(server)
