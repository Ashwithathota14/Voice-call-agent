"""A minimal Pipecat voice agent that answers a phone call in a LiveKit room.

The phone leg reaches the room through LiveKit SIP and a RingTrunk trunk; this
bot only ever sees a LiveKit room. Speech in, speech out:

    caller -> LiveKit room -> Deepgram STT -> OpenAI LLM -> Cartesia TTS -> caller

Written against Pipecat 1.3 (pipecat.transports.livekit.transport, LLMContext).

Run it for one room:  python bot.py call-abc123
server.py starts it automatically from LiveKit's participant_joined webhook.
"""
import asyncio
import os
import sys

from livekit import api
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

SYSTEM_PROMPT = (
    "You are a friendly phone assistant for a small business in India. "
    "Keep answers short: this is a phone call, not a chat window."
)


def agent_token(room_name: str) -> str:
    """A LiveKit access token that lets the bot join exactly this room."""
    return (
        api.AccessToken()  # reads LIVEKIT_API_KEY / LIVEKIT_API_SECRET
        .with_identity("agent")
        .with_name("Agent")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


async def run_bot(room_name: str) -> None:
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=agent_token(room_name),
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=os.environ.get("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            transport.output(),
            assistant_agg,
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, _participant_id):
        # The caller is in the room: greet them. Appending a message and
        # running the LLM is the reliable way to speak first in Pipecat 1.x.
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content": "Greet the caller and ask how you can help."}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, _participant_id, _reason):
        # The caller hung up: end the pipeline so the room can close.
        await task.cancel()

    await PipelineRunner().run(task)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: bot.py <room-name>")
    asyncio.run(run_bot(sys.argv[1]))
