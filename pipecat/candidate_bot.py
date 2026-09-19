"""A second Pipecat voice agent that plays the CANDIDATE side of a screening
call, driven by a test scenario instead of a human at a mic.

Joins the same LiveKit room as bot.py (the recruiter agent) as its own
participant, with its own STT -> LLM -> TTS pipeline. It never talks to
bot.py directly — it hears the recruiter's TTS audio over the room like a
real callee would, replies with its own TTS, and the recruiter's STT picks
that up in turn. This is the same room-based flow real calls and
browser_test.py already use; only who's on the other end changes.

The candidate's persona and required behavior come from a scenario dict (see
scenarios/*.json) — not hardcoded here — so a new test case is a new JSON
file, not a code change.

Does not call end_call or otherwise try to hang up: bot.py (the recruiter)
owns ending the call and deletes the room when it does, which disconnects
this bot's transport and ends its pipeline naturally.
"""
import asyncio
import json
import os
import sys

import pip_system_certs.wrapt_requests  # noqa: F401
from dotenv import load_dotenv
from livekit import api
from loguru import logger

load_dotenv()
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.azure.llm import AzureLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.workers.runner import WorkerRunner

# Behavior-specific instructions, keyed by scenario["persona"]["behavior"].
# Each entry is appended to the shared persona preamble below. Add a new
# scenario type here as new test cases come in — the runner and this file
# don't otherwise need to change per scenario.
BEHAVIOR_INSTRUCTIONS = {
    "not_good_time": """
Your situation on this call: right now is genuinely NOT a good time for you to talk — you're
busy, in a meeting, about to walk into something, or similar. As soon as the recruiter asks
whether now is a good time (or anything equivalent), say clearly that now isn't a good time.
{callback_line}
Do not answer any screening questions (employment status, location, salary, experience, etc.)
even if the recruiter asks one before or instead of the good-time check — politely say you
can't talk right now and would need to do this another time. If they try to continue past that,
repeat, once, that you really can't talk right now. Keep it brief and polite, like a real person
who is legitimately busy — don't be rude.
""",
    "cooperative": """
Answer every screening question the recruiter asks naturally and truthfully, using the facts
listed below. Don't volunteer information for a question they haven't asked yet. Keep answers
conversational and brief, like a real phone call, not a recitation.
""",
    "wrong_number": """
Important override: ignore the name given above — you are NOT {candidate_name}. You're a
different person entirely who just happens to have this phone number now (give yourself a
different first name if it comes up, e.g. "This is Sam"). As soon as the recruiter asks if
you're {candidate_name} (or uses that name at any point), politely correct them: say that's not
you, they might have the wrong number. Do not answer any screening questions. If they apologize
and end the call, accept politely and don't ask questions back or keep the conversation going.
Keep it brief and natural, not annoyed — just an ordinary person clearing up a mix-up.
""",
    "rambling": """
Answer every screening question the recruiter asks naturally and truthfully, using the facts
listed below, with one exception: when they ask you to describe your experience, give a long,
rambling, tangential answer — mention side stories and unrelated details, trail off, circle back
— before eventually landing on the relevant facts. Say the whole rambling answer in one go, only
pausing where a real person naturally would mid-thought, don't stop partway through waiting for
a reaction. For every other question, answer briefly and normally, like a real phone call.
""",
    "asks_question": """
Answer every screening question the recruiter asks naturally and truthfully, using the facts
listed below. Partway through the call — right after they ask about the location or work
arrangement — pause and ask them a question of your own first: "Actually, quick question — who
would I be reporting to on this team?" (this isn't something in the job description, so a good
recruiter should say they'll check and get back to you rather than guessing an answer). Wait for
their reply, then answer their original location/work-arrangement question, and continue
normally with the rest of the screening.
""",
    "volunteers_info": """
Answer every screening question the recruiter asks naturally and truthfully, using the facts
listed below, with one specific twist: when asked whether you're currently employed (the first
substantive screening question), answer: "Yes, I am — and by the way, my notice period is two
weeks." Volunteer that detail unprompted, right there, even though they haven't asked about
notice period yet. Answer every other question normally when they get to it, including if they
briefly confirm the notice period again later — just confirm it, don't repeat the whole story.
""",
}


def build_candidate_prompt(scenario: dict) -> str:
    """The candidate-persona system prompt, built from a scenario dict:
        {"candidate_name", "job_title", "company_name", "persona": {"behavior": "...", ...facts},
         "resume": "... optional full resume text ..."}
    """
    candidate_name = scenario["candidate_name"]
    job_title = scenario["job_title"]
    company_name = scenario["company_name"]
    persona = scenario.get("persona", {})
    behavior = persona.get("behavior", "cooperative")

    behavior_template = BEHAVIOR_INSTRUCTIONS.get(behavior)
    if behavior_template is None:
        raise ValueError(
            f"unknown scenario behavior {behavior!r} — add it to BEHAVIOR_INSTRUCTIONS "
            "in candidate_bot.py"
        )

    callback_time = persona.get("preferred_callback_time")
    callback_line = (
        f'If they ask for a better time to call back, offer: "{callback_time}."'
        if callback_time
        else "If they ask for a better time to call back, offer a vague but real-sounding "
        "one (e.g. \"maybe later this afternoon\" or \"tomorrow morning\")."
    )
    behavior_section = behavior_template.format(callback_line=callback_line, candidate_name=candidate_name)

    facts_lines = []
    for key in (
        "currently_employed",
        "open_to_new_role",
        "location",
        "salary_target",
        "experience_summary",
        "notice_period",
    ):
        if key in persona:
            facts_lines.append(f"- {key.replace('_', ' ')}: {persona[key]}")
    facts_section = ("\n\nFacts about you, use these if asked:\n" + "\n".join(facts_lines)) if facts_lines else ""

    resume_section = f"\n\nYour resume, for reference if asked about your background:\n{scenario['resume']}" if scenario.get("resume") else ""

    return f"""You are {candidate_name}, a real person. You are receiving a phone call from a
recruiter at {company_name} about the {job_title} position. This is a live spoken phone call,
not a chat window — keep every turn short and natural, one or two sentences, the way a real
person talks on the phone.

Wait for the recruiter to speak first — don't say anything until they've said something to you.
Never break character. If directly and explicitly asked whether you are an AI or a bot, answer
honestly, then continue naturally as yourself. Otherwise, always speak and react as
{candidate_name} would, never as an assistant.

Never try to end the call yourself — just respond naturally; the call ends when the recruiter
ends it.
{behavior_section}{facts_section}{resume_section}"""


async def candidate_token(room_name: str, candidate_name: str) -> str:
    return (
        api.AccessToken()
        .with_identity("candidate")
        .with_name(candidate_name)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


async def run_candidate_bot(room_name: str, scenario: dict) -> None:
    transport = LiveKitTransport(
        url=os.environ["LIVEKIT_URL"],
        token=await candidate_token(room_name, scenario["candidate_name"]),
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = AzureLLMService(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        settings=AzureLLMService.Settings(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            max_tokens=150,
        ),
    )
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSSettings(
            # Deliberately a different voice than the recruiter bot's default, so the two
            # sides are distinguishable in a mixed recording. Override via env if needed.
            voice=os.environ.get("CANDIDATE_CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        ),
    )

    context = LLMContext(messages=[{"role": "system", "content": build_candidate_prompt(scenario)}])
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
    # enable_rtvi=False: see the matching comment in bot.py — RTVI is unused here and,
    # left on, floods the shared room with cross-bot messages neither side can parse.
    task = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True), enable_rtvi=False)

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, _participant_id, _reason):
        # The recruiter bot ended the call and deleted the room (or otherwise left) —
        # nothing left to talk to, so stop this pipeline too.
        logger.info(f"[{room_name}] recruiter left/room ending — stopping candidate bot")
        await task.cancel()

    runner = WorkerRunner()
    await runner.add_workers(task)
    await runner.run()
    logger.info(f"[{room_name}] candidate bot finished")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: candidate_bot.py <room-name> <scenario-json-path>")
    with open(sys.argv[2], encoding="utf-8") as f:
        scenario = json.load(f)
    if scenario.get("resume_file") and not scenario.get("resume"):
        with open(scenario["resume_file"], encoding="utf-8") as f:
            scenario["resume"] = f.read()
    asyncio.run(run_candidate_bot(sys.argv[1], scenario))
