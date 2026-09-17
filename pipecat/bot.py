"""A Pipecat voice agent that runs an outbound recruiter screening call in a
LiveKit room.

The phone leg reaches the room through LiveKit SIP and a RingTrunk trunk; this
bot only ever sees a LiveKit room. Speech in, speech out:

    caller -> LiveKit room -> Deepgram STT -> Azure OpenAI LLM -> Cartesia TTS -> caller

Who the bot is calling and which job it's screening for is not hardcoded: it's
read from the room's metadata (set when the call/dispatch was created), as
JSON: {"candidate_name": "...", "job_title": "...", "company_name": "...",
"job_description": "... full JD text ..."}. The system prompt is built from
that at call start, so the recruiter script and JD questions change per call.

Written against Pipecat 1.8 (pipecat.transports.livekit.transport, LLMContext).

Run it for one room:  python bot.py call-abc123
server.py starts it automatically from LiveKit's participant_joined webhook.
"""
import asyncio
import functools
import json
import os
import re
import sys
import wave

import pip_system_certs.wrapt_requests  # noqa: F401  trust Windows CA store (corp TLS-inspecting proxy breaks certifi)
from dotenv import load_dotenv
from livekit import api
from loguru import logger
from openai import AsyncAzureOpenAI

load_dotenv()
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndWorkerFrame,
    LLMMessagesAppendFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.idle_frame_processor import IdleFrameProcessor
from pipecat.services.azure.llm import DATED_API_VERSION, AzureLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService, CartesiaTTSSettings
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport
from pipecat.workers.runner import WorkerRunner

def transcript_from_messages(messages: list[dict]) -> str:
    """Render an LLMContext message list (system/user/assistant/tool) as a plain dialogue."""
    lines = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        speaker = "Candidate" if role == "user" else "Recruiter (AI)"
        lines.append(f"{speaker}: {content.strip()}")
    return "\n".join(lines)


SALARY_EXTRACTION_PROMPT = """From this phone screening transcript, extract ONLY the salary \
figure the candidate stated for themselves (their target/expected/current salary or rate) — \
not any salary range mentioned by the recruiter, and not a number from the job description. \
This is pure extraction: do not judge, evaluate, or compare the figure to anything.

Respond with ONLY a JSON object, no other text.

If the candidate stated a figure, in any form ("13 LPA", "around 50k a month", "120 to 150 \
thousand a year", "9 lakhs"):
{{
  "as_stated": "<the figure roughly as they said it>",
  "amount": <the plain number they gave, e.g. 13 for "13 LPA", 50 for "50k">,
  "unit": "<one of: 'LPA' (lakhs per annum), 'lakh' (a bare lakh figure with no per-annum said), \
'thousand', 'k', 'plain' (already a full number like 130000)>",
  "period": "<one of: 'annual', 'monthly', 'hourly', 'unclear'>",
  "currency": "<your best-guess currency from context and phrasing, e.g. 'USD', 'INR' — \
'unclear' if you genuinely can't tell>",
  "annualized_amount_same_currency": <the amount converted to a plain annual figure, IN THE \
SAME CURRENCY THEY STATED — e.g. "13 LPA" -> 1300000, "50k a month" -> 600000, an already-annual \
plain figure -> unchanged. Never convert between currencies — leave that to a human.>
}}

If the candidate never stated any salary figure of their own anywhere in the transcript:
{{"as_stated": null}}

TRANSCRIPT:
{transcript}
"""


def _extract_json_object(text: str) -> dict:
    """Models sometimes wrap JSON in prose, a code fence, or a <think> reasoning block
    despite instructions; salvage it."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


async def extract_salary_info(transcript: str) -> dict | None:
    """One-shot LLM call: pull out whatever salary figure the candidate stated on the call
    and normalize it into a plain {as_stated, amount, unit, period, currency,
    annualized_amount_same_currency} shape, so review doesn't have to mentally parse "13 LPA"
    or "50k a month" out of the raw transcript. Pure extraction, not evaluation — no judgment
    on whether the number is reasonable, no cross-currency conversion (a stale/approximate FX
    rate baked into a prompt would misinform review, so this deliberately leaves that to a
    human). Returns None if the candidate never stated a figure, or if extraction fails for any
    reason — this is a bonus data point, not a required part of the call.
    """
    try:
        client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", DATED_API_VERSION),
        )
        response = await client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": SALARY_EXTRACTION_PROMPT.format(transcript=transcript),
                }
            ],
            temperature=0,
            max_tokens=300,
        )
        result = _extract_json_object(response.choices[0].message.content)
    except Exception:
        logger.exception("salary extraction failed — leaving it out of the saved result")
        return None

    if not result.get("as_stated"):
        return None
    return result


# Neither RingTrunk nor LiveKit's SIP integration exposes answering-machine detection
# (checked both: no AMD/voicemail/call-progress field anywhere in LiveKit's
# CreateSIPParticipantRequest, and RingTrunk's own docs say plainly they do no audio
# analysis on calls at all) — so this is a same-call STT heuristic, not a carrier
# signal. Applied ONLY to the very first candidate utterance after the bot's opening
# line (see VoicemailDetector below); never re-checked later in the call, so a long
# answer to a later question can't retroactively trigger this.
VOICEMAIL_PHRASES = (
    "leave a message",
    "leave your message",
    "after the tone",
    "after the beep",
    "at the tone",
    "at the beep",
    "voice mailbox",
    "voicemail",
    "voice mail",
    "mailbox is full",
    "not available to take your call",
    "unable to take your call",
    "record your message",
    "press pound",
    "press the pound",
    "press star",
    "currently unavailable",
    "leave your name and number",
)
# A human picking up says "Hello?" or "This is <name>" — a handful of words. A
# voicemail greeting is a full uninterrupted paragraph. This alone (without a phrase
# match) is a weaker signal, so the threshold is set high to keep false positives on a
# genuinely chatty human low — tune here if real calls show it's off in either
# direction.
VOICEMAIL_LONG_MONOLOGUE_WORD_THRESHOLD = 30


def looks_like_voicemail(text: str) -> bool:
    """True if this first response has the shape of a voicemail/answering-machine
    greeting rather than a human answering the phone. Heuristic, not certain — see
    VOICEMAIL_PHRASES/VOICEMAIL_LONG_MONOLOGUE_WORD_THRESHOLD above."""
    lowered = text.lower()
    if any(phrase in lowered for phrase in VOICEMAIL_PHRASES):
        return True
    return len(text.split()) >= VOICEMAIL_LONG_MONOLOGUE_WORD_THRESHOLD


def build_system_prompt(call: dict) -> str:
    """The recruiter-screening script, filled in with this call's candidate and JD.

    `call` is the room-metadata JSON: candidate_name, job_title, company_name,
    job_description (full JD text), optionally resume (full resume text — when
    given, the bot probes its claims instead of asking from scratch), and
    optionally candidate_location (if not given, the bot just asks for it
    instead of confirming an already-known one).

    Unlike the old fixed-question script, mandatory screening questions beyond
    the base ones (employment status, location, salary, experience, notice
    period) are NOT hardcoded here — the model derives them itself from
    whichever requirements the job description below actually marks as
    mandatory/required, per the JOB DESCRIPTION REQUIREMENTS section. This
    call never produces a fit score or a go/no-go verdict; it only collects
    answers for a human recruiter to review.
    """
    candidate_name = call["candidate_name"]
    company_name = call["company_name"]
    job_title = call["job_title"]
    job_description = call["job_description"]

    pronunciation_note = ""
    if company_name.strip().lower() == "navitas business consulting":
        pronunciation_note = '\n• "Navitas" is pronounced "nah-VEE-tahs."'

    candidate_location = call.get("candidate_location")
    if candidate_location:
        location_step = (
            f"3. I have your location listed as {candidate_location}. Is that still "
            "correct? If incorrect, ask for the updated location and confirm it."
        )
    else:
        location_step = "3. What's your current location?"

    resume_section = ""
    resume_step = ""
    if call.get("resume"):
        resume_section = f"\n\nCandidate's resume on file:\n{call['resume']}"
        resume_step = (
            "Where the candidate's resume (below) claims something relevant to the "
            "job description — a skill, a tool, years on a technology — ask a brief "
            "question that checks whether they can actually speak to it, instead of "
            "asking generically. Don't read the resume back at them.\n"
        )

    return f"""You are SCOUT, a professional Recruiting Coordinator for {company_name}.
IMPORTANT{pronunciation_note}
• Speak naturally and conversationally.
• Sound like a real recruiter.
• Never sound scripted.
• Never rush.

YOUR ROLE
You are conducting an initial recruiter screening for the {job_title} position.
This is NOT an interview.
Your only responsibility is to collect information.
You are NOT responsible for deciding whether someone is qualified.
You are NOT responsible for determining fit.
Never compare the candidate to the job requirements.
Never tell the candidate they have too much or too little experience.
Never suggest they may not qualify.
Never reject a candidate.
Never discourage a candidate.
Never recommend another position.
Simply collect information and allow the recruiter to make all hiring decisions. This call
produces no fit score, no verdict, no go/no-go — that happens later, by a human, not you.

LISTENING
Only ask ONE question at a time.
Wait until the candidate completely finishes speaking.
Do not interrupt.
Ignore brief pauses, background conversations, dogs, keyboard sounds, TV, traffic.
If unsure whether the candidate has finished speaking, wait another second.
If both people begin talking simultaneously, immediately stop talking.
Never guess what someone said. If their answer is vague, mumbled, cut off, or hard to parse —
they DID respond, this is not silence — ask one brief clarifying follow-up on that same
question. Never end the call and never move to end_call because an answer was unclear; only a
genuine lack of any response should ever lead toward ending the call.

STAYING ON TRACK
If the candidate's answer conflicts with something you already asked about (e.g. they say
they'd only want remote work after you asked about a hybrid requirement, or they say they lack
a mandatory qualification), acknowledge it plainly ONCE ("Understood, I'll make a note of
that.") and move straight to the next question. Do not ask the same question again in
different words, do not keep circling back to it, and do not repeat "I'll note that" more than
once for the same point — that reads as broken, not attentive, and it's what makes candidates
disengage and hang up. You are still not evaluating or rejecting them by doing this — you are
just not getting stuck.

NAMES
The candidate's name is {candidate_name}. Use exactly that name.
Never substitute another name, never guess another name, never change the pronunciation.
After greeting them once, avoid repeating their name.
If at any point they say they're not actually {candidate_name}, apologize briefly and call
end_call with reason "wrong_number" right after — don't continue the screening.

JOB TITLES
If the job title contains numbers or requisition IDs, do NOT slowly read every digit — treat
them as an internal reference and naturally say the actual job title. If the number is
difficult to pronounce, omit the numeric identifier and simply say the job title.

JOB DESCRIPTION REQUIREMENTS
Before conducting the screening, review the job description below. Identify requirements
explicitly stated as mandatory, required, must have, must be, or equivalent mandatory wording.
These may include: candidate location, job location, onsite/hybrid/remote requirement,
relocation requirement, travel requirement, US citizenship, work authorization, visa
sponsorship restrictions, Green Card/permanent residency, security clearance, ability to
obtain security clearance, professional certifications, education requirements, years of
experience, required technical skills, required domain experience, employment type, shift or
schedule requirements, or any other mandatory requirement explicitly mentioned.

For EVERY mandatory requirement you identify, ask the candidate an appropriate screening
question in step 8 below. Do NOT use a fixed list — generate the questions dynamically from
the requirements in THIS job description. If a requirement is only listed as preferred,
desired, nice-to-have, bonus, or a plus, do NOT treat it as mandatory and do NOT ask about it.
Do NOT ask about a requirement that isn't mentioned in the job description at all.

OPENING
This is a real phone call, not a chat window: keep every turn short, spoken, and natural.
Say, adjusting only for natural spoken flow: "Hi, am I speaking with {candidate_name}?" Then
stop and wait for their answer.
If they confirm it's them, say: "I'm calling from {company_name} regarding the {job_title}
position you recently applied for. Thank you for applying. This is just a brief recruiter
screening and should only take about five minutes. Is now a good time to talk?" Then stop and
wait.
If they are busy, driving, in a meeting, or only have a minute, say exactly, in the same turn,
with no question in between and without waiting for any further reply: "No problem at all.
Thank you for your time. We'll reach out another time. Have a wonderful day. Goodbye." Then
call end_call with reason "not_available" immediately after — in the same turn, right after
that line, don't continue into the screening, and don't let the call just go quiet without
calling end_call.
This "not a good time" handling applies for the whole call, not just the opening — if the
candidate says at ANY point they need to stop, can't talk right now, or need to reschedule, say
that exact same line immediately and call end_call with reason "not_available" right after —
don't try to finish the remaining questions first, and don't stop before actually calling
end_call.

If at any point the candidate says something like "stop calling me," "take me off your list,"
"don't contact me again," or otherwise clearly asks not to be contacted — this is different
from simply being busy. Stop immediately, do not ask any more questions, do not try to
continue the screening. Say: "Understood, I'll make sure you're taken off our contact list.
Sorry for the inconvenience, and have a good day." Then call end_call with reason
"do_not_contact" right after.

If the candidate answers question 2 (open to a new opportunity) with a clear "no" — they are
not interested in exploring anything new — do not continue into the rest of the screening
order. Say: "No problem at all, I appreciate you letting me know. Have a great day." Then call
end_call with reason "not_interested" right after.

BOT IDENTITY
If the candidate asks whether you're a real person or an AI/bot, answer honestly — don't deny
it and don't dodge. Say something like: "I'm an AI recruiting assistant — I handle the initial
screening so our recruiters can spend their time with candidates who are a good fit. A real
recruiter will follow up with you after this." Then continue naturally from where you left off.

DIFFICULT MOMENTS
If the candidate is hard to hear (soft-spoken, poor connection, background noise drowning them
out), say so plainly and ask them to repeat or speak up rather than guessing: "Sorry, you're a
little hard to hear — could you say that again?" or "Would you mind speaking up a bit?" If audio
quality stays too poor to continue after a couple of tries, offer to reschedule and end the call
with reason "not_available".
If the candidate becomes hostile, defensive, or pushes back sharply (e.g. when asked about
credentials or eligibility), do not match their tone or get defensive yourself. De-escalate
briefly ("Not at all, I just want to make sure I have the right information for the team") and
continue the screening calmly.
If the candidate discloses something that may be protected information (health/disability,
pregnancy, family status, religion, etc.) — even unprompted — do not ask follow-up questions
about it, do not record it as an answer to any screening question, and briefly reassure them:
"Thanks for sharing that — just so you know, that's not something we ask about or factor into
this process." Then return to wherever you were in the screening order.
If the candidate tries to negotiate salary, title, or terms during this call (e.g. pushing back
on a number, asking you to match a competing offer), do not negotiate — that's not your role.
Say something like: "Got it, I'll pass that along to the recruiter." Record what they said and
move on; never counter-offer, never agree to a number, never say whether it's acceptable.
If the candidate mentions they're currently interviewing elsewhere or holding other offers,
simply acknowledge it neutrally ("Good to know, thanks for sharing that") and record it — don't
react, don't try to sell them on this role, don't ask for details about the other company.
If the candidate makes an off-color joke or comment, do not laugh along, echo it, or scold them
— stay professional and businesslike, and move the conversation back to the next question.

SCREENING ORDER — ask ONE question at a time, wait for the full answer each time, never
combine two questions into one, never comment on or evaluate an answer, never re-ask a
question already answered earlier in the call. If an answer volunteers information that
answers a LATER question before you get there (e.g. they mention their notice period while
answering the employment question, or state a number that answers the salary question early),
record it silently and don't ask that later question again verbatim — a brief one-line
confirmation when you reach that point is fine ("And just to confirm, your notice period is
two weeks?"), a fresh full question is not.
1. Are you currently employed? — this is ONLY about their current employment status (working
   or not). Track it as its own separate fact.
2. Would you be open to exploring a new opportunity? — this is a SEPARATE, independent question
   from question 1; a "not employed" answer to question 1 is not itself an answer to this one,
   and vice versa. If their answer to question 1 already clearly states their interest too
   (e.g. "No, I'm not employed, and I'm looking for opportunities" or "No, I'm actually happily
   employed and not looking"), don't re-ask this as a fresh question — briefly confirm it
   instead ("Great, and just to confirm — you're open to exploring new roles?") and move on.
{location_step}
4. Determine the job's location/work arrangement from the job description below (onsite,
   relocation required, remote, or hybrid) and ask only the one matching question:
   - onsite: "This position is located in {{job location}}. Would you be comfortable
     commuting to this location?"
   - relocation required: "This position is located in {{job location}}. Would you be
     comfortable relocating if needed?"
   - remote: "This position is listed as remote. Would you be comfortable working remotely?"
   - hybrid: "This position is based in {{job location}} and follows a hybrid work
     arrangement. Would you be comfortable with that arrangement?"
5. What salary range are you targeting? If they answer in a unit other than plain annual USD
   (e.g. "13 LPA", "50k a month", a different currency), don't just repeat the raw number back
   — briefly restate it in your own words the way you understood it (e.g. "So that's about 13
   lakhs per year, is that right?") so it's unambiguous in the recording, then move on once
   confirmed.
6. Can you briefly tell me about your experience in your field? Listen, acknowledge, don't
   interview or challenge them, don't compare it with the job.
7. If the job description contains a mandatory years-of-experience or specific-experience
   requirement, ask: "How many years of professional experience do you have with {{required
   skill or field}}?" Simply record the answer — never say whether it meets the requirement.
{resume_step}8. Ask the mandatory JD-specific screening questions you identified above, one at a
   time, each as its own separate question — never combine two requirements into one
   question. Simply record each answer, never comment on it. Examples of phrasing:
   - US citizenship mandatory: "Are you a US citizen?"
   - work authorization mandatory: "May I know your current work authorization?"
   - sponsorship restricted: "Would you require sponsorship now or in the future?"
   - clearance mandatory: "Do you currently hold an active {{clearance level}} security
     clearance?"
   - must be able to obtain clearance: "Would you be able to obtain the required {{clearance
     level}} security clearance if needed?"
   - travel mandatory: "Are you comfortable with the travel requirements for this position?"
     (or, with a stated percentage: "...traveling up to {{travel percentage}}...")
   - relocation mandatory: "Would you be willing to relocate to {{job location}} for this
     position?"
   - certification mandatory: "Do you currently hold the required {{certification}}
     certification?"
   - education mandatory: "Do you have a {{required degree or qualification}}?"
   - specific skill mandatory: "How much professional experience do you have with {{required
     skill}}?"
   - schedule/shift mandatory: "Would you be comfortable working the {{required schedule or
     shift}} for this position?"
9. If they said (question 1) they're currently employed, ask: "What is your notice period for
   leaving your current position?" If they're NOT currently employed, ask this instead, as its
   natural substitute — don't skip straight past without asking anything: "How soon would you
   be able to start?"
10. Do you have any questions for me? Answer factual questions only. If unsure, say: "I'll
    make a note so one of our recruiters can follow up."

ADDITIONAL QUESTIONS
If additional questions are supplied below (outside the job-description requirements), ask
them naturally. Do NOT interpret the answers, do NOT compare them against the job description,
do NOT evaluate the candidate — simply collect the response and continue.

TIME
The whole call must fit under 7 minutes, and every applicable question above must be asked in
THIS call — there is no follow-up call, so nothing gets deferred. If time is running short,
shorten follow-ups first rather than dropping a mandatory question.

If the candidate asks you something you can't answer from this job description or from what's
given here (e.g. exact team, manager, precise start date, benefits details, interview process
specifics), say plainly that you'll check with the hiring team and get back to them — never
guess or make up an answer.

ENDING
Do NOT say this goodbye and do NOT call end_call until you have actually asked every
applicable question in the SCREENING ORDER above (1 through 10, including every mandatory
JD-specific question from step 8) and the candidate has answered each one. Getting a normal
answer to an early question (like question 1) is not a reason to wrap up — it means move to
the next question, nothing else. The one exception is a clear "no" to question 2 (open to a new
opportunity) — see the not_interested handling above, which ends the call early on purpose. If
you're unsure whether you've covered everything, that means you haven't — keep going instead of
closing early.
Once every question has actually been asked and answered, say exactly: "{GOODBYE_LINE}"
Immediately after this goodbye line, call end_call with reason "screening_complete" — call it
exactly once, right after you say goodbye, and don't keep talking after calling it. Never say
this goodbye line, or any version of it, more than once in the call.

STRICT RULES
• Ask only ONE question at a time. Always wait for the candidate to finish before the next one.
• Never ask two questions together. Never interrupt the candidate.
• Never make hiring decisions, never reject a candidate, never compare them to the job
  description, never comment on whether they're qualified, never mention required years of
  experience as a judgment.
• Never substitute another name, never invent information, never guess what the candidate
  meant — ask for clarification instead.
• Always screen every requirement explicitly marked mandatory/required in the job description.
  Never treat preferred/desired/nice-to-have/bonus/plus requirements as mandatory. Never ask
  about a requirement not mentioned in the job description. Never assume citizenship, work
  authorization, sponsorship, clearance, travel, relocation, certification, education, or any
  other requirement that isn't explicitly stated.
• Avoid duplicate questions — if a mandatory requirement was already answered earlier in the
  conversation, don't ask it again.
• Whenever a question doesn't apply and you skip it (not currently employed, already answered,
  not a mandatory JD requirement, etc.), skip it silently — never tell the candidate you're
  skipping a question or explain why. Just move straight to the next applicable question.
• Keep the call under 7 minutes. Maintain a warm, professional, conversational tone throughout.

Job description for {job_title}:
{job_description}{resume_section}

Reminder: you are the recruiter calling FROM {company_name}. You are not {candidate_name} and
must never speak as them or adopt their voice — everything above the resume is your own
briefing, not something you say."""


GOODBYE_LINE = "Thanks so much for your time today — have a wonderful day!"

# Rough floor on how many times the bot must have actually spoken before a
# "screening_complete" end_call is allowed to succeed (name check, good-time
# check, then at minimum the base screening questions). Confirmed necessary on
# a real call where the model called end_call with zero goodbye content after
# only 4 questions (employed/opportunity/location/salary) — a model eagerness
# quirk, not something prompt wording alone reliably stops. "wrong_number",
# "not_available", and "silence_timed_out" are exempt — those are legitimately
# short calls, and for silence_timed_out specifically: the candidate has
# already gone silent through two check-ins (see on_user_idle below), so
# rejecting that end_call and forcing the bot to keep talking would just have
# it ask questions into dead air instead of hanging up.
MIN_ASSISTANT_TURNS_BEFORE_END = 8
END_CALL_TURN_FLOOR_EXEMPT_REASONS = {
    "wrong_number",
    "not_available",
    "silence_timed_out",
    "do_not_contact",
    "voicemail",
    "not_interested",
}


async def end_call(params: FunctionCallParams, call_outcome: dict, candidate_spoke: dict) -> None:
    """Hang up — after the closing goodbye (reason="screening_complete"), after
    apologizing for a wrong-number/not-the-right-person mix-up (reason="wrong_number"),
    after the candidate said it's not a good time or asked to reschedule
    (reason="not_available"), after the candidate asked not to be contacted again
    (reason="do_not_contact"), after the candidate said they're not open to a new
    opportunity (reason="not_interested"), after leaving a callback message on voicemail
    (reason="voicemail"), or after two silent check-ins went unanswered
    (reason="silence_timed_out") — all but screening_complete exempt from the turn
    floor below since those are legitimately short calls.

    `call_outcome` is a mutable dict the caller reads after the pipeline stops, so the
    actual end reason (not just "the call ended") reaches the saved transcript/results —
    downstream review needs to know a no-answer looks nothing like a real decline.

    `candidate_spoke` is a mutable dict set True the moment a real TranscriptionFrame
    (actual mic audio) is seen (see MarkCandidateSpoke below) — NOT the same as a
    role="user" message existing in params.context.messages, because the opening
    trigger and idle-nudge instructions this bot injects are themselves role="user"
    (Groq's chat template needs a user turn to exist at all), so the turn-floor check
    above can't tell "candidate actually spoke" from "we injected an instruction."
    Confirmed necessary on a real call where the model called end_call(reason=
    "wrong_number") on the very first turn, having never actually spoken the opening
    line or heard anything — it hallucinated a name mismatch out of nothing."""
    reason = params.arguments.get("reason")
    call_outcome["end_reason"] = reason
    if reason != "silence_timed_out" and not candidate_spoke["flag"]:
        logger.warning(
            f"end_call({reason!r}) called before the candidate ever actually spoke — "
            "rejecting and nudging the LLM to actually say the opening line and wait"
        )
        await params.result_callback(
            {"success": False, "reason": "you haven't actually heard the candidate say anything yet"}
        )
        await params.llm.push_frame(
            LLMMessagesAppendFrame(
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "You have not actually heard the candidate say anything yet — do not "
                            "end the call for any reason until you have said your opening line out "
                            "loud and gotten a real spoken reply. Say the OPENING line now (or "
                            "continue it if you already started) and wait for their answer."
                        ),
                    }
                ],
                run_llm=True,
            )
        )
        return
    if reason not in END_CALL_TURN_FLOOR_EXEMPT_REASONS:
        messages = params.context.messages if params.context else []
        goodbye_already_said = any(
            m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and GOODBYE_LINE in m["content"]
            for m in messages
        )
        if goodbye_already_said:
            # The goodbye line is already out — accepting here instead of rejecting
            # again is what actually prevents a repeat; a rejection nudge relies on
            # the model obeying "don't say it again," which it isn't reliable at.
            await params.result_callback({"success": True})
            await params.llm.push_frame(EndWorkerFrame())
            return
        spoken_turns = sum(
            1
            for m in messages
            if m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
            and m["content"].strip()
        )
        if spoken_turns < MIN_ASSISTANT_TURNS_BEFORE_END:
            logger.warning(
                f"end_call({reason!r}) called after only {spoken_turns} spoken turns — "
                "too early, rejecting and nudging the LLM to continue the screening"
            )
            await params.result_callback(
                {"success": False, "reason": "too early — the screening isn't finished yet"}
            )
            await params.llm.push_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "That was too early to end the call — you have not asked every "
                                "applicable question from the SCREENING ORDER yet. If you already "
                                "said a goodbye line out loud, do NOT say it again and do NOT "
                                "apologize or acknowledge the mix-up to the candidate — just "
                                "continue naturally. Look back at this conversation to see which "
                                "questions you've actually already asked and gotten an answer to, "
                                "then ask the next one from the script that you have NOT asked "
                                "yet — never a question already answered earlier in this call."
                            ),
                        }
                    ],
                    run_llm=True,
                )
            )
            return
    await params.result_callback({"success": True})
    await params.llm.push_frame(EndWorkerFrame())  # downstream: lets the goodbye audio finish first


def build_end_call_tool(call_outcome: dict, candidate_spoke: dict) -> FunctionSchema:
    """`call_outcome` and `candidate_spoke` are per-call mutable state (see end_call's
    docstring) — this factory closes over them so each call's end_call handler reads/
    writes that call's own dicts, not shared module-level ones."""
    return FunctionSchema(
        name="end_call",
        description=(
            "End the phone call. Call this once, immediately after either your closing "
            "goodbye line, or your apology for a wrong-number/not-the-right-person mix-up."
        ),
        properties={
            "reason": {
                "type": "string",
                "enum": [
                    "screening_complete",
                    "wrong_number",
                    "not_available",
                    "silence_timed_out",
                    "do_not_contact",
                    "not_interested",
                    "voicemail",
                ],
                "description": (
                    "'screening_complete' after your normal closing goodbye. "
                    "'wrong_number' after apologizing because this isn't the right person. "
                    "'not_available' after the candidate said it's not a good time to talk or "
                    "needs to reschedule. 'do_not_contact' after the candidate explicitly asked "
                    "not to be contacted/called again. 'not_interested' after the candidate said "
                    "they're not open to a new opportunity (question 2). 'voicemail' — only ever used when you "
                    "were explicitly told you've reached voicemail/an answering machine, right "
                    "after leaving the callback message; never choose this yourself. "
                    "'silence_timed_out' — only ever used when you were explicitly told the "
                    "candidate stayed silent through two check-ins; never choose this yourself."
                ),
            }
        },
        required=["reason"],
        handler=functools.partial(end_call, call_outcome=call_outcome, candidate_spoke=candidate_spoke),
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


async def fetch_call_context(room_name: str) -> dict:
    """Candidate + job info attached as this room's metadata when the call was
    dispatched, e.g.:
        {"candidate_name": "Dinesh", "job_title": "...", "company_name": "...",
         "job_description": "... full JD text ..."}
    """
    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        rooms = await lkapi.room.list_rooms(api.ListRoomsRequest(names=[room_name]))
    if not rooms.rooms or not rooms.rooms[0].metadata:
        raise RuntimeError(
            f"room {room_name!r} has no metadata — expected candidate_name/job_title/"
            "company_name/job_description JSON set when the call was dispatched"
        )
    return json.loads(rooms.rooms[0].metadata)


async def run_bot(room_name: str) -> None:
    call = await fetch_call_context(room_name)

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

    # nova-2-phonecall: Deepgram's model tuned for narrowband telephone audio
    # (8kHz, G.711) instead of the default general-purpose model tuned for
    # clean mic audio. Without this, real phone calls only transcribe short
    # loud exclamations ("Hello?") and drop full sentences — confirmed on a
    # real RingTrunk test call where the candidate's actual answers never
    # reached the LLM even though VAD correctly detected them speaking.
    #
    # Only for real phone calls, though (dial.py sets audio_channel="phone").
    # Applied to a browser_test.py mic call instead, it over-segments clean
    # full-band audio into many tiny fragments — confirmed on a browser test
    # where one spoken answer came through as 10+ separate "Candidate:" turns,
    # each one re-triggering the LLM and causing repeated/overlapping replies.
    stt_settings = (
        DeepgramSTTSettings(model="nova-2-phonecall")
        if call.get("audio_channel") == "phone"
        else DeepgramSTTSettings()
    )
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], settings=stt_settings)
    # Azure OpenAI deployment. AZURE_OPENAI_ENDPOINT is the resource endpoint from the
    # Azure portal / AI Foundry — a regional endpoint like
    # https://eastus.api.cognitive.microsoft.com/ routes through the dated
    # AZURE_OPENAI_API_VERSION below; a resource endpoint ending in /openai/v1 instead
    # uses Azure's newer v1 surface and ignores api_version entirely. AZURE_OPENAI_DEPLOYMENT
    # is the deployment name you created in Azure (not the base model name) — check the
    # resource's "Model deployments" tab if unsure, "gpt-4o-mini" is just a fallback default.
    llm = AzureLLMService(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        settings=AzureLLMService.Settings(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            max_tokens=150,  # recruiter turns are one short spoken sentence — plenty
        ),
    )
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSSettings(
            voice=os.environ.get("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22"),
        ),
    )

    # How this call actually ended — set by end_call (see build_end_call_tool) once
    # the model calls it, or left "unknown" if the candidate hangs up first
    # (on_participant_left below) without the bot ever getting to call it. Read after
    # the pipeline stops and saved into results/ so downstream review can tell a
    # no-answer/silence-timeout apart from a real decline instead of every ended call
    # looking the same.
    call_outcome = {"end_reason": "unknown"}

    # Set True the moment real mic audio is transcribed (see MarkCandidateSpoke below) —
    # end_call uses this to refuse to hang up on any reason before the candidate has
    # actually said anything, since the injected opening-trigger/idle-nudge messages are
    # themselves role="user" in params.context.messages and can't be told apart from
    # real candidate speech by looking at that alone.
    candidate_spoke = {"flag": False}

    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(call)}],
        tools=[build_end_call_tool(call_outcome, candidate_spoke)],
    )
    user_agg, assistant_agg = LLMContextAggregatorPair(context)

    # Records the call locally (mixed user + bot audio, mono) — no cloud storage
    # account needed, since bot.py is just a room participant and sees the audio
    # directly. Placed after transport.output() so it sees both directions: user
    # mic audio flowing downstream from transport.input(), and bot TTS audio
    # flowing downstream from transport.output().
    audiobuffer = AudioBufferProcessor()

    # If the candidate goes silent for 30s (no TranscriptionFrame — nothing
    # transcribed from their mic), nudge them. Nudge twice; if they're still
    # silent after the second nudge, end the call. idle_retries is a mutable
    # container (not a plain int) so the closure below can update it in place.
    idle_retries = {"count": 0}
    # Once we've told the bot to say goodbye and hang up, stop reacting to further
    # idle timeouts — EndWorkerFrame takes a moment to actually stop the pipeline
    # (goodbye audio has to finish playing first), and if the candidate is still
    # silent during that gap another 30s can tick over and fire this callback
    # again, queuing a second goodbye before the first one's done.
    call_ending = {"flag": False}
    # Set True once the candidate actually joins (see on_first_participant_joined).
    # bot.py now typically joins the room and starts this pipeline BEFORE the
    # candidate picks up (dial.py starts it while the phone is still ringing, so
    # the bot's already warmed up when they answer) — without this guard, a long
    # ring could let the 30s idle timeout fire into an empty room and, after two
    # rounds, end the call before anyone ever joined it.
    call_started = {"flag": False}

    async def on_user_idle(_processor: IdleFrameProcessor) -> None:
        if not call_started["flag"] or call_ending["flag"]:
            return
        idle_retries["count"] += 1
        if idle_retries["count"] > 2:
            call_ending["flag"] = True
            logger.warning(f"[{room_name}] candidate silent through 2 reprompts — ending call")
            await task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "The candidate has not responded even after being checked on "
                                    "twice. Politely say you're unable to continue since there's no "
                                    "response, and that you'll follow up another time. Then call "
                                    "end_call with reason \"silence_timed_out\" right after."
                                ),
                            }
                        ],
                        run_llm=True,
                    )
                ]
            )
        else:
            await task.queue_frames(
                [
                    LLMMessagesAppendFrame(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "The candidate hasn't responded in a while. Briefly check if "
                                    "they're still there, then repeat your last question."
                                ),
                            }
                        ],
                        run_llm=True,
                    )
                ]
            )

    # Resets on UserStartedSpeakingFrame (VAD: they've started talking) as well as
    # TranscriptionFrame (STT finished transcribing) — watching only the final
    # transcript let the 30s timer expire mid-answer on a real phone call, where a
    # candidate pausing a few seconds before answering plus normal STT finalization
    # lag can eat most of that window before the transcript ever arrives, making the
    # bot nudge/interrupt someone who is actively talking, not silent.
    idle_processor = IdleFrameProcessor(
        callback=on_user_idle,
        # Configurable per-call so agent-vs-agent test scenarios (candidate_bot.py via
        # run_scenario.py) can give the candidate bot's own LLM+TTS round-trip latency
        # more room than a real candidate's speaking pause needs — real calls never set
        # this, so they keep the same 30s as before.
        timeout=call.get("idle_timeout_secs", 30),
        types=[TranscriptionFrame, UserStartedSpeakingFrame],
    )

    # Only ever check the very first candidate utterance after the bot's opening line —
    # never again later in the call (see looks_like_voicemail's docstring above for why).
    # Only meaningful on a real phone call; browser_test.py's mic input isn't voicemail.
    voicemail_check = {"pending": call.get("audio_channel") == "phone"}

    class VoicemailDetector(FrameProcessor):
        """Swallows the first TranscriptionFrame instead of letting it reach the LLM
        normally if it looks like a voicemail greeting — otherwise the model would just
        try to have a normal conversation with an answering machine. Injects its own
        instruction to leave a callback message and hang up instead."""

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if (
                voicemail_check["pending"]
                and call_started["flag"]
                and isinstance(frame, TranscriptionFrame)
            ):
                voicemail_check["pending"] = False  # only ever check once, match or not
                if looks_like_voicemail(frame.text):
                    logger.warning(
                        f"[{room_name}] first response looks like voicemail — "
                        f"leaving a callback message instead of screening: {frame.text!r}"
                    )
                    await task.queue_frames(
                        [
                            LLMMessagesAppendFrame(
                                messages=[
                                    {
                                        "role": "user",
                                        "content": (
                                            "You appear to have reached voicemail or an "
                                            f"answering machine, not {call['candidate_name']} "
                                            "directly. Do not continue with the opening "
                                            "question or any screening questions. Instead, "
                                            "leave a brief, natural callback message: say "
                                            f"who you're calling from ({call['company_name']}), "
                                            f"that you're trying to reach "
                                            f"{call['candidate_name']} regarding the "
                                            f"{call['job_title']} position, and ask them to "
                                            "call back at their convenience. Keep it under 20 "
                                            "seconds of speech. Then call end_call with reason "
                                            "\"voicemail\" immediately after."
                                        ),
                                    }
                                ],
                                run_llm=True,
                            )
                        ]
                    )
                    return  # swallow — don't also push the voicemail greeting downstream
            await self.push_frame(frame, direction)

    class ResetIdleRetriesOnSpeech(FrameProcessor):
        """The candidate spoke — they're not unresponsive, so clear the strike
        count. Otherwise one reply sandwiched between two silent stretches would
        wrongly count toward the 2-strike end-call limit instead of each silent
        stretch getting its own two chances."""

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                idle_retries["count"] = 0
            await self.push_frame(frame, direction)

    class MarkCandidateSpoke(FrameProcessor):
        """Sets candidate_spoke["flag"] the moment any real mic audio is transcribed —
        placed before VoicemailDetector so it fires even on a frame VoicemailDetector
        swallows (a voicemail greeting is still real audio, not silence). This is the
        only reliable "did the candidate actually say anything" signal end_call has —
        see end_call's docstring for why params.context.messages can't be used for
        this instead."""

        async def process_frame(self, frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame) and frame.text.strip():
                candidate_spoke["flag"] = True
            await self.push_frame(frame, direction)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            idle_processor,
            MarkCandidateSpoke(),
            VoicemailDetector(),
            ResetIdleRetriesOnSpeech(),
            user_agg,
            llm,
            tts,
            transport.output(),
            audiobuffer,
            assistant_agg,
        ]
    )
    task = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(_buffer, audio: bytes, sample_rate: int, num_channels: int):
        # Fires once, with the full call's audio, when stop_recording() runs below
        # (buffer_size=0 means no periodic chunks — just this one final dump).
        os.makedirs("results", exist_ok=True)
        wav_path = os.path.join("results", f"{room_name}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)  # 16-bit PCM, matches the raw audio frames
            wf.setframerate(sample_rate)
            wf.writeframes(audio)
        logger.info(f"[{room_name}] recording saved to {wav_path}")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(_transport, _participant_id):
        # Guard against a duplicate opening greeting: LiveKit can fire this event more
        # than once for the same room (e.g. the other side's connection blips and
        # briefly reconnects) — without this guard, a re-fire re-queues the whole
        # OPENING-trigger message and the bot greets twice back to back. Confirmed on
        # an agent-vs-agent test run where the candidate bot's own connection warmup
        # caused exactly this.
        if call_started["flag"]:
            return
        call_started["flag"] = True
        await audiobuffer.start_recording()

        # The caller is in the room: greet them. Appending a message and
        # running the LLM is the reliable way to speak first in Pipecat 1.x.
        # The actual opening script lives in the OPENING section of the system
        # prompt (build_system_prompt) — this just triggers it.
        #
        # role must be "user", not "system": with only the main system prompt
        # ahead of it, a second system message leaves the context with no user
        # turn at all, and some chat templates (qwen3.6 on Groq, confirmed)
        # flat-out reject that — "400: No user query found in messages" —
        # right at the start of every call.
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Open the call now, following the OPENING section of your "
                                "instructions exactly."
                            ),
                        }
                    ],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_participant_left")
    async def on_participant_left(_transport, _participant_id, _reason):
        # The caller hung up: save the recording, then end the pipeline so the
        # room can close. If end_call never ran first (this fired before the bot's
        # own goodbye), this is the candidate hanging up on their own — a real,
        # distinct outcome from anything the bot decided, so record it as such
        # rather than leaving end_reason at "unknown".
        if call_outcome["end_reason"] == "unknown":
            call_outcome["end_reason"] = "candidate_hung_up"
        await audiobuffer.stop_recording()
        await task.cancel()

    runner = WorkerRunner()
    await runner.add_workers(task)
    await runner.run()

    # Safety net: if the bot ended the call itself (end_call tool) rather than
    # the candidate hanging up first, on_participant_left may not have fired.
    # stop_recording() is a no-op if already stopped, so this is safe either way.
    await audiobuffer.stop_recording()

    # The bot leaving the room (above) does NOT hang up the candidate's phone — in
    # LiveKit SIP, the callee's leg stays connected (silently) until the room itself
    # closes or their SIP participant is explicitly removed. Without this, every call
    # the bot ends itself (goodbye, not_available, do_not_contact, etc.) leaves the
    # candidate's phone sitting connected with nothing happening, even though our
    # side considers the call over. Deleting the room drops every remaining
    # participant, including the SIP leg, so their phone actually hangs up.
    try:
        async with api.LiveKitAPI(
            os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
        ) as lkapi:
            await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception:
        # The candidate may have already hung up and the room may already be gone —
        # that's fine, not an error. Still log it in case it's something else.
        logger.warning(f"[{room_name}] deleting room after call end failed (may already be gone)")

    logger.info(f"[{room_name}] call ended, reason={call_outcome['end_reason']!r}")
    try:
        await save_transcript(room_name, call, context.messages, call_outcome["end_reason"])
    except Exception:
        # Saving the transcript is a bonus on top of a finished call — never let it
        # look like the call itself failed. Full traceback still goes to the log.
        logger.exception(f"[{room_name}] saving transcript failed; call itself completed fine")


async def save_transcript(room_name: str, call: dict, messages: list[dict], end_reason: str) -> None:
    """After the call ends, save the raw transcript alongside the call metadata and how
    the call actually ended, for a human recruiter to review. No fit score, no verdict,
    no go/no-go — this call only collects what was said; deciding what to do with it is
    a human decision, made outside this bot.

    end_reason is one of the end_call tool's reasons (screening_complete, wrong_number,
    not_available, do_not_contact, silence_timed_out), "candidate_hung_up" (they left the
    room without the bot ever calling end_call), or "unknown" (neither happened —
    shouldn't normally occur). Keeping these distinct matters for review: a
    silence_timed_out or candidate_hung_up call that only got through a few questions is
    an incomplete call, not a candidate who answered everything and was simply screened
    out.

    Also extracts a normalized salary figure (see extract_salary_info) when the candidate
    stated one — pure data extraction, not a fit judgment, so it doesn't reintroduce
    scoring.
    """
    transcript = transcript_from_messages(messages)
    # Require an actual candidate turn, not just the bot's own greeting — otherwise
    # "joined and left without saying anything" still saves a transcript with nothing
    # useful in it.
    candidate_spoke = any(
        m.get("role") == "user" and isinstance(m.get("content"), str) and m["content"].strip()
        for m in messages
    )
    if not candidate_spoke:
        logger.warning(f"[{room_name}] candidate never spoke on this call — skipping transcript save")
        return

    salary = await extract_salary_info(transcript)

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"{room_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"call": call, "transcript": transcript, "end_reason": end_reason, "salary": salary},
            f,
            indent=2,
        )

    print(f"\n[{room_name}] transcript saved (end_reason={end_reason!r}) -> {out_path}\n", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: bot.py <room-name>")
    asyncio.run(run_bot(sys.argv[1]))
