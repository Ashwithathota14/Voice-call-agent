"""Place one outbound recruiter-screening call and run the screening — one command.

Creates the LiveKit room with candidate/JD metadata attached, then does two
things at once: joins the room as the bot (loading its speech models and
connecting to Deepgram/Groq/Cartesia) WHILE the phone is still ringing, and
dials the candidate through your RingTrunk SIP outbound trunk. By the time
they pick up, the bot is already sitting in the room fully warmed up — its
greeting fires the instant they join instead of ~10s after, which is what
happens if that warmup only starts after pickup. No webhook, no tunnel, no
second command needed — that path (server.py's `participant_joined` webhook)
is still there for a real deployed server, but for running a call from the
command line this is simpler and doesn't depend on a webhook reaching your
machine.

Usage:
    python dial.py +919876543210 \\
        --candidate "Dinesh" \\
        --job-title "Business Analyst" \\
        --company "Navitas Business Consulting" \\
        --jd-file job_descriptions/job_description.txt \\
        --resume-file resumes/dinesh.txt

Or omit the phone number entirely and let it find one in the resume:
    python dial.py \\
        --candidate "Dinesh" \\
        --job-title "Business Analyst" \\
        --company "Navitas Business Consulting" \\
        --jd-file job_descriptions/job_description.txt \\
        --resume-file resumes/dinesh.txt
This only ever finds an Indian mobile number (10 digits, starting 6-9) —
that's a hard limit of the RingTrunk trunk itself (destinations are Indian
mobiles only, see the repo README), not something this script chooses to
restrict. A resume with no such number, or only a non-Indian one, fails
loudly instead of guessing.

Screening script picks up US-specific questions (work authorization, clearance,
degree) automatically for +1 numbers, or force it with --country US|IN.

Requires env vars: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
OUTBOUND_TRUNK_ID (the RingTrunk trunk id, looks like "ST_xxxxxxxx" — find it
with: lk sip outbound list). CALLER_ID (optional) sets the number shown to
the person you're calling.
"""
import argparse
import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv
from livekit import api

load_dotenv()

from bot import run_bot


def extract_phone_number(resume_text: str) -> str | None:
    """Find a callable Indian mobile number in a resume's text and normalize
    it to E.164 (+91XXXXXXXXXX). Matches "+91 62818 48886", "+91-6281848886",
    "916281848886", "06281848886", or a bare "6281848886" — spaces/dashes/dots
    between digits are stripped before matching. Returns None if nothing in
    the text reduces to a plausible Indian mobile number (10 digits starting
    6-9) — RingTrunk's outbound trunk can't dial anything else anyway, so
    there's no point guessing at a non-Indian number here.
    """
    for raw in re.findall(r"[\d][\d\s\-.]{8,14}\d", resume_text):
        digits = re.sub(r"\D", "", raw)
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        elif digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) == 10 and digits[0] in "6789":
            return f"+91{digits}"
    return None


async def create_call_room(call_context: dict) -> str:
    room_name = f"call-{int(time.time())}"
    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        await lkapi.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=json.dumps(call_context))
        )
    return room_name


async def dial_sip_participant(room_name: str, phone_number: str, candidate_name: str) -> None:
    """Blocks until the callee actually picks up (or rejects/times out)."""
    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=room_name,
                sip_trunk_id=os.environ["OUTBOUND_TRUNK_ID"],
                sip_call_to=phone_number,
                sip_number=os.environ.get("CALLER_ID") or None,  # caller ID shown to the callee
                participant_identity="candidate",
                participant_name=candidate_name,
                wait_until_answered=True,
            )
        )


async def delete_room(room_name: str) -> None:
    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))


async def dial_and_screen(phone_number: str, call_context: dict) -> None:
    room_name = await create_call_room(call_context)

    # Start the bot joining and warming up (VAD/turn model load, Deepgram/Groq/
    # Cartesia connect) at the same time as dialing, not after — run_bot() only
    # actually speaks once on_first_participant_joined fires, so it's safe to
    # have it sitting in the room, fully warmed up, before the candidate ever
    # answers. That's what turns "~10s of dead air after pickup" into "bot
    # greets the instant they pick up".
    bot_task = asyncio.create_task(run_bot(room_name))
    print(f"Dialing {phone_number} (bot warming up in parallel)...", flush=True)

    try:
        await dial_sip_participant(room_name, phone_number, call_context["candidate_name"])
    except Exception as exc:
        bot_task.cancel()
        await delete_room(room_name)
        raise RuntimeError(f"call to {phone_number} did not connect: {exc}") from exc

    print("Answered — screening underway.\n", flush=True)
    await bot_task
    print(f"\nCall finished. Result saved under results/{room_name}.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phone_number",
        nargs="?",
        default=None,
        help="E.164 format, e.g. +919876543210 — omit to dial the number found in --resume-file",
    )
    parser.add_argument("--candidate", required=True, help="Candidate's name")
    parser.add_argument("--job-title", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--jd-file", required=True, help="Path to a text file with the full JD")
    parser.add_argument("--resume-file", help="Candidate's resume (optional) — bot probes its claims")
    parser.add_argument(
        "--country",
        choices=["IN", "US"],
        help="Screening script variant. Default: inferred from the phone number's "
        "country code (+1 -> US, everything else -> IN). US adds work-authorization, "
        "clearance, and degree questions (see bot.py US_SCREENING_QUESTIONS).",
    )
    args = parser.parse_args()

    with open(args.jd_file, encoding="utf-8") as f:
        job_description = f.read()

    resume_text = None
    if args.resume_file:
        with open(args.resume_file, encoding="utf-8") as f:
            resume_text = f.read()

    phone_number = args.phone_number
    if not phone_number:
        if not resume_text:
            parser.error("phone_number is required when --resume-file isn't given")
        phone_number = extract_phone_number(resume_text)
        if not phone_number:
            parser.error(
                "no phone number given, and couldn't find an Indian mobile number "
                f"(10 digits, starting 6-9) in {args.resume_file!r} — pass phone_number "
                "explicitly instead"
            )
        print(f"No phone number given — dialing the number found in the resume: {phone_number}")

    country = args.country or ("US" if phone_number.startswith("+1") else "IN")

    call_context = {
        "candidate_name": args.candidate,
        "job_title": args.job_title,
        "company_name": args.company,
        "job_description": job_description,
        "country": country,
        # Real phone audio (RingTrunk/SIP, 8kHz G.711) needs a different STT
        # model than a browser mic — see bot.py's nova-2-phonecall handling.
        "audio_channel": "phone",
    }
    if resume_text:
        call_context["resume"] = resume_text

    try:
        asyncio.run(dial_and_screen(phone_number, call_context))
    except RuntimeError as exc:
        # place_call's own error (call not answered/rejected) — already a clear
        # message, no traceback needed.
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
