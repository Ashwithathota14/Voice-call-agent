"""Test the recruiter bot from a browser — no phone/RingTrunk needed.

Creates a LiveKit room with candidate/JD metadata (same shape bot.py expects
from a real dispatched call), starts bot.py against that room, and prints a
join link for LiveKit's hosted web client so you can talk to it with your
mic/speakers.

Usage:
    python browser_test.py \\
        --candidate "Dinesh" \\
        --job-title "Business Analyst" \\
        --company "Navitas Business Consulting" \\
        --jd-file sample_jd.txt

Requires env vars: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
(same .env bot.py already uses).
"""
import argparse
import asyncio
import json
import os
import time
import urllib.parse
import webbrowser

from dotenv import load_dotenv
from livekit import api

from bot import run_bot

load_dotenv()


def browser_token(room_name: str) -> str:
    return (
        api.AccessToken()
        .with_identity("tester")
        .with_name("Tester")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--job-title", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--jd-file", required=True)
    parser.add_argument("--resume-file", help="Candidate's resume (optional) — bot probes its claims")
    parser.add_argument(
        "--country",
        choices=["IN", "US"],
        default="IN",
        help="Screening script variant — US adds work-authorization, clearance, and "
        "degree questions (see bot.py US_SCREENING_QUESTIONS). Default: IN.",
    )
    args = parser.parse_args()

    with open(args.jd_file, encoding="utf-8") as f:
        job_description = f.read()

    room_name = f"test-{int(time.time())}"
    call_context = {
        "candidate_name": args.candidate,
        "job_title": args.job_title,
        "company_name": args.company,
        "job_description": job_description,
        "country": args.country,
    }
    if args.resume_file:
        with open(args.resume_file, encoding="utf-8") as f:
            call_context["resume"] = f.read()

    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        await lkapi.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=json.dumps(call_context))
        )

    token = browser_token(room_name)
    join_url = "https://meet.livekit.io/custom?" + urllib.parse.urlencode(
        {"liveKitUrl": os.environ["LIVEKIT_URL"], "token": token}
    )

    print(f"Room: {room_name}", flush=True)
    print(f"Join link (opening in your browser now):\n\n{join_url}\n", flush=True)
    webbrowser.open(join_url)
    print(f"After the call ends, run:  python show_result.py {room_name}\n", flush=True)
    print("Starting bot... (Ctrl+C to stop)", flush=True)

    await run_bot(room_name)


if __name__ == "__main__":
    asyncio.run(main())
