"""Run one agent-vs-agent test scenario: bot.py (recruiter) and candidate_bot.py
(candidate persona) both join a fresh LiveKit room and talk to each other — no
browser, no phone, no human on either end.

Each bot runs as its OWN process (not two pipelines in one asyncio event loop) —
each pipeline is doing real-time audio I/O (STT/LLM/TTS), and running both in one
process starves them of scheduling time and causes STT dropouts/stutter under
load. This is also how it works for real calls: dial.py/server.py always launch
bot.py as its own process.

Usage:
    python run_scenario.py scenarios/not_good_time.json
    python run_scenario.py scenarios/not_good_time.json --listen   # also opens a
                                                                    # browser tab so
                                                                    # you can hear
                                                                    # both bots live

After both bots finish, prints PASS/FAIL against the scenario's
expected_end_reason and the full transcript, and points at the saved
results/<room_name>.json for a closer look (or `python show_result.py <room_name>`).
"""
import argparse
import asyncio
import json
import os
import sys
import time
import urllib.parse
import webbrowser

from dotenv import load_dotenv
from livekit import api

load_dotenv()


def listener_join_url(room_name: str) -> str:
    """Join link for a silent listener: subscribes to both bots' audio but can't
    publish, so it never interferes with the recruiter's STT (no mic feedback
    picked up as a third voice in the room)."""
    token = (
        api.AccessToken()
        .with_identity("listener")
        .with_name("Listener")
        .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish=False))
        .to_jwt()
    )
    return "https://meet.livekit.io/custom?" + urllib.parse.urlencode(
        {"liveKitUrl": os.environ["LIVEKIT_URL"], "token": token}
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_file", help="Path to a scenarios/*.json file")
    parser.add_argument(
        "--listen", action="store_true",
        help="Open a browser tab as a silent listener so you can hear the two bots talk live",
    )
    args = parser.parse_args()

    with open(args.scenario_file, encoding="utf-8") as f:
        scenario = json.load(f)

    with open(scenario["jd_file"], encoding="utf-8") as f:
        job_description = f.read()

    resume_text = None
    if scenario.get("resume_file"):
        with open(scenario["resume_file"], encoding="utf-8") as f:
            resume_text = f.read()

    room_name = f"scenario-{int(time.time())}"
    call_context = {
        "candidate_name": scenario["candidate_name"],
        "job_title": scenario["job_title"],
        "company_name": scenario["company_name"],
        "job_description": job_description,
        # The candidate bot has its own LLM+TTS round-trip latency on every turn (a real
        # candidate just talks) — 30s is tight enough that a normal reply can eat most of
        # it, so the recruiter's idle-nudge can fire mid-reply. Give scenario runs more
        # slack; real calls never set this and keep the default 30s.
        "idle_timeout_secs": scenario.get("idle_timeout_secs", 60),
    }
    if resume_text:
        call_context["resume"] = resume_text

    async with api.LiveKitAPI(
        os.environ["LIVEKIT_URL"], os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    ) as lkapi:
        await lkapi.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=json.dumps(call_context))
        )

    print(f"Room: {room_name}")
    print(f"Test case: {scenario.get('test_case', '(none given)')}")
    print(f"Expected:  {scenario.get('expected', '(none given)')}")

    if args.listen:
        join_url = listener_join_url(room_name)
        print(f"\nListener join link (opening now):\n\n{join_url}\n", flush=True)
        webbrowser.open(join_url)

    print("Launching recruiter + candidate bots as separate processes...\n", flush=True)

    # Launched together, not staggered: the candidate bot needs several seconds of its
    # own warmup (Deepgram/Azure/Cartesia connects) before it's actually ready to be
    # greeted, same as the recruiter bot does — staggering them just stacks both
    # warmups back to back into one long silence before the recruiter's opening line.
    # Starting both now overlaps that warmup instead.
    here = os.path.dirname(os.path.abspath(__file__))
    recruiter_proc = await asyncio.create_subprocess_exec(
        sys.executable, "bot.py", room_name, cwd=here,
    )
    candidate_proc = await asyncio.create_subprocess_exec(
        sys.executable, "candidate_bot.py", room_name, args.scenario_file, cwd=here,
    )

    await asyncio.gather(recruiter_proc.wait(), candidate_proc.wait())

    result_path = os.path.join("results", f"{room_name}.json")
    if not os.path.exists(result_path):
        print(f"\nNo result saved at {result_path} — the call may have ended before "
              "the candidate ever spoke (check the logs above).")
        sys.exit(1)

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    expected_end_reason = scenario.get("expected_end_reason")
    actual_end_reason = result.get("end_reason")
    end_reason_passed = expected_end_reason is None or actual_end_reason == expected_end_reason

    print(f"\n=== end_reason check: {'PASS' if end_reason_passed else 'FAIL'} ===")
    print(f"Expected end_reason: {expected_end_reason!r}")
    print(f"Actual end_reason:   {actual_end_reason!r}")
    print("\n--- Transcript ---\n")
    print(result.get("transcript", "(no transcript saved)"))
    print(f"\nFull result: {result_path}")

    # end_reason matching only proves the call terminated the way it was supposed to — it
    # says nothing about whether the recruiter's behavior along the way was actually right
    # (right questions, nothing it shouldn't have said, etc). The judge reads the transcript
    # against the scenario's test_case/expected text for that; see judge.py.
    from judge import judge_call, print_verdict

    verdict = judge_call(scenario, result)
    print_verdict(verdict)

    judge_path = result_path.rsplit(".json", 1)[0] + ".judge.json"
    with open(judge_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print(f"Judge verdict saved: {judge_path}")

    overall_passed = end_reason_passed and verdict.get("verdict") == "PASS"
    print(f"\n=== OVERALL: {'PASS' if overall_passed else 'FAIL'} ===")

    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
