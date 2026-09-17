"""Show a finished call's saved transcript.

bot.py writes one JSON file per call to results/<room_name>.json right after
the call ends (candidate + JD metadata, and the call transcript — no fit
score or verdict; that's a human recruiter's call, not this bot's). This just
reads one back and prints it readably.

Usage:
    python show_result.py test-1788944011      # a specific room
    python show_result.py --latest             # most recently finished call
"""
import argparse
import glob
import json
import os


def find_latest() -> str:
    files = glob.glob(os.path.join("results", "*.json"))
    if not files:
        raise SystemExit("No results yet — results/ is empty. Finish a call first.")
    return max(files, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("room_name", nargs="?", help="Room name, e.g. test-1788944011")
    parser.add_argument("--latest", action="store_true", help="Show the most recently finished call")
    args = parser.parse_args()

    if args.latest or not args.room_name:
        path = find_latest()
    else:
        path = os.path.join("results", f"{args.room_name}.json")
        if not os.path.exists(path):
            raise SystemExit(f"No result at {path} yet — has that call finished?")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    call = data["call"]

    print(f"\n=== {call['candidate_name']} — {call['job_title']} ({os.path.basename(path)}) ===\n")
    print(f"Company:       {call.get('company_name')}")
    print(f"End reason:    {data.get('end_reason', 'unknown')}")
    if call.get("resume"):
        print("Resume on file: yes")

    salary = data.get("salary")
    if salary:
        annualized = salary.get("annualized_amount_same_currency")
        currency = salary.get("currency", "unclear")
        print(
            f"Salary stated: {salary.get('as_stated')!r}"
            + (f" (~{annualized:,.0f} {currency}/yr)" if annualized else "")
        )

    print("\n--- Transcript ---\n")
    print(data.get("transcript", "(no transcript saved)"))

    print(f"\nFull transcript + raw data: {path}\n")


if __name__ == "__main__":
    main()
