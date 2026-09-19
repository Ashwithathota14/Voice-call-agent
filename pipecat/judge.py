"""LLM judge: grades a finished call's transcript against its scenario's test
case, instead of the crude expected_end_reason string-match run_scenario.py
already does.

end_reason match only tells you the call *terminated* the way you expected —
it says nothing about whether the recruiter bot behaved correctly along the
way (asked the right questions, didn't leak info it shouldn't have, handled
the candidate's behavior appropriately, etc). This reads the transcript and
the scenario's test_case/expected description and asks an LLM to verdict it,
so you get PASS/FAIL/PARTIAL plus concrete reasoning you can act on when
tuning the recruiter prompt in bot.py.

Usage:
    python judge.py results/scenario-1789639191.json scenarios/not_good_time.json
    python judge.py --latest scenarios/not_good_time.json

Prints the verdict and also writes it next to the result, as
results/<room_name>.judge.json, so repeated runs of the same scenario can be
diffed over time.
"""
import argparse
import glob
import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

JUDGE_SYSTEM_PROMPT = """You are a strict QA evaluator for an AI recruiter voice agent. You are given:
1. A test case description and its expected behavior for the recruiter agent.
2. The actual transcript of a call between the recruiter agent and a candidate (real or
   simulated) that was run against that test case.
3. The expected end_reason the call should have terminated with, and the actual end_reason.

Judge ONLY the recruiter agent's behavior (lines from "assistant"/the recruiter side) against
the expected behavior — the candidate's lines are just the stimulus. Be strict: if the agent
did something the test case says it should NOT do (e.g. asked screening questions when it was
told not to), that is a FAIL even if the call ended with the "correct" end_reason.

Respond with ONLY a JSON object, no markdown fences, no prose outside the JSON:
{
  "verdict": "PASS" | "FAIL" | "PARTIAL",
  "reasoning": "2-4 sentences on why, citing specific transcript moments",
  "matched": ["short bullet per expected behavior the agent got right"],
  "violated": ["short bullet per expected behavior the agent got wrong or missed"],
  "prompt_fix_suggestion": "one concrete suggestion for what to change in the recruiter's
      system prompt to fix the worst violation, or empty string if verdict is PASS"
}

verdict rules:
- PASS: agent's behavior fully matches the expected description, end_reason included.
- PARTIAL: agent got the core behavior right but missed a secondary expectation (e.g. right
  outcome but forgot to capture a callback time that was offered).
- FAIL: agent violated an explicit "must not" in the expected behavior, or the end_reason is
  wrong, or the agent's behavior would embarrass the company / mishandle the candidate.
"""


def build_judge_client() -> tuple[AzureOpenAI, str]:
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-12-01-preview",
    )
    return client, deployment


def judge_call(scenario: dict, result: dict) -> dict:
    client, deployment = build_judge_client()

    user_content = f"""TEST CASE: {scenario.get('test_case', '(none given)')}

EXPECTED BEHAVIOR: {scenario.get('expected', '(none given)')}

EXPECTED end_reason: {scenario.get('expected_end_reason', '(not specified)')}
ACTUAL end_reason:   {result.get('end_reason', 'unknown')}

TRANSCRIPT:
{result.get('transcript', '(no transcript saved)')}
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def find_latest_result() -> str:
    files = glob.glob(os.path.join("results", "*.json"))
    files = [f for f in files if not f.endswith(".judge.json")]
    if not files:
        raise SystemExit("No results yet — results/ is empty. Finish a call first.")
    return max(files, key=os.path.getmtime)


def print_verdict(verdict: dict) -> None:
    print(f"\n=== JUDGE VERDICT: {verdict.get('verdict', 'UNKNOWN')} ===")
    print(f"\n{verdict.get('reasoning', '')}")
    if verdict.get("matched"):
        print("\nMatched expected behavior:")
        for line in verdict["matched"]:
            print(f"  + {line}")
    if verdict.get("violated"):
        print("\nViolated / missed:")
        for line in verdict["violated"]:
            print(f"  - {line}")
    if verdict.get("prompt_fix_suggestion"):
        print(f"\nSuggested prompt fix:\n  {verdict['prompt_fix_suggestion']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", nargs="?", help="Path to results/<room>.json")
    parser.add_argument("scenario_file", help="Path to the scenarios/*.json this call was testing")
    parser.add_argument("--latest", action="store_true", help="Judge the most recently finished call")
    args = parser.parse_args()

    result_path = find_latest_result() if (args.latest or not args.result_file) else args.result_file

    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)
    with open(args.scenario_file, encoding="utf-8") as f:
        scenario = json.load(f)

    verdict = judge_call(scenario, result)
    print_verdict(verdict)

    out_path = result_path.rsplit(".json", 1)[0] + ".judge.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
