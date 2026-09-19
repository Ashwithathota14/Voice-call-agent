"""Run several scenarios back-to-back (sequentially, never in parallel — two full
recruiter+candidate pipelines already contend for local CPU/network; five in parallel would be
ten) and print one scorecard at the end instead of five separate outputs to eyeball by hand.

Each scenario runs as its own `run_scenario.py` subprocess (full isolation, matching how
run_scenario.py itself isolates bot.py/candidate_bot.py) and inherits all of that script's
behavior: end_reason check + judge verdict, transcript printed live, results/<room>.json and
results/<room>.judge.json saved per run.

Usage:
    python run_all_scenarios.py
    python run_all_scenarios.py scenarios/a.json scenarios/b.json
"""
import glob
import json
import subprocess
import sys

DEFAULT_SCENARIOS = [
    "scenarios/not_good_time.json",
    "scenarios/wrong_number.json",
    "scenarios/rambling_answer.json",
    "scenarios/mid_screening_question.json",
    "scenarios/volunteered_notice_period.json",
]


def latest_result_for(scenario_file: str, before: set[str]) -> str | None:
    """The results/*.json file this run just created (anything new since `before`)."""
    after = set(glob.glob("results/*.json")) - {f for f in glob.glob("results/*.judge.json")}
    new = after - before
    return max(new, key=lambda f: f, default=None) if new else None


def main() -> None:
    scenario_files = sys.argv[1:] or DEFAULT_SCENARIOS
    scorecard: list[dict] = []

    for i, scenario_file in enumerate(scenario_files, 1):
        with open(scenario_file, encoding="utf-8") as f:
            scenario = json.load(f)

        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(scenario_files)}] {scenario_file} — {scenario.get('test_case', '')}")
        print(f"{'=' * 70}\n", flush=True)

        before = set(glob.glob("results/*.json")) - {f for f in glob.glob("results/*.judge.json")}
        proc = subprocess.run([sys.executable, "run_scenario.py", scenario_file])
        result_path = latest_result_for(scenario_file, before)

        entry = {
            "scenario_file": scenario_file,
            "test_case": scenario.get("test_case", ""),
            "passed": proc.returncode == 0,
            "result_path": result_path,
        }
        if result_path:
            judge_path = result_path.rsplit(".json", 1)[0] + ".judge.json"
            try:
                with open(judge_path, encoding="utf-8") as f:
                    judge = json.load(f)
                entry["verdict"] = judge.get("verdict")
                entry["reasoning"] = judge.get("reasoning")
            except FileNotFoundError:
                entry["verdict"] = None
        scorecard.append(entry)

    print(f"\n\n{'#' * 70}")
    print("# SCORECARD")
    print(f"{'#' * 70}\n")
    passed_count = sum(1 for e in scorecard if e["passed"])
    for e in scorecard:
        status = "PASS" if e["passed"] else "FAIL"
        print(f"[{status}] {e['test_case']}  ({e['scenario_file']})")
        if e.get("verdict"):
            print(f"       judge: {e['verdict']} — {e.get('reasoning', '')}")
        if e.get("result_path"):
            print(f"       {e['result_path']}")
        print()
    print(f"{passed_count}/{len(scorecard)} scenarios passed\n")

    sys.exit(0 if passed_count == len(scorecard) else 1)


if __name__ == "__main__":
    main()
