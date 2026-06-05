"""
src/run_judge.py

Runs LLM-as-Judge evaluation on existing pipeline responses.
Reads from the latest responses JSON and adds judge scores.

Usage:
    python src/run_judge.py
    python src/run_judge.py --questions 3   # evaluate only 3 questions (quota-safe)
"""

import json
import argparse
import glob
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.llm_judge import run_llm_judge

ROOT = Path(__file__).parent.parent
RESPONSES_DIR = ROOT / "data" / "responses"
SCORES_DIR = ROOT / "data" / "scores"
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.json"

SCORES_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_responses() -> list:
    files = glob.glob(str(RESPONSES_DIR / "responses_*.json"))
    if not files:
        raise FileNotFoundError("No response files found. Run run_pipeline.py first.")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading responses from: {latest}")
    with open(latest, encoding="utf-8") as f:
        return json.load(f)


def load_questions_map() -> dict:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    return {q["id"]: q for q in questions}


def run_judge_pipeline(n_questions: int = None):
    print("\n" + "=" * 60)
    print("  LLM-as-Judge Evaluation Pipeline")
    print("=" * 60)

    responses_data = load_latest_responses()
    questions_map = load_questions_map()

    if n_questions:
        responses_data = responses_data[:n_questions]

    # Estimate API calls
    total_responses = sum(
        len([r for r in item["model_responses"] if r["response"] is not None])
        for item in responses_data
    )
    print(f"\n  Questions to evaluate : {len(responses_data)}")
    print(f"  Valid responses found : {total_responses}")
    print(f"  Estimated API calls   : {total_responses} (1 judge call per response)")
    print(f"  Judge model           : gpt-oss-120b\n")

    all_results = []

    for item in responses_data:
        q_id = item["question_id"]
        question = questions_map.get(q_id)
        if not question:
            continue

        print(f"\n[{q_id}] {item['question'][:60]}...")
        judged_responses = []

        for resp in item["model_responses"]:
            model_key = resp["model_key"]
            response_text = resp.get("response")

            if response_text is None:
                print(f"  [{model_key}] skipped (no response)")
                judged_responses.append({**resp, "llm_judge": None})
                continue

            print(f"  [{model_key}] judging...", end=" ", flush=True)
            judge_result = run_llm_judge(response_text, question)

            composite = judge_result.get("composite_judge_score")
            if composite is not None:
                print(f"score={composite:.3f}")
            else:
                print(f"parse_error={judge_result.get('parse_error', 'unknown')}")

            judged_responses.append({**resp, "llm_judge": judge_result})

        all_results.append({**item, "model_responses": judged_responses})

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCORES_DIR / f"judged_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("  JUDGE SCORE SUMMARY")
    print("=" * 60)
    print(f"\n  {'Question':<15} {'Model':<15} {'Auto':>6} {'Judge':>6} {'Halluc':>8}")
    print(f"  {'-'*15} {'-'*15} {'-'*6} {'-'*6} {'-'*8}")

    for item in all_results:
        for resp in item["model_responses"]:
            if resp.get("llm_judge") is None:
                continue
            judge = resp["llm_judge"]
            auto_score = resp.get("auto_eval", {}).get("composite_auto_score", "-")
            judge_score = judge.get("composite_judge_score", "-")
            halluc = judge.get("hallucination_risk", {})
            halluc_level = halluc.get("level", "-") if halluc else "-"

            auto_str = f"{auto_score:.3f}" if isinstance(auto_score, float) else str(auto_score)
            judge_str = f"{judge_score:.3f}" if isinstance(judge_score, float) else str(judge_score)

            print(f"  {item['question_id']:<15} {resp['model_key']:<15} {auto_str:>6} {judge_str:>6} {halluc_level:>8}")

    print(f"\n  Results saved → {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=None,
                        help="Number of questions to judge (default: all)")
    args = parser.parse_args()

    run_judge_pipeline(n_questions=args.questions)
