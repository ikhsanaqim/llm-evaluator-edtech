"""
src/run_consistency.py

Runs consistency evaluation on a subset of questions.

Strategy: 5 questions × 1 model × 3 repeats = 15 API calls
Keeps quota usage minimal while still producing meaningful data.

Usage:
    python src/run_consistency.py
    python src/run_consistency.py --model gpt-oss-120b
    python src/run_consistency.py --questions 3
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.consistency_eval import run_consistency_test
from src.models.caller import MODELS

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.json"
SCORES_DIR = ROOT / "data" / "scores"
SCORES_DIR.mkdir(parents=True, exist_ok=True)


def load_sample_questions(n: int = 5) -> list:
    """
    Load a representative sample of questions.
    Picks questions spread across subjects and levels.
    """
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Pick spread: 2 MTK, 2 IPA, 1 BIN across different levels
    targets = ["MTK-SD-001", "MTK-SMP-002", "IPA-SD-002", "IPA-SMP-001", "BIN-SMA-002"]
    selected = [q for q in all_questions if q["id"] in targets]

    return selected[:n]


def run_consistency_pipeline(model_key: str = "gpt-oss-120b", n_questions: int = 5):
    print("\n" + "=" * 60)
    print("  LLM Consistency Evaluator")
    print("=" * 60)
    print(f"\n  Model  : {model_key}")
    print(f"  Questions : {n_questions}")
    print(f"  Repeats   : 3x per question")
    print(f"  API calls : {n_questions * 3} total")
    print(f"  Temp      : 0.3 (intentional variance)\n")

    questions = load_sample_questions(n_questions)
    all_results = []

    for q in questions:
        print(f"\n[{q['id']}] {q['question'][:65]}...")
        result = run_consistency_test(q, model_key)
        score = result["consistency"]["consistency_score"]
        print(f"  → Consistency score: {score:.3f} | {result['consistency']['reasoning']}")
        all_results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = SCORES_DIR / f"consistency_{model_key}_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  CONSISTENCY SUMMARY")
    print("=" * 60)
    scores = [r["consistency"]["consistency_score"] for r in all_results]
    avg = sum(scores) / len(scores)
    print(f"\n  Model: {model_key}")
    print(f"  Avg consistency score: {avg:.3f}")
    print(f"\n  Per question:")
    for r in all_results:
        score = r["consistency"]["consistency_score"]
        bar = "█" * int(score * 20)
        print(f"  [{r['question_id']}] {score:.3f} {bar}")

    print(f"\n  Results saved → {output_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-oss-120b",
                        choices=list(MODELS.keys()),
                        help="Which model to test")
    parser.add_argument("--questions", type=int, default=5,
                        help="Number of questions to test (default: 5)")
    args = parser.parse_args()

    run_consistency_pipeline(args.model, args.questions)
