"""
src/run_pipeline.py

Main pipeline for Week 1: Load questions → Call all models → Save raw responses.
Run this script to execute the full evaluation batch.

Usage:
    python src/run_pipeline.py
    python src/run_pipeline.py --dry-run        # Test with 2 questions only
    python src/run_pipeline.py --subject MTK    # Filter by subject
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.caller import call_all_models, MODELS
from src.evaluators.auto_eval import run_auto_evaluation

# Paths
ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.json"
RESPONSES_DIR = ROOT / "data" / "responses"
SCORES_DIR = ROOT / "data" / "scores"

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)


def load_questions(subject_filter: str = None, dry_run: bool = False) -> list:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if subject_filter:
        subject_map = {"MTK": "Matematika", "IPA": "IPA", "BIN": "Bahasa Indonesia"}
        full_subject = subject_map.get(subject_filter.upper(), subject_filter)
        questions = [q for q in questions if q["subject"] == full_subject]

    if dry_run:
        questions = questions[:2]

    return questions


def run_pipeline(dry_run: bool = False, subject_filter: str = None):
    print("\n" + "=" * 60)
    print("  LLM Evaluator for Edtech — Pipeline Runner")
    print("=" * 60)

    questions = load_questions(subject_filter, dry_run)
    print(f"\n📋 Questions loaded: {len(questions)}")
    print(f"🤖 Models: {', '.join(MODELS.keys())}")
    if dry_run:
        print("⚠️  DRY RUN MODE — only 2 questions")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for q in tqdm(questions, desc="Processing questions"):
        print(f"\n[{q['id']}] {q['question'][:60]}...")

        # Call all models
        model_responses = call_all_models(q["question"])

        # Run automated evaluation on each response
        evaluated_responses = []
        for resp in model_responses:
            auto_scores = run_auto_evaluation(resp["response"], q)
            evaluated_responses.append({**resp, "auto_eval": auto_scores})

        result = {
            "question_id": q["id"],
            "subject": q["subject"],
            "level": q["level"],
            "grade": q["grade"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "expected_answer_short": q["expected_answer_short"],
            "model_responses": evaluated_responses,
            "evaluated_at": timestamp,
        }
        all_results.append(result)

    # Save raw responses
    response_file = RESPONSES_DIR / f"responses_{timestamp}.json"
    with open(response_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Save summary scores CSV
    import csv
    scores_file = SCORES_DIR / f"scores_{timestamp}.csv"
    with open(scores_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "subject", "level", "grade", "model_key",
            "composite_auto_score", "accuracy_score", "structure_score",
            "length_score", "word_count", "latency_ms", "error"
        ])
        writer.writeheader()
        for result in all_results:
            for resp in result["model_responses"]:
                ae = resp.get("auto_eval", {})
                writer.writerow({
                    "question_id": result["question_id"],
                    "subject": result["subject"],
                    "level": result["level"],
                    "grade": result["grade"],
                    "model_key": resp["model_key"],
                    "composite_auto_score": ae.get("composite_auto_score", ""),
                    "accuracy_score": ae.get("accuracy", {}).get("score", ""),
                    "structure_score": ae.get("structure", {}).get("score", ""),
                    "length_score": ae.get("length", {}).get("score", ""),
                    "word_count": ae.get("length", {}).get("word_count", ""),
                    "latency_ms": resp.get("latency_ms", ""),
                    "error": resp.get("error", ""),
                })

    print(f"\n✅ Done!")
    print(f"   Responses → {response_file}")
    print(f"   Scores    → {scores_file}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run with only 2 questions")
    parser.add_argument("--subject", type=str, help="Filter by subject: MTK, IPA, or BIN")
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run, subject_filter=args.subject)
