"""
src/evaluators/consistency_eval.py

Consistency evaluation: sends the same question to a model 3 times
and measures how stable the output is.

Why this matters for edtech:
- A student asking the same question twice shouldn't get contradictory answers
- Inconsistent models are unreliable as learning tools
- This reveals whether a model is truly "confident" or just guessing

Metrics:
- Answer consistency: does the final answer stay the same?
- Numeric consistency: do key numbers appear in all 3 responses?
- Length variance: how much does response length fluctuate?
- Consistency score: composite 0.0-1.0
"""

import time
import re
import json
from typing import Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.caller import call_model


def extract_key_numbers(text: str) -> set:
    """Extract numeric values from text for comparison."""
    if not text:
        return set()
    return set(re.findall(r'\b\d+\.?\d*\b', text))


def extract_key_terms(text: str, question: dict) -> set:
    """Extract key concept terms that should appear in all responses."""
    if not text:
        return set()
    text_lower = text.lower()
    concepts = [c.lower() for c in question.get("key_concepts", [])]
    return set(c for c in concepts if c in text_lower)


def score_consistency(responses: list[Optional[str]], question: dict) -> dict:
    """
    Given 3 responses to the same question, compute consistency metrics.

    Args:
        responses: List of 3 response strings (can contain None for failed calls)
        question: Question dict from questions.json

    Returns:
        dict with consistency scores and breakdown
    """
    valid = [r for r in responses if r is not None]

    if len(valid) == 0:
        return {
            "consistency_score": 0.0,
            "valid_responses": 0,
            "answer_consistency": 0.0,
            "numeric_consistency": 0.0,
            "length_variance_pct": None,
            "reasoning": "All responses failed"
        }

    if len(valid) == 1:
        return {
            "consistency_score": 0.5,
            "valid_responses": 1,
            "answer_consistency": None,
            "numeric_consistency": None,
            "length_variance_pct": None,
            "reasoning": "Only 1 valid response — cannot measure consistency"
        }

    # --- Numeric consistency ---
    # Key numbers should appear in ALL valid responses
    number_sets = [extract_key_numbers(r) for r in valid]
    common_numbers = number_sets[0].intersection(*number_sets[1:])
    all_numbers = number_sets[0].union(*number_sets[1:])

    if len(all_numbers) == 0:
        numeric_consistency = 1.0  # No numbers to compare (e.g. essay questions)
    else:
        numeric_consistency = round(len(common_numbers) / len(all_numbers), 3)

    # --- Key concept consistency ---
    concept_sets = [extract_key_terms(r, question) for r in valid]
    common_concepts = concept_sets[0].intersection(*concept_sets[1:])
    all_concepts = concept_sets[0].union(*concept_sets[1:])

    if len(all_concepts) == 0:
        concept_consistency = 1.0
    else:
        concept_consistency = round(len(common_concepts) / len(all_concepts), 3)

    # --- Length variance ---
    lengths = [len(r.split()) for r in valid]
    avg_length = sum(lengths) / len(lengths)
    variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    length_variance_pct = round((std_dev / avg_length) * 100, 1) if avg_length > 0 else 0

    # Length variance penalty: >50% variance is bad
    if length_variance_pct <= 20:
        length_score = 1.0
    elif length_variance_pct <= 50:
        length_score = round(1.0 - ((length_variance_pct - 20) / 30) * 0.5, 3)
    else:
        length_score = 0.5

    # --- Composite consistency score ---
    # Weighted: numeric most important for edtech, then concepts, then length
    weights = {"numeric": 0.45, "concept": 0.35, "length": 0.20}
    composite = round(
        numeric_consistency * weights["numeric"]
        + concept_consistency * weights["concept"]
        + length_score * weights["length"],
        3
    )

    return {
        "consistency_score": composite,
        "valid_responses": len(valid),
        "numeric_consistency": numeric_consistency,
        "concept_consistency": concept_consistency,
        "length_variance_pct": length_variance_pct,
        "word_counts": lengths,
        "common_numbers": sorted(list(common_numbers)),
        "common_concepts": sorted(list(common_concepts)),
        "reasoning": (
            f"Numbers: {len(common_numbers)}/{len(all_numbers)} consistent. "
            f"Concepts: {len(common_concepts)}/{len(all_concepts)} consistent. "
            f"Length variance: {length_variance_pct}%."
        )
    }


def run_consistency_test(
    question: dict,
    model_key: str,
    n_repeats: int = 3,
    temperature: float = 0.3,
    delay: float = 3.0,
) -> dict:
    """
    Run consistency test for one question on one model.

    Note: temperature=0.3 instead of 0.0 — we WANT some variance to stress-test.
    At temperature=0.0 models are deterministic, so consistency would always be 1.0.
    That would defeat the purpose of the test.

    Args:
        question: Question dict
        model_key: Which model to test
        n_repeats: How many times to ask (default 3)
        temperature: Sampling temperature (use >0 for meaningful variance)
        delay: Seconds between calls

    Returns:
        dict with all responses and consistency scores
    """
    print(f"  Running {n_repeats}x consistency test for [{model_key}]...")
    responses = []
    latencies = []

    for i in range(n_repeats):
        if i > 0:
            time.sleep(delay)
        result = call_model(question["question"], model_key, temperature=temperature)
        responses.append(result["response"])
        latencies.append(result["latency_ms"])
        status = "✓" if result["error"] is None else "✗"
        print(f"    Run {i+1}: [{status}] {result.get('latency_ms', 'N/A')} ms")

    scores = score_consistency(responses, question)

    return {
        "question_id": question["id"],
        "subject": question["subject"],
        "level": question["level"],
        "model_key": model_key,
        "n_repeats": n_repeats,
        "temperature": temperature,
        "responses": responses,
        "latencies_ms": latencies,
        "consistency": scores,
    }
