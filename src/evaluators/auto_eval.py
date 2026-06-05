"""
src/evaluators/auto_eval.py

Automated evaluation metrics that don't require an LLM.
These run fast, are deterministic, and provide the quantitative backbone.
"""

import re
import json
from typing import Optional


def score_accuracy(response: Optional[str], question: dict) -> dict:
    """
    Checks if the model's response contains the key answer.

    Strategy: keyword matching against expected_answer_short and key_concepts.
    This is deliberately simple — the goal is a fast, reproducible signal,
    not perfect grading.

    Returns:
        dict with score (0.0-1.0) and reasoning
    """
    if response is None:
        return {"score": 0.0, "method": "keyword_match", "reasoning": "No response (API error)"}

    response_lower = response.lower()
    expected = question.get("expected_answer_short", "").lower()
    key_concepts = [k.lower() for k in question.get("key_concepts", [])]

    # Extract key numeric/formula parts from expected answer
    # e.g. "96 cm²" → ["96"]
    numeric_matches = re.findall(r'\d+\.?\d*', expected)
    concept_hits = sum(1 for concept in key_concepts if concept in response_lower)
    numeric_hits = sum(1 for n in numeric_matches if n in response_lower)

    # Scoring logic
    total_signals = len(key_concepts) + len(numeric_matches)
    if total_signals == 0:
        score = 0.5  # Can't evaluate without signals — neutral score
        reasoning = "No key concepts or numerics to match"
    else:
        total_hits = concept_hits + numeric_hits
        score = round(total_hits / total_signals, 2)
        reasoning = (
            f"Key concepts: {concept_hits}/{len(key_concepts)} matched. "
            f"Numeric values: {numeric_hits}/{len(numeric_matches)} matched."
        )

    return {
        "score": min(score, 1.0),
        "method": "keyword_match",
        "reasoning": reasoning,
    }


def score_response_structure(response: Optional[str]) -> dict:
    """
    Checks if the response includes structured reasoning steps.

    Signals of good structure:
    - Uses numbered steps or bullet points
    - Contains transition words (jadi, sehingga, maka, karena)
    - Has multiple sentences (not a one-liner dump)

    Returns:
        dict with score (0.0-1.0) and breakdown
    """
    if response is None:
        return {"score": 0.0, "method": "structure_check", "reasoning": "No response"}

    signals = {
        "has_numbered_steps": bool(re.search(r'\d+[\.\)]\s', response)),
        "has_bullet_points": bool(re.search(r'[-•*]\s', response)),
        "has_causal_connectors": bool(
            re.search(r'\b(jadi|sehingga|maka|karena|oleh karena itu|dengan demikian)\b',
                      response.lower())
        ),
        "multi_sentence": len(re.split(r'[.!?]', response)) > 3,
        "has_formula_or_equation": bool(re.search(r'[=×÷+\-²³√]', response)),
    }

    score = sum(signals.values()) / len(signals)
    return {
        "score": round(score, 2),
        "method": "structure_check",
        "reasoning": signals,
    }


def score_length_appropriateness(response: Optional[str], level: str) -> dict:
    """
    Checks if response length is appropriate for the student level.

    Heuristic (adjustable):
    - SD: 50-200 words (too long = overwhelming)
    - SMP: 100-300 words
    - SMA: 150-400 words

    Returns:
        dict with score and word count
    """
    if response is None:
        return {"score": 0.0, "method": "length_check", "word_count": 0}

    word_count = len(response.split())

    ranges = {
        "SD": (50, 200),
        "SMP": (100, 300),
        "SMA": (150, 400),
    }

    min_w, max_w = ranges.get(level, (50, 400))

    if min_w <= word_count <= max_w:
        score = 1.0
    elif word_count < min_w:
        score = round(word_count / min_w, 2)
    else:
        # Penalty for excessively long responses
        overshoot = word_count - max_w
        score = max(0.0, round(1.0 - (overshoot / max_w), 2))

    return {
        "score": score,
        "method": "length_check",
        "word_count": word_count,
        "target_range": f"{min_w}-{max_w} words",
        "reasoning": f"{word_count} words for {level} level (target: {min_w}-{max_w})",
    }


def run_auto_evaluation(response: Optional[str], question: dict) -> dict:
    """
    Runs all automated evaluations for a single response.

    Args:
        response: Model's text response
        question: Question dict from questions.json

    Returns:
        Combined evaluation results
    """
    level = question.get("level", "SMP")

    accuracy = score_accuracy(response, question)
    structure = score_response_structure(response)
    length = score_length_appropriateness(response, level)

    # Composite auto score (weighted)
    weights = {"accuracy": 0.5, "structure": 0.3, "length": 0.2}
    composite = round(
        accuracy["score"] * weights["accuracy"]
        + structure["score"] * weights["structure"]
        + length["score"] * weights["length"],
        3
    )

    return {
        "composite_auto_score": composite,
        "accuracy": accuracy,
        "structure": structure,
        "length": length,
    }
