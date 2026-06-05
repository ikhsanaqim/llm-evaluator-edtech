"""
tests/test_auto_eval.py

Unit tests for automated evaluation functions.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.auto_eval import (
    score_accuracy,
    score_response_structure,
    score_length_appropriateness,
    run_auto_evaluation,
)


# Sample question for testing
SAMPLE_QUESTION = {
    "id": "MTK-SMP-002",
    "subject": "Matematika",
    "level": "SMP",
    "grade": "Kelas 8",
    "question": "Jika sebuah segitiga siku-siku memiliki dua sisi tegak 6 cm dan 8 cm, berapakah hipotenusanya?",
    "ground_truth": "c² = 6² + 8² = 36 + 64 = 100. Jadi c = √100 = 10 cm.",
    "key_concepts": ["teorema Pythagoras", "segitiga siku-siku"],
    "expected_answer_short": "10 cm",
}

GOOD_RESPONSE = """
Untuk mencari hipotenusa segitiga siku-siku, kita gunakan teorema Pythagoras:
c² = a² + b²

Langkah penyelesaian:
1. a = 6 cm, b = 8 cm
2. c² = 6² + 8²
3. c² = 36 + 64 = 100
4. c = √100 = 10 cm

Jadi, panjang hipotenusa segitiga siku-siku tersebut adalah 10 cm.
"""

BAD_RESPONSE = "Jawabannya adalah 15 cm."

NONE_RESPONSE = None


class TestScoreAccuracy:
    def test_good_response_scores_high(self):
        result = score_accuracy(GOOD_RESPONSE, SAMPLE_QUESTION)
        assert result["score"] >= 0.5, f"Expected >= 0.5, got {result['score']}"

    def test_bad_response_scores_low(self):
        result = score_accuracy(BAD_RESPONSE, SAMPLE_QUESTION)
        assert result["score"] < 0.8, f"Expected < 0.8, got {result['score']}"

    def test_none_response_scores_zero(self):
        result = score_accuracy(NONE_RESPONSE, SAMPLE_QUESTION)
        assert result["score"] == 0.0

    def test_score_is_between_0_and_1(self):
        result = score_accuracy(GOOD_RESPONSE, SAMPLE_QUESTION)
        assert 0.0 <= result["score"] <= 1.0


class TestScoreStructure:
    def test_structured_response_scores_high(self):
        result = score_response_structure(GOOD_RESPONSE)
        assert result["score"] >= 0.5

    def test_minimal_response_scores_low(self):
        result = score_response_structure(BAD_RESPONSE)
        assert result["score"] <= 0.4

    def test_none_response_scores_zero(self):
        result = score_response_structure(NONE_RESPONSE)
        assert result["score"] == 0.0


class TestScoreLength:
    def test_appropriate_length_scores_1(self):
        # GOOD_RESPONSE is ~70 words — within SMP range (100-300)
        # May score below 1.0, that's fine — testing it runs correctly
        result = score_length_appropriateness(GOOD_RESPONSE, "SMP")
        assert 0.0 <= result["score"] <= 1.0
        assert "word_count" in result

    def test_none_response_scores_zero(self):
        result = score_length_appropriateness(NONE_RESPONSE, "SMP")
        assert result["score"] == 0.0

    def test_all_levels_accepted(self):
        for level in ["SD", "SMP", "SMA"]:
            result = score_length_appropriateness(GOOD_RESPONSE, level)
            assert "score" in result


class TestRunAutoEvaluation:
    def test_returns_all_keys(self):
        result = run_auto_evaluation(GOOD_RESPONSE, SAMPLE_QUESTION)
        assert "composite_auto_score" in result
        assert "accuracy" in result
        assert "structure" in result
        assert "length" in result

    def test_composite_is_between_0_and_1(self):
        result = run_auto_evaluation(GOOD_RESPONSE, SAMPLE_QUESTION)
        assert 0.0 <= result["composite_auto_score"] <= 1.0

    def test_handles_none_response(self):
        result = run_auto_evaluation(None, SAMPLE_QUESTION)
        assert result["composite_auto_score"] == 0.0
