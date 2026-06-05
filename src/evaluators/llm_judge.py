"""
src/evaluators/llm_judge.py

LLM-as-Judge: uses a separate LLM call to evaluate response quality
on dimensions that cannot be measured automatically.

Why LLM-as-Judge:
- Automated metrics (keyword match, length) miss nuance
- A response can be technically correct but incomprehensible to a 6th grader
- Hallucination detection requires semantic understanding, not just pattern matching

Dimensions evaluated:
1. Reasoning quality (1-5): Are the solution steps logical and complete?
2. Level appropriateness (1-5): Is the language suited for the student's grade?
3. Hallucination risk (low/medium/high): Does the model make unverifiable claims?

Known limitations (important to state in interview):
- LLM-as-Judge has self-preference bias (judge may favor responses similar to its own style)
- Judge model is also a free-tier model — its own quality is not guaranteed
- Scores are subjective and not reproducible across different judge models
- We use gpt-oss-120b as judge since it was most reliable in our tests
"""

import json
import time
import re
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.caller import call_model

# Use the most reliable model as judge
JUDGE_MODEL = "gpt-oss-120b"

JUDGE_PROMPT_TEMPLATE = """Kamu adalah evaluator pendidikan yang menilai kualitas jawaban AI untuk siswa Indonesia.

Berikut adalah soal dan jawaban yang perlu kamu evaluasi:

=== SOAL ===
Mata Pelajaran: {subject}
Tingkat: {level} ({grade})
Pertanyaan: {question}

=== JAWABAN REFERENSI (Ground Truth) ===
{ground_truth}

=== JAWABAN MODEL YANG DIEVALUASI ===
{response}

=== INSTRUKSI EVALUASI ===
Evaluasi jawaban model berdasarkan tiga dimensi berikut. Berikan skor dan alasan singkat.

Respond HANYA dengan JSON valid berikut (tidak ada teks lain di luar JSON):

{{
  "reasoning_quality": {{
    "score": <angka 1-5>,
    "reasoning": "<alasan singkat dalam 1-2 kalimat>"
  }},
  "level_appropriateness": {{
    "score": <angka 1-5>,
    "reasoning": "<alasan singkat dalam 1-2 kalimat>"
  }},
  "hallucination_risk": {{
    "level": "<low|medium|high>",
    "reasoning": "<alasan singkat dalam 1-2 kalimat>"
  }}
}}

=== RUBRIK ===

reasoning_quality (1-5):
1 = Tidak ada langkah penyelesaian, langsung jawaban salah
2 = Ada usaha tapi langkah tidak lengkap atau ada kesalahan logika
3 = Langkah cukup lengkap tapi ada gap minor
4 = Langkah jelas, logis, dan hampir sempurna
5 = Langkah sangat jelas, lengkap, mudah diikuti siswa

level_appropriateness (1-5):
1 = Bahasa terlalu teknis/sulit untuk tingkat {level}
2 = Beberapa bagian sulit dipahami siswa {level}
3 = Cukup sesuai, tapi ada beberapa istilah yang perlu penyederhanaan
4 = Bahasa sesuai dan mudah dipahami siswa {level}
5 = Bahasa sangat tepat, menggunakan analogi/contoh yang relevan untuk {level}

hallucination_risk:
low = Semua klaim dapat diverifikasi dan sesuai dengan ground truth
medium = Ada 1-2 klaim yang tidak ada di ground truth tapi tidak salah
high = Ada klaim yang salah atau tidak dapat diverifikasi
"""


def parse_judge_response(response_text: str) -> Optional[dict]:
    """
    Parse JSON from judge response.
    Handles cases where model adds extra text around the JSON.
    """
    if not response_text:
        return None

    # Try direct parse first
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from text
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def hallucination_to_score(level: str) -> float:
    """Convert hallucination risk level to numeric score (for aggregation)."""
    mapping = {"low": 1.0, "medium": 0.5, "high": 0.0}
    return mapping.get(level.lower(), 0.5)


def run_llm_judge(
    response: Optional[str],
    question: dict,
    delay: float = 2.0,
) -> dict:
    """
    Run LLM-as-Judge evaluation on a single model response.

    Args:
        response: The model response to evaluate
        question: Question dict from questions.json
        delay: Seconds to wait before calling judge (rate limit safety)

    Returns:
        dict with judge scores and raw judge response
    """
    if response is None:
        return {
            "judge_model": JUDGE_MODEL,
            "reasoning_quality": None,
            "level_appropriateness": None,
            "hallucination_risk": None,
            "hallucination_score": None,
            "composite_judge_score": None,
            "raw_judge_response": None,
            "parse_error": "No response to evaluate"
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        subject=question["subject"],
        level=question["level"],
        grade=question["grade"],
        question=question["question"],
        ground_truth=question["ground_truth"],
        response=response,
    )

    time.sleep(delay)
    result = call_model(prompt, JUDGE_MODEL, temperature=0.0)

    if result["error"]:
        return {
            "judge_model": JUDGE_MODEL,
            "reasoning_quality": None,
            "level_appropriateness": None,
            "hallucination_risk": None,
            "hallucination_score": None,
            "composite_judge_score": None,
            "raw_judge_response": None,
            "parse_error": result["error"]
        }

    parsed = parse_judge_response(result["response"])

    if not parsed:
        return {
            "judge_model": JUDGE_MODEL,
            "reasoning_quality": None,
            "level_appropriateness": None,
            "hallucination_risk": None,
            "hallucination_score": None,
            "composite_judge_score": None,
            "raw_judge_response": result["response"],
            "parse_error": "Failed to parse JSON from judge response"
        }

    # Extract scores
    rq = parsed.get("reasoning_quality", {})
    la = parsed.get("level_appropriateness", {})
    hr = parsed.get("hallucination_risk", {})

    rq_score = rq.get("score", 0) / 5.0 if rq.get("score") else None
    la_score = la.get("score", 0) / 5.0 if la.get("score") else None
    hr_level = hr.get("level", "medium")
    hr_score = hallucination_to_score(hr_level)

    # Composite judge score (weighted)
    # Hallucination weighted highest — a hallucinating model is dangerous for students
    weights = {"reasoning": 0.35, "level": 0.30, "hallucination": 0.35}
    if rq_score is not None and la_score is not None:
        composite = round(
            rq_score * weights["reasoning"]
            + la_score * weights["level"]
            + hr_score * weights["hallucination"],
            3
        )
    else:
        composite = None

    return {
        "judge_model": JUDGE_MODEL,
        "reasoning_quality": {
            "score_raw": rq.get("score"),
            "score_normalized": rq_score,
            "reasoning": rq.get("reasoning", "")
        },
        "level_appropriateness": {
            "score_raw": la.get("score"),
            "score_normalized": la_score,
            "reasoning": la.get("reasoning", "")
        },
        "hallucination_risk": {
            "level": hr_level,
            "score": hr_score,
            "reasoning": hr.get("reasoning", "")
        },
        "composite_judge_score": composite,
        "raw_judge_response": result["response"],
        "parse_error": None
    }
