"""
src/models/caller.py

Handles calling multiple LLMs via OpenRouter using LangChain.
All models are on OpenRouter's free tier.
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Models available on OpenRouter free tier
# Verify current availability at: https://openrouter.ai/models?q=free
MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "gemma-4-31b": "google/gemma-4-31b-it:free",
    "nemotron-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "kimi-k2": "moonshotai/kimi-k2.6:free",
}

SYSTEM_PROMPT = """Kamu adalah asisten belajar untuk siswa Indonesia.
Jawab setiap pertanyaan dengan jelas, terstruktur, dan sesuai tingkat pemahaman siswa.
Selalu tunjukkan langkah-langkah penyelesaian jika itu soal hitungan.
Gunakan Bahasa Indonesia yang baik dan mudah dipahami."""


def get_llm(model_key: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI instance configured for OpenRouter.

    Args:
        model_key: Key from MODELS dict (e.g., 'mistral-7b')
        temperature: 0.0 for deterministic output (better for evaluation)

    Returns:
        Configured ChatOpenAI instance
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    model_name = MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model key: '{model_key}'. Choose from: {list(MODELS.keys())}")

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://github.com/yourusername/llm-evaluator-edtech",
            "X-Title": "LLM Evaluator Edtech",
        },
    )


def call_model(
    question: str,
    model_key: str,
    temperature: float = 0.0,
    retry_on_fail: bool = True,
) -> dict:
    """
    Calls a single model with a question and returns the result.

    Args:
        question: The question text
        model_key: Which model to call
        temperature: Sampling temperature
        retry_on_fail: Whether to retry once on failure

    Returns:
        dict with keys: model_key, response, latency_ms, error (if any)
    """
    llm = get_llm(model_key, temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    start = time.time()
    try:
        response = llm.invoke(messages)
        latency_ms = round((time.time() - start) * 1000)
        return {
            "model_key": model_key,
            "model_name": MODELS[model_key],
            "response": response.content,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        if retry_on_fail:
            time.sleep(35)
            return call_model(question, model_key, temperature, retry_on_fail=False)
        return {
            "model_key": model_key,
            "model_name": MODELS.get(model_key, "unknown"),
            "response": None,
            "latency_ms": None,
            "error": str(e),
        }


def call_all_models(
    question: str,
    temperature: float = 0.0,
    delay_between_calls: float = 5.0,
) -> list[dict]:
    """
    Calls all models with the same question.

    Args:
        question: The question text
        temperature: Sampling temperature
        delay_between_calls: Seconds to wait between API calls (rate limit safety)

    Returns:
        List of result dicts from call_model()
    """
    results = []
    for i, model_key in enumerate(MODELS.keys()):
        if i > 0:
            time.sleep(delay_between_calls)
        result = call_model(question, model_key, temperature)
        results.append(result)
        status = "✓" if result["error"] is None else "✗"
        print(f"  [{status}] {model_key} — {result.get('latency_ms', 'N/A')} ms")
    return results
