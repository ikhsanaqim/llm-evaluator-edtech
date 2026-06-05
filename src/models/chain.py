"""
src/models/chain.py

Refactored model calling using LangChain LCEL (LangChain Expression Language).
Uses PromptTemplate + ChatOpenAI + StrOutputParser as a proper chain.

Why this matters:
- LangSmith automatically traces every chain invocation
- PromptTemplate makes prompt versioning and testing easier
- LCEL chains are composable — easy to add steps later (RAG, memory, tools)
- This is the pattern used in production LangChain applications

Difference from caller.py:
- caller.py: direct ChatOpenAI.invoke() — simple but not composable
- chain.py: PromptTemplate | ChatOpenAI | Parser — production pattern
"""

import os
import time
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Models available on OpenRouter free tier
MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "gemma-4-31b": "google/gemma-4-31b-it:free",
    "nemotron-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "nemotron-nano": "nvidia/nemotron-3-nano-30b-a3b:free",
}

# System prompt as a proper template
SYSTEM_TEMPLATE = """Kamu adalah asisten belajar untuk siswa Indonesia tingkat {level}.
Jawab setiap pertanyaan dengan jelas, terstruktur, dan sesuai tingkat pemahaman siswa {level}.
Selalu tunjukkan langkah-langkah penyelesaian jika itu soal hitungan.
Gunakan Bahasa Indonesia yang baik dan mudah dipahami."""

USER_TEMPLATE = """{question}"""

# Build the prompt template
EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("human", USER_TEMPLATE),
])


def get_chain(model_key: str, temperature: float = 0.0):
    """
    Build a LangChain LCEL chain for a specific model.

    Chain structure:
        input dict → PromptTemplate → ChatOpenAI → StrOutputParser → str

    Args:
        model_key: Key from MODELS dict
        temperature: Sampling temperature

    Returns:
        Runnable LCEL chain
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    model_name = MODELS.get(model_key)
    if not model_name:
        raise ValueError(f"Unknown model key: '{model_key}'. Choose from: {list(MODELS.keys())}")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=temperature,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://github.com/ikhsanaqim/llm-evaluator-edtech",
            "X-Title": "LLM Evaluator Edtech",
        },
    )

    # LCEL chain: prompt | llm | parser
    chain = EVAL_PROMPT | llm | StrOutputParser()
    return chain


def invoke_chain(
    question: str,
    level: str,
    model_key: str,
    temperature: float = 0.0,
    retry_on_fail: bool = True,
) -> dict:
    """
    Invoke a chain for a single question.
    LangSmith will automatically trace this if LANGCHAIN_TRACING_V2=true.

    Args:
        question: The question text
        level: Student level (SD/SMP/SMA) — used in system prompt
        model_key: Which model to use
        temperature: Sampling temperature
        retry_on_fail: Retry once on failure

    Returns:
        dict with model_key, response, latency_ms, error
    """
    chain = get_chain(model_key, temperature)

    start = time.time()
    try:
        response = chain.invoke(
            {"question": question, "level": level},
            config={"run_name": f"eval-{model_key}"}  # LangSmith run name
        )
        latency_ms = round((time.time() - start) * 1000)
        return {
            "model_key": model_key,
            "model_name": MODELS[model_key],
            "response": response,
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as e:
        if retry_on_fail:
            time.sleep(35)
            return invoke_chain(question, level, model_key, temperature, retry_on_fail=False)
        return {
            "model_key": model_key,
            "model_name": MODELS.get(model_key, "unknown"),
            "response": None,
            "latency_ms": None,
            "error": str(e),
        }


def invoke_all_chains(
    question: str,
    level: str,
    temperature: float = 0.0,
    delay_between_calls: float = 5.0,
) -> list[dict]:
    """
    Invoke all model chains with the same question.

    Args:
        question: The question text
        level: Student level for system prompt
        temperature: Sampling temperature
        delay_between_calls: Rate limit safety delay

    Returns:
        List of result dicts
    """
    results = []
    for i, model_key in enumerate(MODELS.keys()):
        if i > 0:
            time.sleep(delay_between_calls)
        result = invoke_chain(question, level, model_key, temperature)
        results.append(result)
        status = "✓" if result["error"] is None else "✗"
        print(f"  [{status}] {model_key} — {result.get('latency_ms', 'N/A')} ms")
    return results
