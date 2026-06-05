# LLM Evaluator for Edtech 🎓

A multi-dimensional evaluation framework that benchmarks LLM performance on Indonesian K-12 educational question answering.

## Problem

Not all LLMs perform equally well for educational contexts. A model that scores high on general benchmarks may still give confusing explanations to a 6th grader, or produce confident but incorrect answers. This project systematically evaluates multiple LLMs across dimensions that matter for student learning.

## What It Evaluates

| Dimension | Method | Description |
|-----------|--------|-------------|
| Accuracy | Automated | Is the answer correct vs ground truth? |
| Consistency | Automated | Does the model give the same answer when asked 3x? |
| Reasoning Quality | LLM-as-Judge | Are the solution steps logical and complete? |
| Level Appropriateness | LLM-as-Judge | Is the language suited for the student's grade level? |
| Hallucination Risk | LLM-as-Judge | Does the model make unverifiable claims? |

## Models Compared

- `mistralai/mistral-7b-instruct` (free)
- `google/gemma-2-9b-it` (free)
- `meta-llama/llama-3-8b-instruct` (free)
- `qwen/qwen-2.5-7b-instruct` (free)

## Dataset

20 questions across:
- **Subjects**: Mathematics, Natural Science (IPA), Bahasa Indonesia
- **Levels**: SD (Elementary), SMP (Middle School), SMA (High School)

## Stack

| Layer | Tool |
|-------|------|
| LLM API | OpenRouter (free tier) |
| Orchestration | LangChain |
| Observability | LangSmith |
| Dashboard | Streamlit |
| Hosting | Streamlit Cloud |

## Project Structure

```
llm-evaluator-edtech/
├── data/
│   ├── questions/       # Question dataset (JSON)
│   ├── responses/       # Raw model responses
│   └── scores/          # Evaluation results
├── src/
│   ├── models/          # Model calling logic
│   ├── evaluators/      # Evaluation logic (auto + LLM-as-judge)
│   └── utils/           # Helpers
├── dashboard/           # Streamlit app
├── notebooks/           # Exploratory analysis
├── tests/               # Unit tests
└── docs/                # Architecture diagrams
```

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourusername/llm-evaluator-edtech
cd llm-evaluator-edtech

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 4. Run evaluation pipeline
python src/run_pipeline.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Key Findings

*(To be updated after experiments)*

## Limitations

- LLM-as-Judge has inherent bias toward its own reasoning style
- Free tier rate limits mean batch evaluation takes longer
- Ground truth for open-ended questions is manually defined, introducing human bias
- Consistency test uses 3 repetitions — larger N would be more statistically robust
