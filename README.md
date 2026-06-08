# LLM Evaluator for Edtech 🎓

A multi-dimensional evaluation framework that benchmarks LLM performance on Indonesian K-12 educational question answering across 20 questions, 4 models, 3 subjects, and 3 grade levels.

**Live Dashboard** → [huggingface.co/spaces/X-san/llm-evaluator-edtech](https://huggingface.co/spaces/X-san/llm-evaluator-edtech)

**Traces** → [LangSmith Project](https://smith.langchain.com) · `llm-evaluator-edtech`

---

## Motivation

Not all LLMs perform equally well for educational contexts. A model that scores high on general benchmarks may still give confusing explanations to a 6th grader, produce confident but incorrect answers, or respond inconsistently when the same question is asked twice. This project systematically evaluates multiple LLMs across dimensions that matter for student learning — not just accuracy.

---

## Evaluation Dimensions

| Dimension | Method | Description |
|-----------|--------|-------------|
| **Accuracy** | Automated | Keyword + numeric match vs ground truth |
| **Structure** | Automated | Presence of reasoning steps, connectors, formulas |
| **Length Fit** | Automated | Response length appropriate for grade level (SD/SMP/SMA) |
| **Consistency** | Automated | Same question asked 3× at temp=0.3 — answer stability |
| **Reasoning Quality** | LLM-as-Judge | Logical completeness of solution steps (1–5) |
| **Level Appropriateness** | LLM-as-Judge | Language suitability for student's grade (1–5) |
| **Hallucination Risk** | LLM-as-Judge | Presence of unverifiable claims (low/medium/high) |

---

## Models Compared

All models accessed via [OpenRouter](https://openrouter.ai) free tier:

| Model | Provider | Parameters |
|-------|----------|------------|
| `openai/gpt-oss-120b:free` | OpenAI | 120B |
| `google/gemma-4-31b-it:free` | Google | 31B |
| `nvidia/nemotron-3-super-120b-a12b:free` | NVIDIA | 120B |
| `nvidia/nemotron-3-nano-30b-a3b:free` | NVIDIA | 30B |

---

## Dataset

20 questions across:
- **Subjects**: Matematika, IPA, Bahasa Indonesia
- **Levels**: SD (Elementary), SMP (Middle School), SMA (High School)
- **Format**: Multiple open-ended questions with ground truth answers and key concepts

---

## Key Findings

### 1. Gemma-4-31b scores highest despite being the smallest model tested at scale

| Model | Composite Score | Accuracy | Structure | Avg Latency |
|-------|----------------|----------|-----------|-------------|
| gemma-4-31b | **0.877** | 0.837 | 0.960 | 11,073 ms |
| gpt-oss-120b | 0.813 | 0.738 | 0.930 | 16,502 ms |
| nemotron-nano | 0.790 | 0.694 | 0.943 | **4,110 ms** |
| nemotron-super | 0.774 | 0.745 | 0.814 | 20,061 ms |

Gemma-4-31b outperforms both 120B models on composite score, suggesting parameter count is not the primary driver of educational Q&A quality for Bahasa Indonesia.

### 2. nemotron-nano offers the best speed-quality tradeoff

At 4,110ms average latency — 5× faster than nemotron-super — with only a marginal quality gap (0.790 vs 0.774). For real-time student interactions where latency matters, nemotron-nano is the practical choice.

### 3. IPA questions score highest; Bahasa Indonesia scores lowest

| Subject | Avg Score |
|---------|-----------|
| IPA | **0.890** |
| SMP level | 0.816 |
| Matematika | 0.775 |
| Bahasa Indonesia | 0.745 |

Open-ended language questions (e.g. majas, teks eksposisi) are harder to evaluate automatically — models may give correct answers that don't match ground truth keywords, deflating scores. This is a known limitation of keyword-based evaluation.

### 4. Auto evaluation underestimates model performance on simple questions

LLM-as-Judge scores are consistently higher than automated scores for the same responses. Example: nemotron-super scored 0.603 (auto) vs 1.000 (judge) on MTK-SD-002. Keyword matching penalizes paraphrased but correct answers — a known limitation of lexical evaluation methods.

### 5. Consistency varies by question type

gpt-oss-120b consistency scores:
- MTK-SD-001 (persegi panjang): **1.000** — deterministic numeric answer
- MTK-SMP-002 (Pythagoras): **0.462** — intermediate calculation steps vary
- IPA-SD-002 (fotosintesis): **0.625** — essay answers vary in structure

Numeric consistency metric performs poorly on essay questions because list numbering is counted as "numbers", inflating the denominator. A future improvement would separate numeric vs. conceptual questions before applying this metric.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Flow                         │
│                                                         │
│  questions.json                                         │
│       │                                                 │
│       ▼                                                 │
│  LangChain PromptTemplate (level-aware system prompt)   │
│       │                                                 │
│       ▼                                                 │
│  OpenRouter API ──► 4 Models (parallel calls)           │
│       │                  │                              │
│       │              LangSmith                          │
│       │              (traces all runs)                  │
│       ▼                                                 │
│  Auto Evaluator                                         │
│  ├── Accuracy (keyword + numeric match)                 │
│  ├── Structure (reasoning step detection)               │
│  └── Length fit (grade-level heuristic)                 │
│       │                                                 │
│  LLM-as-Judge (gpt-oss-120b)                           │
│  ├── Reasoning quality (1–5)                            │
│  ├── Level appropriateness (1–5)                        │
│  └── Hallucination risk (low/medium/high)               │
│       │                                                 │
│  Consistency Evaluator (3× repetition test)             │
│       │                                                 │
│       ▼                                                 │
│  Streamlit Dashboard (HF Spaces)                        │
└─────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| LLM API | OpenRouter (free tier) | Multi-model access |
| Orchestration | LangChain LCEL | Chain composition |
| Observability | LangSmith | Trace every run |
| Dashboard | Streamlit | Visualization |
| Hosting | Hugging Face Spaces | Free deployment |

---

## Project Structure

```
llm-evaluator-edtech/
├── data/
│   ├── questions/          # 20 questions (JSON)
│   ├── responses/          # Raw model responses
│   └── scores/             # Evaluation results (CSV + JSON)
├── src/
│   ├── models/
│   │   ├── caller.py       # Direct API calls
│   │   └── chain.py        # LangChain LCEL chains
│   ├── evaluators/
│   │   ├── auto_eval.py    # Automated metrics
│   │   ├── consistency_eval.py  # Consistency testing
│   │   └── llm_judge.py    # LLM-as-Judge
│   ├── run_pipeline_v2.py  # Main pipeline (LangChain)
│   ├── run_consistency.py  # Consistency runner
│   └── run_judge.py        # Judge runner
├── tests/
│   └── test_auto_eval.py   # 13 unit tests
└── app.py                  # Streamlit dashboard (HF Spaces)
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/ikhsanaqim/llm-evaluator-edtech
cd llm-evaluator-edtech

# 2. Setup environment
conda create -n llm-evaluator python=3.11 -y
conda activate llm-evaluator
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add OPENROUTER_API_KEY and LANGCHAIN_API_KEY

# 4. Run pipeline
python src/run_pipeline_v2.py

# 5. Run evaluations
python src/run_consistency.py --model gpt-oss-120b
python src/run_judge.py --questions 5

# 6. Launch dashboard
streamlit run app.py
```

---

## Known Limitations

- **Free tier rate limits**: OpenRouter free tier caps at 200 requests/day per account. Full evaluation (20 questions × 4 models = 80 requests) consumes 40% of daily quota in one run.
- **LLM-as-Judge bias**: The judge model may favor responses stylistically similar to its own output. Scores are not reproducible across different judge models.
- **Keyword evaluation gap**: Auto accuracy metric underperforms on paraphrased correct answers and essay-type questions. Semantic similarity (e.g. cosine similarity with embeddings) would be more robust.
- **Consistency metric**: Numeric consistency is not well-suited for essay questions — future improvement would classify question type before applying the metric.
- **Dataset size**: 20 questions is sufficient for a proof-of-concept but insufficient for statistically significant conclusions.

---

## Author

Built by [Ikhsan Mustaqim](https://github.com/ikhsanaqim) as a portfolio project targeting AI/LLM engineering internship roles.

Electrical Engineering student, Institut Teknologi Bandung (ITB). Expected graduation: April 2027.
