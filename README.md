# Ask My Docs — A Court of Mist and Fury

A production-style RAG (Retrieval-Augmented Generation) chatbot grounded entirely in the text of *A Court of Mist and Fury* by Sarah J. Maas. Ask it anything about the book — characters, plot, lore, magic — and it answers using only what the pages say, with inline citations pointing back to the source passages.

Built to demonstrate a full RAG engineering stack: hybrid retrieval, cross-encoder reranking, confidence-aware prompting, streaming responses, and an automated RAGAS quality gate running in CI.

---

## Demo

![Chatbot UI](assets/demo.png)

---

## How it works

Every answer follows the same pipeline:

```
User query
    │
    ├─ BM25 keyword search  ──┐
    │                          ├─ Reciprocal Rank Fusion → top 20 candidates
    └─ Vector similarity search ─┘
                │
        Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
                │
        Confidence scoring  (high / moderate / low / none)
        based on top reranker score
                │
        Claude on AWS Bedrock
        (system prompt + confidence signal + top 5 chunks)
                │
        Streamed response with inline [Source N] citations
```

**Hybrid retrieval** combines BM25 (keyword matching) and dense vector search (Amazon Titan embeddings in ChromaDB) via Reciprocal Rank Fusion. Neither alone is sufficient — BM25 catches exact name matches, vector search catches semantic meaning.

**Cross-encoder reranking** re-scores the fused candidates with a more expensive but accurate model, picking the 5 passages most likely to answer the specific query.

**Confidence signals** convert the top reranker score into a prompt instruction (`high / moderate / low / none`). Low-confidence queries — typos, obscure characters, weak matches — trigger a specific Claude behaviour: don't guess, tell the user what couldn't be found, suggest corrections if it looks like a misspelling.

---

## Features

- **Token-by-token streaming** — response renders as Claude generates it, not after
- **Inline citations** — `[Source N]` markers woven naturally into prose, rendered as page-number pills in the UI
- **Source cards** — each cited passage shown below the answer with page reference
- **Multi-turn conversation** — last 6 turns of chat history passed to the LLM
- **AWS Bedrock Guardrails** — input pre-checked before any LLM call; topic policy blocks off-topic questions
- **Confidence-aware prompting** — Claude is instructed differently depending on retrieval quality
- **RAGAS evaluation** — automated quality gate (faithfulness, answer relevancy, context precision) runs on every push via GitHub Actions

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Claude (Haiku / Sonnet) via AWS Bedrock |
| Embeddings | Amazon Titan Embed Text v2 |
| Vector store | ChromaDB (local) |
| Keyword search | BM25 (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Orchestration | LangChain (`langchain-aws`) |
| Frontend | Streamlit |
| Evaluation | RAGAS 0.4 |
| CI/CD | GitHub Actions |
| Config | python-dotenv |

---

## Project structure

```
acotar-chatbot/
├── app.py                    # Streamlit entrypoint
├── citation_chain.py         # LLM call, streaming, prompt construction
├── retriever.py              # Hybrid retrieval + reranking + confidence
├── ingest.py                 # PDF → chunks → ChromaDB
├── config.py                 # All config loaded from .env
├── ui.py                     # Theme, components, citation pill renderer
├── test_retrieval.py         # Manual retrieval smoke test
├── evaluation/
│   ├── evaluate.py           # RAGAS evaluation runner
│   ├── code_grader.py        # Exact-match grader for factual questions
│   └── eval_dataset.json     # Labelled Q&A pairs for evaluation
├── guardrails/
│   └── setup_guardrail.py    # One-time Bedrock guardrail setup script
├── .github/workflows/
│   └── evaluate.yml          # CI pipeline — RAGAS quality gate on push
├── .env.example              # Required environment variables (no real values)
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11
- AWS account with Bedrock access enabled for Claude and Amazon Titan models
- AWS credentials configured locally (`aws configure` or environment variables)

### Install

```bash
git clone https://github.com/your-username/acotar-chatbot.git
cd acotar-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

```env
AWS_REGION=us-east-1
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
LLM_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
TEMPERATURE=0.3
MAX_TOKENS=700
GUARDRAIL_ID=your-guardrail-id        # optional — see guardrails/ setup
GUARDRAIL_VERSION=1
AUTHOR_NAME=Your Name                  # optional — shown in sidebar
```

### Ingest the book

Place the PDF at `data/ACOTAR_MistAndFury.pdf`, then run:

```bash
python ingest.py
```

This chunks the PDF, embeds each chunk via Amazon Titan, and stores the vectors in `chroma_db/`. Run once — the vector store is cached locally.

### Create a guardrail (optional)

```bash
python guardrails/setup_guardrail.py
```

Prints a `GUARDRAIL_ID` and `GUARDRAIL_VERSION` to add to `.env`. The guardrail blocks off-topic queries and PII at the AWS layer before the LLM is called.

### Run the app

```bash
streamlit run app.py
```

---

## Evaluation

The evaluation pipeline tests retrieval and generation quality against a labelled dataset of ACOMAF questions.

**Factual questions** (exact character names, places, events) go through a code grader that checks for keyword matches without calling an LLM.

**Descriptive and synthesis questions** are escalated to RAGAS, which scores:

| Metric | Minimum threshold |
|---|---|
| Faithfulness | 0.7 |
| Answer relevancy | 0.7 |
| Context precision | 0.6 |

Run locally:

```bash
python evaluation/evaluate.py
```

### CI/CD

The GitHub Actions workflow (`.github/workflows/evaluate.yml`) runs the full evaluation on every push to `main`. It:

1. Restores the ChromaDB vector store from cache (keyed on `pdf_version.txt`)
2. Downloads the PDF from S3 and re-ingests only on cache miss
3. Runs `evaluate.py` and enforces the RAGAS thresholds as a quality gate
4. Uploads results as a build artifact

AWS credentials are stored as GitHub repository secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `S3_BUCKET`). No credentials are hardcoded anywhere in the codebase.

---

## Confidence thresholds

The reranker score of the top-ranked passage maps to a confidence level that changes how Claude responds:

| Top score | Confidence | Claude behaviour |
|---|---|---|
| > 0.3 | `high` | Answer normally |
| 0.1 – 0.3 | `moderate` | Answer but flag uncertainty |
| < 0.1 | `low` | Decline to guess; suggest corrections |
| No results | `none` | Decline to guess; suggest corrections |

Thresholds are set in `retriever.py:score_to_confidence()` and can be tuned as the corpus grows.

---

## License

This project is for educational and portfolio purposes. *A Court of Mist and Fury* is the intellectual property of Sarah J. Maas and Bloomsbury Publishing. The PDF is not included in this repository.
