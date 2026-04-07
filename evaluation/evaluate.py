import json
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import load_vectorstore, build_bm25_index, retrieve
from citation_chain import load_llm, answer_with_citations
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrock, BedrockEmbeddings
from datasets import Dataset

# ─── Configuration ───────────────────────────────────────
EVAL_FILE        = "evaluation/eval_dataset.json"
FAITHFULNESS_MIN = 0.7
RELEVANCY_MIN    = 0.7
PRECISION_MIN    = 0.6

# ─── Load evaluation dataset ─────────────────────────────
def load_eval_dataset():
    with open(EVAL_FILE) as f:
        return json.load(f)

# ─── Run evaluation ───────────────────────────────────────
def run_evaluation():
    print("Loading components...")
    vs                = load_vectorstore()
    bm25, docs, metas = build_bm25_index(vs)
    llm               = load_llm()

    # Tell RAGAS to use Bedrock instead of OpenAI
    ragas_llm = LangchainLLMWrapper(ChatBedrock(
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region_name="us-east-1"
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    ))

    dataset = load_eval_dataset()
    dataset = dataset[:5]  # remove this line once everything works
    print(f"Loaded {len(dataset)} evaluation questions\n")

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, item in enumerate(dataset):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i+1}/{len(dataset)}] {question}")

        try:
            chunks          = retrieve(question, vs, bm25, docs, metas)
            answer, sources = answer_with_citations(question, chunks, llm)
            context         = [c["doc"] for c in chunks]

            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            ground_truths.append(ground_truth)

            time.sleep(3)  # wait 3 seconds between questions

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\nRunning RAGAS evaluation...")
    ragas_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        run_config={"max_workers": 1}  # forces sequential, no parallel calls
    )

    return results

# ─── Check thresholds ─────────────────────────────────────
def check_thresholds(results):
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    scores = {
        "faithfulness":      results["faithfulness"],
        "answer_relevancy":  results["answer_relevancy"],
        "context_precision": results["context_precision"]
    }

    thresholds = {
        "faithfulness":      FAITHFULNESS_MIN,
        "answer_relevancy":  RELEVANCY_MIN,
        "context_precision": PRECISION_MIN
    }

    passed = True
    for metric, score in scores.items():
        threshold = thresholds[metric]
        status    = "✓ PASS" if score >= threshold else "✗ FAIL"
        if score < threshold:
            passed = False
        print(f"  {status}  {metric}: {score:.3f}  (min: {threshold})")

    print("=" * 50)
    if passed:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ QUALITY GATE FAILED")
        sys.exit(1)

if __name__ == "__main__":
    results = run_evaluation()
    check_thresholds(results)