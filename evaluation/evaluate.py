import json
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import load_vectorstore, build_bm25_index, retrieve
from citation_chain import load_llm, answer_with_citations
from code_grader import grade
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrock, BedrockEmbeddings
from datasets import Dataset
from ragas.run_config import RunConfig

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

    # ── Collect answers from pipeline (unchanged) ─────────
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []
    types         = []  # NEW — track question type for routing

    for i, item in enumerate(dataset):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        q_type       = item.get("type", "descriptive")  # default to descriptive if missing
        print(f"[{i+1}/{len(dataset)}] {question}")

        try:
            chunks          = retrieve(question, vs, bm25, docs, metas)
            answer, sources = answer_with_citations(question, chunks, llm)
            context         = [c["doc"] for c in chunks]

            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            ground_truths.append(ground_truth)
            types.append(q_type)  # NEW

            time.sleep(3)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # ── Route each question to code grader or RAGAS ───────
    code_results  = []   # questions handled by exact match
    ragas_indices = []   # indices that need RAGAS

    print("\nRouting questions...")
    for i, (q, a, gt, t) in enumerate(zip(questions, answers, ground_truths, types)):
        result = grade(q, a, gt, t)
        result["question"] = q

        if not result["escalate"]:
            code_results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [CODE]  {status} | {q[:55]} | {result['detail']}")
        else:
            ragas_indices.append(i)
            print(f"  [RAGAS] routed | {q[:55]}")

    # ── Run RAGAS only on escalated questions ─────────────
    ragas_passed = True   # default if nothing routes to RAGAS

    if ragas_indices:
        print(f"\nRunning RAGAS on {len(ragas_indices)} questions...")
        ragas_dataset = Dataset.from_dict({
            "question":     [questions[i]     for i in ragas_indices],
            "answer":       [answers[i]       for i in ragas_indices],
            "contexts":     [contexts[i]      for i in ragas_indices],
            "ground_truth": [ground_truths[i] for i in ragas_indices]
        })

        time.sleep(8)

        results = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
            run_config=RunConfig(max_workers=1, timeout=60)
        )

        ragas_passed = check_thresholds(results)

    # ── Code grader summary ────────────────────────────────
    code_failures = [r for r in code_results if not r["passed"]]
    code_passed   = len(code_failures) == 0

    print(f"\n[CODE]  {len(code_results) - len(code_failures)}/{len(code_results)} passed")
    if code_failures:
        for f in code_failures:
            # Disagreement log — code failed but was escalated to RAGAS
            # If RAGAS passed this question, ground truth needs better variants
            print(f"  [WARN] code=FAIL escalated to RAGAS: {f['question'][:60]} | {f['detail']}")

    # ── Final quality gate ─────────────────────────────────
    if code_passed and ragas_passed:
        print("\n✓ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n✗ QUALITY GATE FAILED")
        sys.exit(1)


# ─── Check RAGAS thresholds ───────────────────────────────
def check_thresholds(results) -> bool:
    print("\n" + "=" * 50)
    print("RAGAS RESULTS")
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
    return passed


if __name__ == "__main__":
    run_evaluation()