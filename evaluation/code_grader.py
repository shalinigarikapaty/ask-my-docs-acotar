import re

def extract_key_terms(ground_truth: str) -> list:
    """
    Extract proper nouns from ground truth by looking for capitalised words.
    'Rhysand is Feyre's mate' → ['Rhysand', 'Feyre']
    """
    words = re.findall(r'\b[A-Z][a-z]+\b', ground_truth)
    stopwords = {"They", "The", "His", "Her", "He", "She", "It", "This", "That", "Their"}
    return [w for w in words if w not in stopwords]


def exact_match(predicted: str, ground_truth: str) -> dict:
    """
    Check if key terms from ground truth appear in predicted answer.
    Returns score and which terms were found/missing.
    """
    key_terms    = extract_key_terms(ground_truth)
    predicted_lower = predicted.lower()

    found   = [t for t in key_terms if t.lower() in predicted_lower]
    missing = [t for t in key_terms if t.lower() not in predicted_lower]

    score  = len(found) / len(key_terms) if key_terms else 0.0
    passed = score == 1.0

    return {
        "score":   score,
        "passed":  passed,
        "found":   found,
        "missing": missing
    }


def grade(question: str, predicted: str, ground_truth: str, question_type: str) -> dict:
    """
    Route question to correct grader.
    Factual → exact match first, escalate to RAGAS on failure.
    Everything else → RAGAS directly.
    """
    if question_type == "factual":
        result = exact_match(predicted, ground_truth)
        if result["passed"]:
            return {
                "grader":   "code",
                "score":    result["score"],
                "passed":   True,
                "escalate": False,
                "detail":   f"matched: {result['found']}"
            }
        else:
            return {
                "grader":   "code",
                "score":    result["score"],
                "passed":   False,
                "escalate": True,
                "detail":   f"missing: {result['missing']}"
            }

    # descriptive, analytical, event, multi-turn → RAGAS directly
    return {
        "grader":   "ragas",
        "score":    None,
        "passed":   None,
        "escalate": True,
        "detail":   "routed to RAGAS"
    }