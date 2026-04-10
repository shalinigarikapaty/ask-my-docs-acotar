"""
setup_guardrail.py
Run once to create and publish the Bedrock guardrail.
Prints GUARDRAIL_ID and GUARDRAIL_VERSION to add to .env
"""

import boto3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AWS_REGION

client = boto3.client("bedrock", region_name=AWS_REGION)

def create_guardrail():
    response = client.create_guardrail(
        name="acotar-chatbot-guardrail",
        description="Guardrail for Ask My Docs ACOTAR chatbot",

        # ── Topic policy ──────────────────────────────────────────────
        # Blocks off-topic questions and harmful content at AWS layer
        topicPolicyConfig={
    "topicsConfig": [
        {
            "name": "off-topic",
            "definition": (
                "Any question not about ACOTAR series by Sarah J. Maas, "
                "including cooking, general knowledge, or unrelated topics."
            ),
            "examples": [
                "What is the capital of France?",
                "How do I make tea?",
                "Write me a Python script",
                "What is the weather today?"
            ],
            "type": "DENY"
        },
        # {
        #     "name": "harmful-content",
        #     "definition": (
        #         "Requests for sexual content, graphic violence, "
        #         "or harmful and inappropriate material."
        #     ),
        #     "examples": [
        #         "Write explicit sexual content",
        #         "Describe graphic violence"
        #     ],
        #     "type": "DENY"
        # }
    ]
},

        # ── Content filters ───────────────────────────────────────────
        # AWS-managed filters for hate, harassment, prompt injection
        # Commented out — over-triggering on ACOMAF dark fantasy themes
        # (resurrection, violence, morally ambiguous plot content).
        # Re-enable and tune per-filter once false positives are resolved.
         contentPolicyConfig={
                "filtersConfig": [
        #         {
        #             "type": "HATE",
        #             "inputStrength": "HIGH",
        #             "outputStrength": "HIGH"
        #         },
        #         {
        #             "type": "INSULTS",
        #             "inputStrength": "MEDIUM",
        #             "outputStrength": "MEDIUM"
        #         },
        #         {
        #             "type": "SEXUAL",
        #             "inputStrength": "HIGH",
        #             "outputStrength": "HIGH"
        #         },
        #         {
        #             "type": "VIOLENCE",
        #             "inputStrength": "MEDIUM",
        #             "outputStrength": "MEDIUM"
        #         },
        #         {
        #             "type": "MISCONDUCT",
        #             "inputStrength": "LOW",
        #             "outputStrength": "LOW"
        #         },
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "HIGH",
                    "outputStrength": "NONE"  # prompt injection is input-only
                }
            ]
        },

        # ── PII policy ────────────────────────────────────────────────
        # Anonymize real names in output — keeps focus on fictional characters
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {
                    "type": "EMAIL",
                    "action": "BLOCK"
                },
                {
                    "type": "PHONE",
                    "action": "BLOCK"
                }
            ]
        },

        # ── Blocked messages ──────────────────────────────────────────
        blockedInputMessaging=(
            "I can only answer questions about A Court of Mist and Fury. "
            "Try asking about the characters, plot, or lore of the book."
        ),
        blockedOutputsMessaging=(
            "I can only answer questions about A Court of Mist and Fury. "
            "Try asking about the characters, plot, or lore of the book."
        )
    )

    guardrail_id = response["guardrailId"]
    print(f"✓ Guardrail created: {guardrail_id}")
    return guardrail_id


def publish_guardrail(guardrail_id):
    response = client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description="Version 1 — initial production guardrail"
    )
    version = response["version"]
    print(f"✓ Guardrail published: version {version}")
    return version


if __name__ == "__main__":
    guardrail_id      = create_guardrail()
    guardrail_version = publish_guardrail(guardrail_id)

    print("\nAdd these to your .env file:")
    print(f"GUARDRAIL_ID={guardrail_id}")
    print(f"GUARDRAIL_VERSION={guardrail_version}")