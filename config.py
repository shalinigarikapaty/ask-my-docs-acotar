import os
from dotenv import load_dotenv

load_dotenv()

AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
LLM_MODEL         = os.getenv("LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
TEMPERATURE       = float(os.getenv("TEMPERATURE", "0.3"))
MAX_TOKENS        = int(os.getenv("MAX_TOKENS", "700"))
GUARDRAIL_ID      = os.getenv("GUARDRAIL_ID")
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION", "1")
CHROMA_DIR        = os.getenv("CHROMA_DIR", "./chroma_db")
AUTHOR_NAME       = os.getenv("AUTHOR_NAME", "")
