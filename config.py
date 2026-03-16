import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model: configurable via MODEL_NAME env var, defaults to Llama 3.1 405B
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3.1-405b-instruct")

# Inference settings
TEMPERATURE = 0.2
MAX_TOKENS = 1024

# System prompt for the support assistant
SYSTEM_PROMPT = (
    "You are a helpful AI support assistant. "
    "Answer user questions clearly and concisely. "
    "If you don't know the answer, say so honestly."
)
