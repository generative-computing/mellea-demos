"""Application configuration from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()

LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFLOW_BASE_URL = os.getenv("LANGFLOW_BASE_URL", "http://localhost:7862")
LANGFLOW_API_KEY = os.getenv("LANGFLOW_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug")
