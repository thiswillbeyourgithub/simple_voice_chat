"""
Handles loading environment variables using dotenv and provides access to their raw values.
Configuration logic (like building URLs or choosing defaults) should happen
in the main application script after parsing command-line arguments.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# from loguru import logger # Logging is deferred to main script

# Load environment variables from .env file first
# This makes them available via os.getenv()
load_dotenv()

# --- Provide access to raw environment variables ---
# These functions/variables simply retrieve the value from the environment.
# Default values here are the *ultimate* fallbacks if neither env var nor CLI arg is set.

# --- LLM Configuration (Classic Backend) ---
LLM_HOST_ENV: Optional[str] = os.getenv("LLM_HOST")
LLM_PORT_ENV: Optional[str] = os.getenv("LLM_PORT")
DEFAULT_LLM_MODEL_ENV: str = os.getenv(
    "LLM_MODEL", "openrouter/google/gemini-2.5-pro-preview-03-25"  # Example default
)
LLM_API_KEY_ENV: Optional[str] = os.getenv("LLM_API_KEY") # For classic backend LLM

# --- STT Configuration (Classic Backend) ---
# Defaults changed to OpenAI STT
STT_HOST_ENV: str = os.getenv("STT_HOST", "api.openai.com")  # Default to OpenAI host
STT_PORT_ENV: str = os.getenv("STT_PORT", "443")  # Default to HTTPS port
STT_MODEL_ENV: str = os.getenv("STT_MODEL", "whisper-1")  # Default to OpenAI STT model
STT_LANGUAGE_ENV: Optional[str] = os.getenv("STT_LANGUAGE") # Relevant for both backends
STT_API_KEY_ENV: Optional[str] = os.getenv("STT_API_KEY")  # For classic backend STT (e.g., OpenAI STT)

# --- STT Confidence Thresholds (Classic Backend) ---
# Keep as strings for argparse defaults, conversion happens in main script
STT_NO_SPEECH_PROB_THRESHOLD_ENV: str = os.getenv("STT_NO_SPEECH_PROB_THRESHOLD", "0.6")
STT_AVG_LOGPROB_THRESHOLD_ENV: str = os.getenv("STT_AVG_LOGPROB_THRESHOLD", "-0.7")
STT_MIN_WORDS_THRESHOLD_ENV: str = os.getenv("STT_MIN_WORDS_THRESHOLD", "5")

# --- TTS Configuration (Classic Backend) ---
# Defaults changed to OpenAI TTS
TTS_HOST_ENV: str = os.getenv("TTS_HOST", "api.openai.com")  # Default to OpenAI host
TTS_PORT_ENV: str = os.getenv("TTS_PORT", "443")  # Default to HTTPS port
TTS_MODEL_ENV: str = os.getenv(
    "TTS_MODEL", "tts-1"
)  # Default to standard OpenAI TTS model
DEFAULT_VOICE_TTS_ENV: str = os.getenv("TTS_VOICE", "ash")  # Default to an OpenAI voice for classic backend
TTS_API_KEY_ENV: Optional[str] = os.getenv(
    "TTS_API_KEY"
)  # For classic backend TTS (e.g., OpenAI TTS)
DEFAULT_TTS_SPEED_ENV: str = os.getenv(
    "TTS_SPEED", "1.0"
)  # Keep as string for argparse default

# --- TTS Acronym Preservation (Classic Backend - Kokoro) ---
TTS_ACRONYM_PRESERVE_LIST_ENV: str = os.getenv("TTS_ACRONYM_PRESERVE_LIST", "")

# --- OpenAI Backend Configuration ---
OPENAI_API_KEY_ENV: Optional[str] = os.getenv("OPENAI_API_KEY") # Dedicated key for OpenAI backend
OPENAI_REALTIME_MODEL_ENV: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-mini-realtime-preview-2024-12-17")
OPENAI_REALTIME_VOICE_ENV: str = os.getenv("OPENAI_REALTIME_VOICE", "alloy") # Default voice for OpenAI realtime backend

# --- Application Configuration ---
APP_PORT_ENV: str = os.getenv("APP_PORT", "7860")  # Keep as string for argparse default

# --- System Message ---
SYSTEM_MESSAGE_ENV: Optional[str] = os.getenv("SYSTEM_MESSAGE")


# --- Exported Variables ---
# These are the names available for import into other modules.
__all__ = [
    # LLM (Classic)
    "LLM_HOST_ENV",
    "LLM_PORT_ENV",
    "DEFAULT_LLM_MODEL_ENV",
    "LLM_API_KEY_ENV",
    # STT (Classic, language also for OpenAI backend)
    "STT_HOST_ENV",
    "STT_PORT_ENV",
    "STT_MODEL_ENV",
    "STT_LANGUAGE_ENV",
    "STT_API_KEY_ENV",
    # STT Thresholds (Classic)
    "STT_NO_SPEECH_PROB_THRESHOLD_ENV",
    "STT_AVG_LOGPROB_THRESHOLD_ENV",
    "STT_MIN_WORDS_THRESHOLD_ENV",
    # TTS (Classic)
    "TTS_HOST_ENV",
    "TTS_PORT_ENV",
    "TTS_MODEL_ENV",
    "DEFAULT_VOICE_TTS_ENV",
    "TTS_API_KEY_ENV",
    "DEFAULT_TTS_SPEED_ENV",
    # TTS Acronyms (Classic - Kokoro)
    "TTS_ACRONYM_PRESERVE_LIST_ENV",
    # OpenAI Backend
    "OPENAI_API_KEY_ENV",
    "OPENAI_REALTIME_MODEL_ENV",
    "OPENAI_REALTIME_VOICE_ENV",
    # App
    "APP_PORT_ENV",
    # Misc
    "SYSTEM_MESSAGE_ENV",
]
