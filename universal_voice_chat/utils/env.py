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

# --- LLM Configuration ---
LLM_HOST_ENV: Optional[str] = os.getenv("LLM_HOST")
LLM_PORT_ENV: Optional[str] = os.getenv("LLM_PORT")
DEFAULT_LLM_MODEL_ENV: str = os.getenv(
    "LLM_MODEL", "openrouter/google/gemini-2.5-pro-preview-03-25"  # Example default
)
LLM_API_KEY_ENV: Optional[str] = os.getenv("LLM_API_KEY")

# --- STT Configuration ---
# Defaults changed to OpenAI STT
STT_HOST_ENV: str = os.getenv("STT_HOST", "api.openai.com")  # Default to OpenAI host
STT_PORT_ENV: str = os.getenv("STT_PORT", "443")  # Default to HTTPS port
STT_MODEL_ENV: str = os.getenv("STT_MODEL", "whisper-1")  # Default to OpenAI STT model
STT_LANGUAGE_ENV: Optional[str] = os.getenv("STT_LANGUAGE")
STT_API_KEY_ENV: Optional[str] = os.getenv("STT_API_KEY")  # OpenAI requires an API key

# --- STT Confidence Thresholds ---
# Keep as strings for argparse defaults, conversion happens in main script
STT_NO_SPEECH_PROB_THRESHOLD_ENV: str = os.getenv("STT_NO_SPEECH_PROB_THRESHOLD", "0.6")
STT_AVG_LOGPROB_THRESHOLD_ENV: str = os.getenv("STT_AVG_LOGPROB_THRESHOLD", "-0.7")
STT_MIN_WORDS_THRESHOLD_ENV: str = os.getenv("STT_MIN_WORDS_THRESHOLD", "5")

# --- TTS Configuration ---
# Defaults changed to OpenAI TTS
TTS_HOST_ENV: str = os.getenv("TTS_HOST", "api.openai.com")  # Default to OpenAI host
TTS_PORT_ENV: str = os.getenv("TTS_PORT", "443")  # Default to HTTPS port
TTS_MODEL_ENV: str = os.getenv(
    "TTS_MODEL", "tts-1"
)  # Default to standard OpenAI TTS model (tts-1 is cheaper, tts-1-hd is higher quality)
DEFAULT_VOICE_TTS_ENV: str = os.getenv("TTS_VOICE", "ash")  # Default to an OpenAI voice
TTS_API_KEY_ENV: Optional[str] = os.getenv(
    "TTS_API_KEY"
)  # OpenAI requires an API key, ensure this is set in .env
DEFAULT_TTS_SPEED_ENV: str = os.getenv(
    "TTS_SPEED", "1.0"
)  # Keep as string for argparse default

# --- TTS Acronym Preservation ---
TTS_ACRONYM_PRESERVE_LIST_ENV: str = os.getenv("TTS_ACRONYM_PRESERVE_LIST", "")

# --- Application Configuration ---
APP_PORT_ENV: str = os.getenv("APP_PORT", "7860")  # Keep as string for argparse default

# --- System Message ---
# Default to None if the environment variable is not set.
# Handling (e.g., defaulting to "") is done in the main script.
SYSTEM_MESSAGE_ENV: Optional[str] = os.getenv("SYSTEM_MESSAGE")


# --- Exported Variables ---
# These are the names available for import into other modules.
__all__ = [
    # LLM
    "LLM_HOST_ENV",
    "LLM_PORT_ENV",
    "DEFAULT_LLM_MODEL_ENV",
    "LLM_API_KEY_ENV",
    # STT
    "STT_HOST_ENV",
    "STT_PORT_ENV",
    "STT_MODEL_ENV",
    "STT_LANGUAGE_ENV",
    "STT_API_KEY_ENV",
    # STT Thresholds
    "STT_NO_SPEECH_PROB_THRESHOLD_ENV",
    "STT_AVG_LOGPROB_THRESHOLD_ENV",
    "STT_MIN_WORDS_THRESHOLD_ENV",
    # TTS
    "TTS_HOST_ENV",
    "TTS_PORT_ENV",
    "TTS_MODEL_ENV",
    "DEFAULT_VOICE_TTS_ENV",
    "TTS_API_KEY_ENV",
    "DEFAULT_TTS_SPEED_ENV",
    # TTS Acronyms
    "TTS_ACRONYM_PRESERVE_LIST_ENV",
    # App
    "APP_PORT_ENV",
    # Misc
    "SYSTEM_MESSAGE_ENV",
]
