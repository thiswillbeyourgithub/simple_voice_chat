# simple_voice_chat/config.py
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI # For client type hints

# --- Application Constants ---
APP_VERSION = "3.3.0" # Matches the version in simple_voice_chat.py
OPENAI_TTS_PRICING = {
    "tts-1": 15.00,
    "tts-1-hd": 30.00,
}
OPENAI_TTS_VOICES = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash",
]
# --- End Application Constants ---


class AppSettings(BaseModel):
    """
    Centralized application settings.
    This object will be populated at startup and used throughout the application.
    """
    # --- Paths ---
    # These are set during startup in main()
    app_log_dir: Optional[Path] = None
    chat_log_dir: Optional[Path] = None
    tts_base_dir: Optional[Path] = None # Base dir for all TTS runs
    tts_audio_dir: Optional[Path] = None # Specific dir for current run's TTS audio

    # --- General App Config ---
    # Populated from args/env in main()
    host: str = "127.0.0.1"
    port: int = 7860 # This will be the actual port used, after checking availability
    preferred_port: int = 7860 # Port from args/env
    verbose: bool = False
    browser: bool = False
    system_message: str = ""
    startup_timestamp_str: Optional[str] = None # For log filenames etc.

    # --- LLM Config ---
    # Populated from args/env and derived in main()
    llm_host_arg: Optional[str] = None # From --llm-host
    llm_port_arg: Optional[str] = None # From --llm-port
    llm_model_arg: Optional[str] = None # From --llm-model (initial preference)
    llm_api_key: Optional[str] = None # From --llm-api-key

    llm_api_base: Optional[str] = None # Derived
    use_llm_proxy: bool = False # Derived
    current_llm_model: Optional[str] = None # Actual current model in use

    available_models: List[str] = Field(default_factory=list)
    model_cost_data: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # --- STT Config ---
    # Populated from args/env and derived in main()
    stt_host_arg: str = "api.openai.com" # From --stt-host
    stt_port_arg: str = "443" # From --stt-port
    stt_model_arg: str = "whisper-1" # From --stt-model
    stt_language_arg: Optional[str] = None # From --stt-language (initial preference)
    stt_api_key: Optional[str] = None # From --stt-api-key

    stt_api_base: Optional[str] = None # Derived
    is_openai_stt: bool = False # Derived
    current_stt_language: Optional[str] = None # Actual current STT language

    stt_no_speech_prob_threshold: float = 0.6
    stt_avg_logprob_threshold: float = -0.7
    stt_min_words_threshold: int = 5

    # --- TTS Config ---
    # Populated from args/env and derived in main()
    tts_host_arg: str = "api.openai.com" # From --tts-host
    tts_port_arg: str = "443" # From --tts-port
    tts_model_arg: str = "tts-1" # From --tts-model
    tts_voice_arg: Optional[str] = None # From --tts-voice (initial preference)
    tts_api_key: Optional[str] = None # From --tts-api-key
    tts_speed_arg: float = 1.0 # From --tts-speed (initial preference)
    tts_acronym_preserve_list_arg: str = "" # Raw string from --tts-acronym-preserve-list

    tts_base_url: Optional[str] = None # Derived
    is_openai_tts: bool = False # Derived
    tts_acronym_preserve_set: Set[str] = Field(default_factory=set) # Derived
    current_tts_voice: Optional[str] = None # Actual current TTS voice
    current_tts_speed: float = 1.0 # Actual current TTS speed

    available_voices_tts: List[str] = Field(default_factory=list)

    # --- Clients ---
    # Initialized in main()
    tts_client: Optional[OpenAI] = None
    stt_client: Optional[OpenAI] = None

    class Config:
        arbitrary_types_allowed = True # To allow OpenAI client type

# Global instance of settings. This will be populated in main().
# Other modules can import this instance.
settings = AppSettings()

# Export specific constants and the settings instance
__all__ = [
    "APP_VERSION",
    "OPENAI_TTS_PRICING",
    "OPENAI_TTS_VOICES",
    "settings",
    "AppSettings", # Export class for type hinting if needed elsewhere
]
