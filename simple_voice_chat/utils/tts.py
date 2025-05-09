import asyncio
import io
import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict, Any  # Added Dict, Any

import numpy as np
import requests
from loguru import logger
from openai import OpenAI  # Assuming OpenAI client is used, adjust if different
from pydub import AudioSegment


def get_voices(tts_base_url: str, api_key: Optional[str]) -> List[str]:
    """Fetches available TTS voices from the server."""
    voices = []
    voices_url = f"{tts_base_url}/audio/voices"  # Correct endpoint path
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        logger.info(f"Fetching available TTS voices: {voices_url}")
        response = requests.get(
            voices_url, headers=headers, timeout=10
        )  # Added timeout
        response.raise_for_status()
        data = response.json()

        if "voices" in data and isinstance(data["voices"], list):
            voices = data["voices"]
            logger.info(f"Successfully fetched {len(voices)} voices.")
            logger.debug(f"Available voices: {voices}")
        else:
            logger.error(
                f"Unexpected format in response from {voices_url}: 'voices' key missing or not a list."
            )

    except requests.exceptions.RequestException as e:
        logger.error(
            f"HTTP error fetching voices from {voices_url}: {e}"
        )  # Loguru captures traceback by default on error level if configured, or use logger.exception()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {voices_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching voices: {e}")

    return voices


def prepare_available_voices_data(
    current_voice: str, available_voices: List[str]
) -> Dict[str, Any]:
    """Prepares the data structure for the /available_voices_tts endpoint."""
    return {
        "available": available_voices,
        "current": current_voice,
    }


# --- TTS Helper Function (Returns audio data, doesn't yield) ---
async def generate_tts_for_sentence(
    text: str,
    tts_client: OpenAI,  # Pass the client instance
    tts_model: str,  # Pass the model name
    selected_voice: str,  # Pass the selected voice
    tts_speed: float,  # Pass the desired speed
    acronym_preserve_set: Set[str],  # Pass the set of acronyms to preserve
    temp_dir: Path,  # Directory to save temporary audio files
) -> str | None: # Changed return type hint to str | None
    """
    Generates TTS for the given text, saves it to a temporary MP3 file,
    and returns the filename string or None on failure.
    """

    if not text or text.isspace():
        logger.warning(
            "generate_tts_for_sentence called with empty or whitespace text. Skipping."
        )
        return None

    # --- Preprocess text for TTS (Conditional based on model name) ---
    # Start with the original text
    # Remove markdown bullet points (e.g., "* ", "- ", "+ ") from the beginning of lines
    # Handles optional leading whitespace before the bullet.
    processed_text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    logger.debug(f"Text after markdown bullet stripping: '{processed_text[:100]}...'")


    if "kokoro" in tts_model:
        logger.debug(
            f"Applying 'kokoro'-specific text preprocessing for model: {tts_model}"
        )

        # --- Acronym Processing ---
        # Space out fully capitalized words (e.g., "HTA" -> "H T A") unless in the preserve set
        def acronym_replacer(match: re.Match) -> str:
            """Decides whether to split or preserve an acronym based on acronym_preserve_set."""
            acronym = match.group(1)
            if acronym in acronym_preserve_set:
                logger.debug(f"Preserving acronym: {acronym}")
                return acronym  # Preserve the acronym
            else:
                logger.debug(f"Splitting acronym: {acronym}")
                return " ".join(acronym)  # Split into letters

        # Use word boundaries (\b) to avoid affecting single caps or mixed case words.
        # Apply this processing step by step
        temp_processed_text = re.sub(
            r"\b([A-Z]{2,})\b", acronym_replacer, processed_text
        )
        logger.debug(f"Text after acronym processing: '{temp_processed_text[:50]}...'")

        # --- Newline Insertion ---
        # Add extra newline after sentence endings (.?!) followed by a space
        temp_processed_text = re.sub(r"([.?!]) +", r"\1\n\n", temp_processed_text)
        logger.debug(f"Text after newline insertion: '{temp_processed_text[:50]}...'")

        # --- Bold Replacement ---
        # Replace **bold** with BOLD (applied to potentially spaced-out text)
        temp_processed_text = re.sub(
            r"\*\*(.*?)\*\*", lambda m: m.group(1).upper(), temp_processed_text
        )
        logger.debug(f"Text after bold replacement: '{temp_processed_text[:50]}...'")

        # --- Italic Replacement ---
        # Replace *italic* with , italic, for pauses (applied after bold)
        temp_processed_text = re.sub(r"\*(.*?)\*", r", \1, ", temp_processed_text)
        logger.debug(f"Text after italic replacement: '{temp_processed_text[:50]}...'")

        # Assign the final processed text
        processed_text = temp_processed_text

    else:
        logger.debug(
            f"Skipping 'kokoro'-specific text preprocessing for model: {tts_model}"
        )
    # --- End Preprocessing ---

    start_tts = time.time()
    max_tts_retries = 3
    last_tts_exception = None

    logger.debug(
        f"Starting TTS task for processed sentence (first 50 chars): '{processed_text[:50]}...'"
    )

    for attempt in range(max_tts_retries):
        logger.debug(
            f"TTS attempt {attempt + 1}/{max_tts_retries} ({selected_voice})..."
        )
        try:
            tts_response = (
                await asyncio.to_thread(  # Run blocking OpenAI call in thread
                    tts_client.audio.speech.create,
                    model=tts_model,  # Use parameter
                    voice=selected_voice,  # Use parameter
                    input=processed_text,  # Use processed text (original or modified)
                    response_format="mp3",
                    speed=tts_speed,  # Use parameter
                )
            )

            # Collect TTS audio bytes in memory
            tts_audio_bytes = io.BytesIO()
            byte_count = 0
            for chunk in tts_response.iter_bytes(chunk_size=4096):
                if chunk:
                    tts_audio_bytes.write(chunk)
                    byte_count += len(chunk)

            if byte_count > 0:
                tts_audio_bytes.seek(0)

                # Create a non-deleting temporary file in the specified directory
                # Use a unique name to avoid collisions
                try:
                    # Ensure temp_dir exists (it should, but double-check)
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    # Generate a unique filename
                    temp_filename = f"tts_{uuid.uuid4()}.mp3"
                    temp_filepath = temp_dir / temp_filename

                    # Write the bytes to the file asynchronously if possible, or use thread
                    # Using a simple synchronous write here for simplicity, consider async file I/O if performance critical
                    with open(temp_filepath, "wb") as f:
                         f.write(tts_audio_bytes.getvalue()) # Write all bytes at once

                    logger.info(
                        f"Finished TTS task for processed sentence after {time.time() - start_tts:.2f}s on attempt {attempt + 1} ({byte_count} bytes), saved to '{temp_filepath}'"
                    )
                    return temp_filename # Return the filename string

                except Exception as file_e:
                    logger.error(f"Failed to save TTS audio to temporary file '{temp_filepath}': {file_e}")
                    last_tts_exception = file_e # Store file saving error as the last exception
                    # Fall through to retry logic if applicable

            else:
                logger.warning(
                    f"TTS generation attempt {attempt + 1} produced no audio bytes for processed sentence: '{processed_text[:50]}...'"
                )
                last_tts_exception = RuntimeError("TTS produced no audio bytes")

        except Exception as e:
            logger.error(
                f"Error during TTS generation/decoding attempt {attempt + 1} for processed sentence '{processed_text[:50]}...': {e}"
            )
            last_tts_exception = e

        if attempt < max_tts_retries - 1:
            await asyncio.sleep(0.5 * (attempt + 1))  # Async sleep

    # If loop finishes without success
    error_msg = f"TTS failed after {max_tts_retries} attempts for processed sentence: '{processed_text[:50]}...'"
    if last_tts_exception:
        error_msg += f" Last error: {last_tts_exception}"
    logger.error(error_msg)
    return None  # Return None on failure
