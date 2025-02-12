import asyncio
import tempfile
import time
from typing import Tuple, Optional, Any

import numpy as np
from loguru import logger
from openai import OpenAI
from fastrtc.utils import audio_to_bytes


async def transcribe_audio(
    audio: tuple[int, np.ndarray],
    stt_client: OpenAI,
    stt_model: str,
    stt_language: Optional[str],
    stt_api_base: str,
) -> Tuple[bool, str, Any, Optional[str]]:
    """
    Transcribes the given audio using the STT server.

    Args:
        audio: Tuple containing sample rate and audio data as a numpy array.
        stt_client: Initialized OpenAI client pointing to the STT server.
        stt_model: The STT model name to use.
        stt_language: Optional language code for transcription.
        stt_api_base: The base URL of the STT server (for logging).

    Returns:
        A tuple containing:
        - bool: True if transcription was successful, False otherwise.
        - str: The transcribed text (prompt). Empty if unsuccessful or error.
        - Any: The full STT response object (e.g., Transcription object). None if error.
        - Optional[str]: Error message if an exception occurred, None otherwise.
    """
    prompt = ""
    stt_response_obj = None
    error_message = None
    success = False

    try:
        audio_bytes = audio_to_bytes(audio)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_file.flush()
            stt_start_time = time.time()
            logger.debug(f"Sending audio to STT server at {stt_api_base}...")

            with open(temp_audio_file.name, "rb") as audio_file_handle:
                stt_args = {
                    "model": stt_model,
                    "response_format": "verbose_json",
                    "file": audio_file_handle,
                }
                if stt_language:
                    stt_args["language"] = stt_language
                    logger.debug(f"Passing language='{stt_language}' to STT.")

                stt_response_obj = (
                    await asyncio.to_thread(  # Run blocking STT call in thread
                        stt_client.audio.transcriptions.create, **stt_args
                    )
                )
            # Log the raw response for debugging potential issues
            logger.debug(f"Raw STT Response Object: {stt_response_obj}")

            stt_end_time = time.time()
            # Ensure the response object has the 'text' attribute
            if hasattr(stt_response_obj, "text"):
                prompt = stt_response_obj.text
                logger.info(
                    f"Transcription ({stt_end_time - stt_start_time:.2f}s): {prompt}"
                )
                success = True
            else:
                logger.error(
                    f"STT response object lacks 'text' attribute: {stt_response_obj}"
                )
                error_message = "STT response format unexpected (missing text)."
                success = False

    except Exception as e:
        logger.error(f"Error during STT: {e}")
        error_message = str(e)
        success = False

    return success, prompt, stt_response_obj, error_message


def check_stt_confidence(
    stt_response: Any,
    prompt: str,
    no_speech_prob_threshold: float,
    avg_logprob_threshold: float,
    min_words_threshold: int,
) -> Tuple[bool, str]:
    """
    Checks the confidence of the STT result based on configured thresholds.

    Args:
        stt_response: The full STT response object.
        prompt: The transcribed text.
        no_speech_prob_threshold: Threshold for no_speech_prob.
        avg_logprob_threshold: Threshold for avg_logprob.
        min_words_threshold: Minimum number of words required.

    Returns:
        A tuple containing:
        - bool: True if the transcription should be rejected, False otherwise.
        - str: The reason for rejection, or an empty string if not rejected.
    """
    reject_transcription = False
    rejection_reason = ""

    # 1. Word Count Check
    word_count = len(prompt.split())
    if word_count < min_words_threshold:
        reject_transcription = True
        rejection_reason = f"word_count {word_count} < {min_words_threshold}"
        logger.warning(
            f"Rejecting transcription due to word count: {rejection_reason}. Prompt: '{prompt}'"
        )
        return reject_transcription, rejection_reason

    # 2. Confidence Check (only if word count passed and segments exist)
    if hasattr(stt_response, "segments") and stt_response.segments:
        # Check the first segment for now (as Whisper models often return one segment for shorter audio)
        first_segment = stt_response.segments[0]
        # Use getattr for safety, default to values that would pass the check if attribute missing
        no_speech_prob = getattr(first_segment, "no_speech_prob", 0.0)
        avg_logprob = getattr(first_segment, "avg_logprob", 0.0)

        logger.debug(
            f"STT Segment 1 Confidence: no_speech_prob={no_speech_prob:.4f}, avg_logprob={avg_logprob:.4f}"
        )

        if no_speech_prob > no_speech_prob_threshold:
            reject_transcription = True
            rejection_reason = (
                f"no_speech_prob {no_speech_prob:.4f} > {no_speech_prob_threshold}"
            )
        elif avg_logprob < avg_logprob_threshold:
            reject_transcription = True
            rejection_reason = (
                f"avg_logprob {avg_logprob:.4f} < {avg_logprob_threshold}"
            )

        if reject_transcription:
            logger.warning(
                f"Rejecting transcription due to low confidence: {rejection_reason}. Prompt: '{prompt}'"
            )

    elif (
        not reject_transcription
    ):  # Only log warning if word count was ok but segments were missing
        logger.warning(
            "STT response did not contain segments or segments list was empty. Skipping confidence check."
        )

    return reject_transcription, rejection_reason
