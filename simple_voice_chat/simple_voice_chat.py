import json
import random
import sys
import os # Ensure os is imported early
# Set LITELLM_MODE to PRODUCTION before litellm is imported
os.environ['LITELLM_MODE'] = 'PRODUCTION'
import threading
import time
import datetime
import uvicorn
import webbrowser
import tempfile
import asyncio
import re
import base64 # Ensure it's used if needed by OpenAIRealtimeHandler
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional

from loguru import logger
import litellm
import numpy as np
import webview
import platformdirs
import click # Added click
import openai # Ensure openai is imported for AsyncOpenAI

from fastapi import FastAPI, Request, HTTPException
from google import genai
from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig as GenaiSpeechConfig, # Rename to avoid conflict with our SpeechConfig if any
    VoiceConfig as GenaiVoiceConfig,   # Rename
    ContextWindowCompressionConfig,    # Added for context window compression
    SlidingWindow,                     # Added for context window compression
    # TODO: If specific RecognitionConfig for STT language is found for google-generativeai, import it.
    # from google.ai import generativelanguage as glm -> this requires google-cloud-aiplatform
)
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    FileResponse,
)
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    get_twilio_turn_credentials,
    AlgoOptions,
    SileroVadOptions,
    AsyncStreamHandler, # Added AsyncStreamHandler
    wait_for_item, # Added wait_for_item
)
from gradio.utils import get_space
from openai import OpenAI, AuthenticationError # OpenAI is already here, AuthenticationError too
from pydantic import BaseModel
from pydub import AudioSegment

# --- Import Configuration ---
# This 'settings' instance will be populated in main() and used throughout.
from .utils.config import (
    settings, 
    APP_VERSION, 
    OPENAI_TTS_PRICING, 
    OPENAI_TTS_VOICES, 
    AppSettings, # Added AppSettings for type hint
    OPENAI_REALTIME_MODELS, # Added for OpenAI backend model list
    OPENAI_REALTIME_VOICES, # Added for OpenAI backend
    OPENAI_REALTIME_PRICING, # Updated from OPENAI_REALTIME_PRICING_PER_MINUTE
    GEMINI_LIVE_MODELS,       # Add Gemini model list
    GEMINI_LIVE_VOICES,       # Add Gemini voice list
    GEMINI_LIVE_PRICING,      # Add Gemini pricing
)
# --- End Import Configuration ---


# Import raw environment variable accessors
# These will be used as defaults for argparse arguments
from .utils.env import (
    LLM_HOST_ENV,
    LLM_PORT_ENV,
    DEFAULT_LLM_MODEL_ENV,
    LLM_API_KEY_ENV,
    STT_HOST_ENV,
    STT_PORT_ENV,
    STT_MODEL_ENV,
    STT_LANGUAGE_ENV,
    STT_API_KEY_ENV,
    STT_NO_SPEECH_PROB_THRESHOLD_ENV,
    STT_AVG_LOGPROB_THRESHOLD_ENV,
    STT_MIN_WORDS_THRESHOLD_ENV,
    TTS_HOST_ENV,
    TTS_PORT_ENV,
    TTS_MODEL_ENV,
    DEFAULT_VOICE_TTS_ENV,
    TTS_API_KEY_ENV,
    DEFAULT_TTS_SPEED_ENV,
    TTS_ACRONYM_PRESERVE_LIST_ENV,
    APP_PORT_ENV,
    SYSTEM_MESSAGE_ENV,
    OPENAI_API_KEY_ENV,
    OPENAI_REALTIME_MODEL_ENV,
    OPENAI_REALTIME_VOICE_ENV,
    GEMINI_API_KEY_ENV,       # Add Gemini env var
    GEMINI_MODEL_ENV,         # Add Gemini env var
    GEMINI_VOICE_ENV,         # Add Gemini env var
    GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD_ENV, # Add Gemini threshold env var
    DISABLE_HEARTBEAT_ENV,    # Add disable heartbeat env var
)

# Import other utils functions
from .utils.tts import (
    get_voices,
    generate_tts_for_sentence,
    prepare_available_voices_data,
)
from .utils.llms import (
    get_models_and_costs_from_proxy,
    get_models_and_costs_from_litellm,
    calculate_llm_cost,
)
from .utils.misc import is_port_in_use
from .utils.stt import transcribe_audio, check_stt_confidence
from .utils.logging_config import setup_logging

# --- Global Variables (Runtime State - Not Configuration) ---
# These are primarily for managing the server and UI state, not app settings.
last_heartbeat_time: datetime.datetime | None = None
heartbeat_timeout: int = 15  # Seconds before assuming client disconnected
shutdown_event = threading.Event()  # Used to signal monitor thread to stop
pywebview_window = None  # To hold the pywebview window object if created
uvicorn_server = None # Global variable to hold the Uvicorn server instance
# --- End Global Configuration & State ---

# --- Constants ---
OPENAI_REALTIME_SAMPLE_RATE = 24000
GEMINI_REALTIME_INPUT_SAMPLE_RATE = 16000 # Gemini expects 16kHz input
GEMINI_REALTIME_OUTPUT_SAMPLE_RATE = 24000 # Gemini produces 24kHz output (e.g., Puck voice)
# --- End Constants ---


# --- Chat History Saving Function ---
def save_chat_history(history: List[Dict[str, str]]):
    """Saves the current chat history to a JSON file named after the startup timestamp."""
    if not settings.chat_log_dir or not settings.startup_timestamp_str:
        logger.warning(
            "Chat log directory or startup timestamp not initialized. Cannot save history."
        )
        return

    log_file_path = settings.chat_log_dir / f"{settings.startup_timestamp_str}.json"
    logger.debug(f"Saving chat history to: {log_file_path}")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        logger.debug(f"Successfully saved chat history ({len(history)} messages).")
    except IOError as e:
        logger.error(f"Failed to save chat history to {log_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving chat history: {e}")


# --- Core Response Logic (Async Streaming with Background TTS - CLASSIC BACKEND) ---
async def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
) -> AsyncGenerator[Any, None]:
    """
    Handles audio input, performs STT, streams LLM response text chunks to UI,
    generates TTS concurrently, and yields final audio and updates.
    This function is used by the 'classic' backend.
    """
    # Access module-level variables set after arg parsing in main()
    # Ensure clients are initialized (should be, but good practice)
    if not settings.stt_client or not settings.tts_client:
        logger.error("STT or TTS client not initialized for classic backend. Cannot process request.")
        # Yield error state?
        return

    # Work with a copy to avoid modifying the input list directly and ensure clean state per call
    current_chatbot = (chatbot or []).copy()
    logger.info(
        f"--- Entering response function with history length: {len(current_chatbot)} ---"
    )
    # Extract only role and content for sending to the LLM API
    # Handle both old dict format and new ChatMessage model format during transition if needed,
    # but current_chatbot should ideally contain ChatMessage objects or dicts matching its structure.
    messages = []
    for item in current_chatbot:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        elif hasattr(item, 'role') and hasattr(item, 'content'): # Check if it looks like ChatMessage
             messages.append({"role": item.role, "content": item.content})
        else:
            logger.warning(f"Skipping unexpected item in chatbot history: {item}")


    # Add system message if defined
    if settings.system_message:
        # Prepend system message if not already present (e.g., first turn)
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": settings.system_message})
            logger.debug("Prepended system message to LLM input.")
        elif (
            messages[0].get("role") == "system"
            and messages[0].get("content") != settings.system_message
        ):
            # Update system message if it changed (though this shouldn't happen with current setup)
            messages[0]["content"] = settings.system_message
            logger.debug("Updated existing system message in LLM input.")

    # Signal STT processing start
    yield AdditionalOutputs(
        {
            "type": "status_update",
            "status": "stt_processing",
            "message": "Transcribing...",
        }
    )

    # --- Speech-to-Text using imported function ---
    stt_success, prompt, stt_response_obj, stt_error = await transcribe_audio(
        audio,
        settings.stt_client,
        settings.stt_model_arg, # Use the model name from initial args/env
        settings.current_stt_language,
        settings.stt_api_base,
    )

    if not stt_success:
        logger.error(f"STT failed: {stt_error}")
        stt_error_msg = {
            "role": "assistant",
            "content": f"[STT Error: {stt_error or 'Unknown STT failure'}]",
        }
        yield AdditionalOutputs({"type": "chatbot_update", "message": stt_error_msg})
        # Yield final state and status even on STT error to reset frontend
        logger.warning("Yielding final state after STT error...")
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,
            }  # Yield original state
        )
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (after STT error)",
            }
        )
        logger.info("--- Exiting response function after STT error ---")
        return

    # --- STT Confidence Check using imported function ---
    reject_transcription, rejection_reason = check_stt_confidence(
        stt_response_obj, # The full response object from STT
        prompt, # The transcribed text
        settings.stt_no_speech_prob_threshold,
        settings.stt_avg_logprob_threshold,
        settings.stt_min_words_threshold,
    )
    # Store relevant STT details for potential metadata logging
    stt_metadata_details = {}
    if hasattr(stt_response_obj, 'no_speech_prob'):
        stt_metadata_details['no_speech_prob'] = stt_response_obj.no_speech_prob
    if hasattr(stt_response_obj, 'avg_logprob'):
        stt_metadata_details['avg_logprob'] = stt_response_obj.avg_logprob
    # Add word count if needed: stt_metadata_details['word_count'] = len(prompt.split())

    if reject_transcription:
        logger.warning(f"STT confidence check failed: {rejection_reason}. Details: {stt_metadata_details}")
        # Yield status updates to go back to idle without processing this prompt
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": f"Listening (low confidence: {rejection_reason})",
            }
        )
        # Yield final state (unchanged history)
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,  # Send back original history
            }
        )
        logger.info(
            "--- Exiting response function due to low STT confidence/word count ---"
        )
        return

    # --- Proceed if STT successful and confidence check passed ---
    # Create user message with metadata
    user_metadata = ChatMessageMetadata(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stt_details=stt_metadata_details if stt_metadata_details else None # Add STT details if available
    )
    user_message = ChatMessage(role="user", content=prompt, metadata=user_metadata)

    # Add to the *copy* and yield update (convert to dict for frontend compatibility if needed)
    current_chatbot.append(user_message.model_dump()) # Store as dict in the list
    yield AdditionalOutputs({"type": "chatbot_update", "message": user_message.model_dump()})

    # Update messages list (for LLM) based on the modified copy
    messages.append({"role": user_message.role, "content": user_message.content})

    # Save history after adding user message (save_chat_history expects list of dicts)
    save_chat_history(current_chatbot)

    # --- Streaming Chat Completion & Concurrent TTS ---
    llm_response_stream = None
    full_response_text = ""
    sentence_buffer = "" # Buffer for accumulating raw text including tags
    last_tts_processed_pos = 0 # Tracks the index in the *cleaned* buffer processed for TTS
    final_usage_info = None
    llm_error_occurred = False
    first_chunk_yielded = False # Track if we yielded the first chunk for UI
    response_completed_normally = False  # Track normal completion
    total_tts_chars = 0  # Initialize TTS character counter
    tts_audio_file_paths: List[str] = [] # List to store FILENAMES of generated TTS audio files for this response

    try:
        # Signal waiting for LLM
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "llm_waiting",
                "message": "Waiting for AI...",
            }
        )
        llm_start_time = time.time()
        logger.info(
            f"Sending prompt to LLM ({settings.current_llm_model}) for streaming..."
        )
        llm_args_dict = {
            "model": settings.current_llm_model,
            "messages": messages, # Use the history including the user prompt
            "stream": True,
            "stream_options": {
                "include_usage": True
            },  # Request usage data in the stream
            **({"api_base": settings.llm_api_base} if settings.use_llm_proxy else {}),
        }
        if settings.llm_api_key: # This is for classic backend LLM
            llm_args_dict["api_key"] = settings.llm_api_key

        llm_response_stream = await litellm.acompletion(**llm_args_dict)


        async for chunk in llm_response_stream:
            logger.debug(f"CHUNK: {chunk}")
            # Check for content delta first
            delta_content = None
            if chunk.choices and chunk.choices[0].delta:
                delta_content = chunk.choices[0].delta.content

            if delta_content:
                # Yield text chunk immediately for UI update
                yield AdditionalOutputs(
                    {"type": "text_chunk_update", "content": delta_content}
                )
                first_chunk_yielded = True

                sentence_buffer += delta_content
                full_response_text += delta_content

            # Check for usage information in the chunk (often in the *last* chunk)
            # LiteLLM might attach it differently depending on the provider.
            # Let's check more robustly.
            chunk_usage = getattr(chunk, "usage", None) or getattr(
                chunk, "_usage", None
            )  # Check common attributes
            if chunk_usage:
                # If usage is already a dict, use it, otherwise try .dict()
                if isinstance(chunk_usage, dict):
                    final_usage_info = chunk_usage
                elif hasattr(chunk_usage, "dict"):
                    final_usage_info = chunk_usage.dict()
                else:
                    final_usage_info = vars(chunk_usage)  # Fallback to vars()

                logger.info(
                    f"Captured usage info from LLM chunk: {final_usage_info}"
                )

            # --- Process buffer for TTS ---
            # Clean the *entire* current buffer first, then process new parts ending in newline
            if delta_content: # Only process if there was new content
                buffer_cleaned_for_tts = re.sub(r"<think>.*?</think>", "", sentence_buffer, flags=re.DOTALL)
                new_cleaned_text = buffer_cleaned_for_tts[last_tts_processed_pos:]

                # Find the last newline in the newly added cleaned text
                split_pos = new_cleaned_text.rfind('\n')

                if split_pos != -1:
                    # Extract the chunk ready for TTS (up to and including the last newline)
                    tts_ready_chunk = new_cleaned_text[:split_pos + 1]

                    # Split this chunk into sentences
                    sentences_for_tts = tts_ready_chunk.split('\n')

                    for sentence_for_tts in sentences_for_tts:
                        sentence_for_tts = sentence_for_tts.strip()
                        if sentence_for_tts:
                            # Count characters, log, call generate_tts_for_sentence, handle audio path...
                            total_tts_chars += len(sentence_for_tts)
                            logger.debug(f"Generating TTS for sentence (cleaned): '{sentence_for_tts[:50]}...' ({len(sentence_for_tts)} chars)")
                            audio_file_path: Optional[str] = await generate_tts_for_sentence(
                                sentence_for_tts,
                                settings.tts_client, # Classic backend TTS client
                                settings.tts_model_arg, 
                                settings.current_tts_voice,
                                settings.current_tts_speed,
                                settings.tts_acronym_preserve_set,
                                settings.tts_audio_dir,
                            )
                            if audio_file_path:
                                tts_audio_file_paths.append(audio_file_path)
                                full_audio_file_path = settings.tts_audio_dir / audio_file_path
                                logger.debug(f"TTS audio saved to: {full_audio_file_path}")
                                try:
                                    audio_segment = await asyncio.to_thread(
                                        AudioSegment.from_file, full_audio_file_path, format="mp3"
                                    )
                                    sample_rate = audio_segment.frame_rate
                                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
                                    logger.debug(f"Yielding audio chunk from file '{audio_file_path}' for sentence: '{sentence_for_tts[:50]}...'")
                                    yield (sample_rate, samples)
                                except Exception as read_e:
                                    logger.error(f"Failed to read/decode TTS audio file {full_audio_file_path}: {read_e}")
                            else:
                                logger.warning(f"TTS failed for sentence, skipping audio yield and file save: '{sentence_for_tts[:50]}...'")

                    # Update the position marker for the cleaned buffer
                    last_tts_processed_pos += len(tts_ready_chunk)


        # After the loop, process any remaining text in the cleaned buffer
        buffer_cleaned_for_tts = re.sub(r"<think>.*?</think>", "", sentence_buffer, flags=re.DOTALL)
        remaining_cleaned_text = buffer_cleaned_for_tts[last_tts_processed_pos:].strip()

        if remaining_cleaned_text:
            # Process the final remaining part for TTS
            total_tts_chars += len(remaining_cleaned_text)
            logger.debug(f"Generating TTS for remaining buffer (cleaned): '{remaining_cleaned_text[:50]}...' ({len(remaining_cleaned_text)} chars)")
            audio_file_path: Optional[str] = await generate_tts_for_sentence(
                remaining_cleaned_text,
                settings.tts_client, # Classic backend TTS client
                settings.tts_model_arg, 
                settings.current_tts_voice,
                settings.current_tts_speed,
                settings.tts_acronym_preserve_set,
                settings.tts_audio_dir,
            )
            if audio_file_path:
                tts_audio_file_paths.append(audio_file_path)
                full_audio_file_path = settings.tts_audio_dir / audio_file_path
                logger.debug(f"TTS audio saved to: {full_audio_file_path}")
                try:
                    audio_segment = await asyncio.to_thread(
                        AudioSegment.from_file, full_audio_file_path, format="mp3"
                    )
                    sample_rate = audio_segment.frame_rate
                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.int16)
                    logger.debug(f"Yielding audio chunk from file '{audio_file_path}' for remaining buffer: '{remaining_cleaned_text[:50]}...'")
                    yield (sample_rate, samples)
                except Exception as read_e:
                    logger.error(f"Failed to read/decode TTS audio file {full_audio_file_path}: {read_e}")
            else:
                logger.warning(f"TTS failed for remaining buffer, skipping audio yield and file save: '{remaining_cleaned_text[:50]}...'")


        llm_end_time = time.time()
        logger.info(
            f"LLM streaming finished ({llm_end_time - llm_start_time:.2f}s). Full response length: {len(full_response_text)}"
        )
        logger.info(
            f"Total characters sent to TTS: {total_tts_chars}"
        )  # Log total TTS chars

        # --- Final Updates (After LLM stream and TTS generation/yielding) ---
        response_completed_normally = (
            not llm_error_occurred
        )  # Mark normal completion if no LLM error occurred

        # 1. Cost Calculation (LLM and TTS)
        cost_result = {}  # Initialize cost result dict
        tts_cost = 0.0  # Initialize TTS cost

        # Calculate TTS cost if applicable (Classic backend)
        if settings.is_openai_tts and total_tts_chars > 0:
            tts_model_used = settings.tts_model_arg 
            if tts_model_used in OPENAI_TTS_PRICING:
                price_per_million_chars = OPENAI_TTS_PRICING[tts_model_used]
                tts_cost = (total_tts_chars / 1_000_000) * price_per_million_chars
                logger.info(
                    f"Calculated OpenAI TTS cost for {total_tts_chars} chars ({tts_model_used}): ${tts_cost:.6f}"
                )
            else:
                logger.warning(
                    f"Cannot calculate TTS cost: Pricing unknown for model '{tts_model_used}'."
                )
        elif total_tts_chars > 0:
            logger.info(
                f"TTS cost calculation skipped (not using OpenAI TTS or 0 chars). Total chars: {total_tts_chars}"
            )

        cost_result["tts_cost"] = tts_cost  # Add TTS cost (even if 0) to the result

        # Calculate LLM cost (if usage info available - Classic backend)
        if final_usage_info:
            llm_cost_result = calculate_llm_cost(
                settings.current_llm_model, final_usage_info, settings.model_cost_data
            )
            # Merge LLM cost results into the main cost_result dict
            cost_result.update(llm_cost_result)
            logger.info("LLM cost calculation successful.")
        elif not llm_error_occurred:
            logger.warning(
                "No final usage information received from LLM stream, cannot calculate LLM cost accurately."
            )
            cost_result["error"] = "LLM usage info missing"
            cost_result["model"] = settings.current_llm_model
            # Ensure LLM cost fields are present but potentially zero or marked
            cost_result.setdefault("input_cost", 0.0)
            cost_result.setdefault("output_cost", 0.0)
            cost_result.setdefault(
                "total_cost", 0.0
            ) 

        # Yield combined cost update
        logger.info(f"Yielding combined cost update: {cost_result}")
        yield AdditionalOutputs({"type": "cost_update", "data": cost_result})
        logger.info("Cost update yielded.")

        # 2. Add Full Assistant Text Response to History (to the copy) with Metadata
        assistant_message_obj = None 
        if not llm_error_occurred and full_response_text:
            # Create assistant message metadata
            assistant_metadata = ChatMessageMetadata(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                llm_model=settings.current_llm_model,
                usage=final_usage_info, 
                cost=cost_result, 
                tts_audio_file_paths=tts_audio_file_paths if tts_audio_file_paths else None 
            )
            # Create the full message object
            assistant_message_obj = ChatMessage(
                role="assistant",
                content=full_response_text,
                metadata=assistant_metadata
            )
            assistant_message_dict = assistant_message_obj.model_dump() 

            if not current_chatbot or current_chatbot[-1] != assistant_message_dict:
                current_chatbot.append(assistant_message_dict)
                logger.info(
                    "Full assistant response (with metadata) added to chatbot history copy for next turn."
                )
                # Save history after adding assistant message
                save_chat_history(current_chatbot)
            else:
                logger.info(
                    "Full assistant response already present in history, skipping append."
                )
        elif not llm_error_occurred:
            logger.warning(
                "LLM stream completed but produced no text content. History not updated."
            )

        # 3. Yield Final Chatbot State Update (if response completed normally)
        if response_completed_normally:
            logger.info("Yielding final chatbot state update...")
            yield AdditionalOutputs(
                {
                    "type": "final_chatbot_state",
                    "history": current_chatbot,
                } 
            )
            logger.info("Final chatbot state update yielded.")

        # 4. Yield Final Status Update (always, should be the last yield in the success path)
        final_status_message = (
            "Ready" if response_completed_normally else "Ready (after error)"
        )
        logger.info(f"Yielding final status update ({final_status_message})...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": final_status_message,
            }
        )
        logger.info("Final status update yielded.")

    except AuthenticationError as e: 
        logger.error(f"OpenAI Authentication Error during classic backend processing: {e}")
        response_completed_normally = False
        llm_error_occurred = True 

        error_content = f"\n[Authentication Error: Check your API key ({e})]"
        if not first_chunk_yielded:
            llm_error_msg = {"role": "assistant", "content": error_content.strip()}
            yield AdditionalOutputs(
                {"type": "chatbot_update", "message": llm_error_msg}
            )
        else:
            yield AdditionalOutputs(
                {"type": "text_chunk_update", "content": error_content}
            )

        logger.warning("Yielding final chatbot state (after auth exception)...")
        yield AdditionalOutputs(
            {"type": "final_chatbot_state", "history": current_chatbot}
        )
        logger.warning("Final chatbot state (after auth exception) yielded.")

        logger.warning("Yielding final status update (idle, after auth exception)...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (Auth Error)",
            }
        )
        logger.warning("Final status update (idle, after auth exception) yielded.")

    except Exception as e:
        logger.error(
            f"Error during LLM streaming or TTS processing (classic backend): {type(e).__name__} - {e}", exc_info=True
        )
        response_completed_normally = False 
        llm_error_occurred = True 

        error_content = f"\n[LLM/TTS Error: {type(e).__name__}]" 
        if not first_chunk_yielded:
            llm_error_msg = {"role": "assistant", "content": error_content.strip()}
            yield AdditionalOutputs(
                {"type": "chatbot_update", "message": llm_error_msg}
            )
        else:
            yield AdditionalOutputs(
                {"type": "text_chunk_update", "content": error_content}
            )

        logger.warning("Yielding final chatbot state (after exception)...")
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,
            } 
        )
        logger.warning("Final chatbot state (after exception) yielded.")

        logger.warning("Yielding final status update (idle, after exception)...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (after error)",
            }
        )
        logger.warning("Final status update (idle, after exception) yielded.")

    logger.info(
        f"--- Response function generator finished (Completed normally: {response_completed_normally}) ---"
    )


# --- Pydantic Models ---

class ChatMessageMetadata(BaseModel):
    """Optional metadata associated with a chat message."""
    timestamp: Optional[str] = None 
    llm_model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None # For classic/OpenAI: token counts; For Gemini: char counts
    cost: Optional[Dict[str, Any]] = None 
    stt_details: Optional[Dict[str, Any]] = None 
    tts_audio_file_paths: Optional[List[str]] = None # Classic backend
    output_audio_duration_seconds: Optional[float] = None # OpenAI backend
    raw_openai_usage_events: Optional[List[Dict[str, Any]]] = None # For OpenAI backend to store usage events

class ChatMessage(BaseModel):
    """Represents a single message in the chat history."""
    role: str
    content: str
    metadata: Optional[ChatMessageMetadata] = None

# --- FastAPI Setup ---

class InputData(BaseModel):
    """Model for data received by the /input_hook endpoint."""
    webrtc_id: str
    chatbot: list[ChatMessage] 


# --- Gemini Realtime Handler ---
class GeminiRealtimeHandler(AsyncStreamHandler):
    def __init__(self, app_settings: "AppSettings") -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_REALTIME_OUTPUT_SAMPLE_RATE, # Gemini output is 24kHz
            input_sample_rate=GEMINI_REALTIME_INPUT_SAMPLE_RATE,  # Gemini input is 16kHz
        )
        self.settings = app_settings
        self.client: Optional[genai.Client] = None
        self.session: Optional[genai.live.AsyncLiveConnectSession] = None
        self._input_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue() # For (sample_rate, np.ndarray) or AdditionalOutputs
        self.current_stt_language_code = self.settings.current_stt_language # Store the language code
        self.current_gemini_voice = self.settings.current_gemini_voice

        # State for cost calculation (Gemini Backend - Character based)
        self._current_input_chars: int = 0
        self._current_output_chars: int = 0 # Text that was synthesized
        self._current_input_transcript_parts: List[str] = []
        self._current_output_text_parts: List[str] = []
        self._current_input_audio_duration_this_turn: float = 0.0 # Added for token calculation
        self._current_output_audio_duration_this_turn: float = 0.0 # Added for token calculation
        self._processing_lock = asyncio.Lock() # To protect shared state during event processing
        self._last_seen_usage_metadata: Optional[Any] = None # Stores the most recent usage_metadata

    def copy(self):
        return GeminiRealtimeHandler(self.settings)

    def _reset_turn_usage_state(self):
        self._current_input_chars = 0
        self._current_output_chars = 0
        self._current_input_transcript_parts = []
        self._current_output_text_parts = []
        self._current_input_audio_duration_this_turn = 0.0 # Reset
        self._current_output_audio_duration_this_turn = 0.0 # Reset
        self._last_seen_usage_metadata = None # Reset last_seen_usage_metadata
        logger.debug("GeminiRealtimeHandler: Turn usage state (including _last_seen_usage_metadata) reset.")

    async def _audio_input_stream(self) -> AsyncGenerator[bytes, None]:
        """Yields audio chunks from the input queue for Gemini and accumulates input audio duration."""
        while True:
            try:
                # Wait for audio data, but with a timeout to allow shutdown checks
                audio_chunk = await asyncio.wait_for(self._input_audio_queue.get(), timeout=0.1)
                
                # Accumulate input audio duration (bytes_per_sample = 2 for int16)
                duration_chunk_seconds = len(audio_chunk) / (GEMINI_REALTIME_INPUT_SAMPLE_RATE * 2)
                self._current_input_audio_duration_this_turn += duration_chunk_seconds
                
                yield audio_chunk
                self._input_audio_queue.task_done()
            except asyncio.TimeoutError:
                if self.session is None: # Check if we should break
                    logger.debug("GeminiRealtimeHandler: _audio_input_stream detected inactive session, exiting.")
                    break
            except asyncio.CancelledError:
                logger.debug("GeminiRealtimeHandler: _audio_input_stream cancelled.")
                break


    async def start_up(self):
        logger.info("GeminiRealtimeHandler: Starting up and connecting to Gemini...")
        if not self.settings.gemini_api_key:
            logger.error("GeminiRealtimeHandler: Gemini API Key not configured. Cannot connect.")
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": "Gemini API Key missing."}))
            await self.output_queue.put(None)
            return

        self.client = genai.Client(
            api_key=self.settings.gemini_api_key
        )

        # Update language and voice if changed in settings
        # Language Code Formatting
        formatted_language_code = self.settings.current_stt_language # Use a temporary var from settings
        if formatted_language_code and len(formatted_language_code) == 2:
            formatted_language_code = f"{formatted_language_code}-{formatted_language_code.upper()}"
            logger.info(f"GeminiRealtimeHandler: Formatted 2-letter language code to: {formatted_language_code}")
        
        # Update internal state and log if changed
        if self.current_stt_language_code != formatted_language_code:
            self.current_stt_language_code = formatted_language_code
            logger.info(f"GeminiRealtimeHandler: STT language code set to: {self.current_stt_language_code or 'auto-detect (Gemini default)'}")
        elif not self.current_stt_language_code: # Log even if it was already None/empty
             logger.info(f"GeminiRealtimeHandler: STT language code is not set (auto-detect by Gemini).")


        if self.current_gemini_voice != self.settings.current_gemini_voice:
            self.current_gemini_voice = self.settings.current_gemini_voice
            logger.info(f"GeminiRealtimeHandler: Output voice updated to: {self.current_gemini_voice}")

        speech_config_params: Dict[str, Any] = {
            "voice_config": GenaiVoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(
                    voice_name=self.current_gemini_voice,
                )
            )
        }
        if self.current_stt_language_code: # Only add language_code if it's set
            speech_config_params["language_code"] = self.current_stt_language_code

        # Prepare arguments for LiveConnectConfig
        live_connect_config_args: Dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": GenaiSpeechConfig(**speech_config_params),
            "context_window_compression": ContextWindowCompressionConfig(
                sliding_window=SlidingWindow(), # SlidingWindow takes no arguments
                trigger_tokens=self.settings.gemini_context_window_compression_threshold # Set threshold here
            ),
            "output_audio_transcription": {} # Enable transcription of model's audio output
        }
        logger.info(f"GeminiRealtimeHandler: Context window compression enabled with trigger_token_count={self.settings.gemini_context_window_compression_threshold}.")
        logger.info("GeminiRealtimeHandler: Output audio transcription enabled.")

        if self.settings.system_message:
            logger.info(f"GeminiRealtimeHandler: Preparing system message for LiveConnectConfig: \"{self.settings.system_message[:100]}...\"")
            system_instruction_content = genai.types.Content(
                parts=[genai.types.Part(text=self.settings.system_message)]
            )
            live_connect_config_args["system_instruction"] = system_instruction_content
            # Optional: If you want the system message to appear in the UI immediately via server push
            # This would require waiting for connection and then sending an AdditionalOutput, which is complex here.
        
        live_connect_config = LiveConnectConfig(**live_connect_config_args)

        try:
            self._reset_turn_usage_state()
            selected_model = self.settings.current_llm_model or self.settings.gemini_model_arg # Fallback to arg if current not set
            logger.info(f"GeminiRealtimeHandler: Attempting to connect with model {selected_model}, voice {self.current_gemini_voice or 'default'}.")

            async with self.client.aio.live.connect(model=selected_model, config=live_connect_config) as session:
                self.session = session
                logger.info(f"GeminiRealtimeHandler: Connection established with model {selected_model}, voice {self.current_gemini_voice or 'default'}.")
                
                self._reset_turn_usage_state() # Ensure full state reset before stream starts

                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "stt_processing", "message": "Listening..."}))

                async for result in self.session.start_stream(stream=self._audio_input_stream(), mime_type="audio/pcm"):
                    async with self._processing_lock: # Ensure sequential processing of results for a turn
                        result_j = result.json()
                        if len(result_j) < 1000:
                            logger.debug(f"GeminiRealtime Full Event Structure: {result_j}")
                        else:
                            logger.debug(f"GeminiRealtime Full Event Structure: {result_j[:500]}...{result_j[-500:]}")

                        # Capture usage_metadata if present on *any* event for the current turn
                        if hasattr(result, 'usage_metadata') and result.usage_metadata is not None:
                            self._last_seen_usage_metadata = result.usage_metadata
                            logger.debug(f"GeminiRealtimeHandler: Captured/updated _last_seen_usage_metadata.")

                        # The LiveServerContent is nested inside the 'server_content' attribute of the event
                        live_event_content = getattr(result, 'server_content', None)

                        if not live_event_content:
                            # Handle cases where server_content is not present, 
                            # or if the top-level 'result' itself indicates an error or other state.
                            top_level_error = getattr(result, 'error', None)
                            if top_level_error:
                                logger.error(f"GeminiRealtime API Error (top-level event): Code {top_level_error.code}, Message: {top_level_error.message}")
                                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Gemini Error: {top_level_error.message}"}))
                                error_chat_message = ChatMessage(role="assistant", content=f"[Gemini Error: {top_level_error.message}]")
                                await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": error_chat_message.model_dump()}))
                                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "idle", "message": "Ready (after top-level error)"}))
                                self._reset_turn_usage_state()
                            else:
                                logger.warning(f"GeminiRealtime Event does not contain 'server_content' or is an unhandled type: {result}")
                            continue # Move to the next event

                        # Process STT results from input_transcription
                        srr = getattr(live_event_content, 'input_transcription', None)
                        if srr and srr.transcript:
                            transcript = srr.transcript
                            is_final = srr.is_final
                            logger.debug(f"Gemini STT (from input_transcription): '{transcript}' (Final: {is_final})")
                            self._current_input_transcript_parts.append(transcript)
                            
                            if is_final:
                                full_transcript = "".join(self._current_input_transcript_parts)
                                self._current_input_chars = len(full_transcript)
                                user_message = ChatMessage(
                                    role="user",
                                    content=full_transcript,
                                    metadata=ChatMessageMetadata(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
                                )
                                await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": user_message.model_dump()}))
                                await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "llm_waiting", "message": "AI Responding..."}))

                        # Process model_turn for text and inline_data (audio)
                        model_turn = getattr(live_event_content, 'model_turn', None)
                        if model_turn and model_turn.parts:
                            for part in model_turn.parts:
                                # Process text from part
                                if part.text:
                                    logger.debug(f"Gemini Text from Model Part: '{part.text}'")
                                    self._current_output_text_parts.append(part.text)
                                    # Optionally yield text_chunk_update for faster UI text rendering
                                    # await self.output_queue.put(AdditionalOutputs({"type": "text_chunk_update", "content": part.text}))

                                # Process inline_data (audio) from part
                                inline_data_obj = getattr(part, 'inline_data', None)
                                if inline_data_obj and inline_data_obj.data and 'audio' in inline_data_obj.mime_type:
                                    logger.debug(f"GeminiRealtime: Found audio in model_turn.parts.inline_data ({len(inline_data_obj.data)} bytes, mime_type: {inline_data_obj.mime_type})")
                                    
                                    # Accumulate output audio duration
                                    output_audio_bytes_chunk = len(inline_data_obj.data)
                                    duration_chunk_output_seconds = output_audio_bytes_chunk / (GEMINI_REALTIME_OUTPUT_SAMPLE_RATE * 2)
                                    self._current_output_audio_duration_this_turn += duration_chunk_output_seconds
                                    
                                    audio_data_np = np.frombuffer(inline_data_obj.data, dtype=np.int16)
                                    if audio_data_np.ndim == 1:
                                        audio_data_np = audio_data_np.reshape(1, -1) # Ensure 2D for FastRTC
                                    await self.output_queue.put((GEMINI_REALTIME_OUTPUT_SAMPLE_RATE, audio_data_np))

                        # Process output_audio_transcription if available
                        output_transcription_obj = getattr(live_event_content, 'output_transcription', None)
                        if output_transcription_obj and output_transcription_obj.text:
                            logger.debug(f"GeminiRealtime: Received output audio transcription: '{output_transcription_obj.text}'")
                            self._current_output_text_parts.append(output_transcription_obj.text)
                            # Note: This text is appended to _current_output_text_parts and will be part of the
                            # final assistant message when the turn completes.


                        # Process separate audio_response (often for partial audio chunks or alternative audio stream)
                        audio_response_obj = getattr(live_event_content, 'audio_response', None)
                        if audio_response_obj and audio_response_obj.data:
                            logger.debug(f"GeminiRealtime: Found audio in audio_response ({len(audio_response_obj.data)} bytes)")

                            # Accumulate output audio duration
                            output_audio_bytes_chunk = len(audio_response_obj.data)
                            duration_chunk_output_seconds = output_audio_bytes_chunk / (GEMINI_REALTIME_OUTPUT_SAMPLE_RATE * 2)
                            self._current_output_audio_duration_this_turn += duration_chunk_output_seconds

                            audio_data_np = np.frombuffer(audio_response_obj.data, dtype=np.int16)
                            if audio_data_np.ndim == 1:
                                audio_data_np = audio_data_np.reshape(1, -1) # Ensure 2D for FastRTC
                            await self.output_queue.put((GEMINI_REALTIME_OUTPUT_SAMPLE_RATE, audio_data_np))
                        
                        # Determine if this event signals turn end
                        is_turn_ending_event = False
                        end_of_turn_reason = ""
                        
                        speech_event_details = getattr(live_event_content, 'speech_processing_event', None)
                        logger.debug(f"GeminiRealtime: Inspected speech_event_details. Value: {speech_event_details}. Event type if exists: {speech_event_details.event_type if speech_event_details else 'N/A'}")

                        if speech_event_details and speech_event_details.event_type == "END_OF_SINGLE_UTTERANCE":
                            is_turn_ending_event = True
                            end_of_turn_reason = "END_OF_SINGLE_UTTERANCE"
                            logger.info(f"GeminiRealtime: {end_of_turn_reason} received, will process end of turn.")
                        elif live_event_content and getattr(live_event_content, 'turn_complete', False):
                            # Ensure we are processing an active turn if turn_complete is the trigger
                            if self._current_input_transcript_parts or self._current_output_text_parts or self._last_seen_usage_metadata:
                                is_turn_ending_event = True
                                end_of_turn_reason = "live_event_content.turn_complete"
                                logger.info(f"GeminiRealtime: {end_of_turn_reason} is True, will process end of turn.")
                            else:
                                logger.debug("GeminiRealtime: live_event_content.turn_complete is True, but no significant turn activity detected (no input/output parts, no usage metadata seen). Ignoring as end-of-turn trigger.")
                        
                        if is_turn_ending_event:
                            logger.info(f"GeminiRealtime: Processing end of turn triggered by: {end_of_turn_reason}.")
                            
                            full_input_transcript = "".join(self._current_input_transcript_parts)
                            # _current_input_chars should have been set when STT was final.
                            # If not, set it here for safety, though it might indicate an issue upstream if STT final was missed.
                            if not self._current_input_chars and full_input_transcript:
                                self._current_input_chars = len(full_input_transcript)
                                logger.warning("GeminiRealtime: _current_input_chars was not set prior to turn end processing, setting it now based on accumulated parts.")
                            
                            full_output_text = "".join(self._current_output_text_parts) # Assembled from model_turn parts
                            self._current_output_chars = len(full_output_text)

                            # --- Token Calculation ---
                            # Determine which usage_metadata to use (current event or last seen)
                            current_event_usage_metadata = getattr(result, 'usage_metadata', None)
                            final_usage_metadata_for_turn = None
                            usage_metadata_source_log = "unknown"

                            if current_event_usage_metadata:
                                logger.debug("GeminiRealtime: Found usage_metadata on the current turn-ending event.")
                                final_usage_metadata_for_turn = current_event_usage_metadata
                                usage_metadata_source_log = "current_event"
                            elif self._last_seen_usage_metadata:
                                logger.debug("GeminiRealtime: Using _last_seen_usage_metadata as current turn-ending event lacks it.")
                                final_usage_metadata_for_turn = self._last_seen_usage_metadata
                                usage_metadata_source_log = "_last_seen_usage_metadata"
                            
                            logger.debug(f"GeminiRealtime: EVALUATING final_usage_metadata_for_turn. Is set: {final_usage_metadata_for_turn is not None}. Source: {usage_metadata_source_log if final_usage_metadata_for_turn else 'None'}.")
                            api_prompt_audio_tokens: int = 0
                            api_prompt_text_tokens: int = 0
                            api_response_audio_tokens: int = 0
                            # api_response_text_tokens: int = 0 # If Gemini can respond with text in Live API

                            if final_usage_metadata_for_turn:
                                logger.info(f"GeminiRealtime: Using usage_metadata for cost calculation (Source: {usage_metadata_source_log}). Details: {final_usage_metadata_for_turn}")
                                logger.debug(f"[BACKEND_COST_DEBUG_GEMINI] final_usage_metadata_for_turn: {final_usage_metadata_for_turn}")


                                prompt_details = getattr(final_usage_metadata_for_turn, 'prompt_tokens_details', [])
                                if not prompt_details: 
                                    logger.warning("GeminiRealtime: prompt_tokens_details missing or empty in final_usage_metadata_for_turn. Attempting to use aggregate prompt_token_count.")
                                    top_level_prompt_tokens = getattr(final_usage_metadata_for_turn, 'prompt_token_count', 0)
                                    if top_level_prompt_tokens > 0:
                                        logger.info(f"GeminiRealtime: Using top-level prompt_token_count ({top_level_prompt_tokens}) as prompt_audio_tokens due to missing details.")
                                        api_prompt_audio_tokens = top_level_prompt_tokens # Assuming these are audio unless text modality is specified
                                else: 
                                    for item in prompt_details:
                                        modality = item.modality.name.upper() # Changed: Use .name
                                        token_count = item.token_count
                                        if modality == "AUDIO":
                                            api_prompt_audio_tokens += token_count
                                        elif modality == "TEXT":
                                            api_prompt_text_tokens += token_count
                                
                                response_details = getattr(final_usage_metadata_for_turn, 'response_tokens_details', [])
                                if not response_details: 
                                     logger.warning("GeminiRealtime: response_tokens_details missing or empty in final_usage_metadata_for_turn. Attempting to use aggregate response_token_count.")
                                     top_level_response_tokens = getattr(final_usage_metadata_for_turn, 'response_token_count', 0)
                                     if top_level_response_tokens > 0:
                                         logger.info(f"GeminiRealtime: Using top-level response_token_count ({top_level_response_tokens}) as response_audio_tokens due to missing details.")
                                         api_response_audio_tokens = top_level_response_tokens # Assuming these are audio
                                else: 
                                    for item in response_details:
                                        modality = item.modality.name.upper() # Changed: Use .name
                                        token_count = item.token_count
                                        if modality == "AUDIO":
                                            api_response_audio_tokens += token_count
                                
                                logger.info(
                                    f"GeminiRealtime: Parsed API Tokens from final_usage_metadata_for_turn: "
                                    f"Prompt Audio: {api_prompt_audio_tokens}, Prompt Text: {api_prompt_text_tokens}, "
                                    f"Response Audio: {api_response_audio_tokens}"
                                )
                            else: # final_usage_metadata_for_turn is None
                                logger.warning(
                                    "GeminiRealtime: No usage_metadata available for cost calculation (neither current event nor last_seen). Costs will be $0.00."
                                )

                            # --- Cost Calculation (Using parsed API tokens and GEMINI_LIVE_PRICING) ---
                            prompt_audio_token_cost = 0.0
                            prompt_text_token_cost = 0.0
                            response_audio_token_cost = 0.0 # For TTS

                            if GEMINI_LIVE_PRICING:
                                price_input_audio_per_mil = GEMINI_LIVE_PRICING.get("input_audio_tokens", 0.0)
                                price_input_text_per_mil = GEMINI_LIVE_PRICING.get("input_text_tokens", 0.0)
                                price_output_audio_per_mil = GEMINI_LIVE_PRICING.get("output_audio_tokens", 0.0)
                                # price_output_text_per_mil = GEMINI_LIVE_PRICING.get("output_text_tokens", 0.0) # If text output

                                prompt_audio_token_cost = (api_prompt_audio_tokens / 1_000_000) * price_input_audio_per_mil
                                prompt_text_token_cost = (api_prompt_text_tokens / 1_000_000) * price_input_text_per_mil
                                response_audio_token_cost = (api_response_audio_tokens / 1_000_000) * price_output_audio_per_mil
                                # response_text_token_cost = (api_response_text_tokens / 1_000_000) * price_output_text_per_mil
                            
                            total_gemini_cost = prompt_audio_token_cost + prompt_text_token_cost + response_audio_token_cost # + response_text_token_cost
                            
                            logger.debug(f"[BACKEND_COST_DEBUG_GEMINI] Extracted API Tokens: "
                                         f"Prompt Audio: {api_prompt_audio_tokens}, Prompt Text: {api_prompt_text_tokens}, "
                                         f"Response Audio: {api_response_audio_tokens}")
                            logger.debug(f"[BACKEND_COST_DEBUG_GEMINI] Calculated Costs: "
                                         f"Prompt Audio: ${prompt_audio_token_cost:.8f}, Prompt Text: ${prompt_text_token_cost:.8f}, "
                                         f"Response Audio: ${response_audio_token_cost:.8f}, Total: ${total_gemini_cost:.8f}")

                            logger.debug( # Added debug print for turn costs
                                f"GeminiRealtime Turn Token Costs DEBUG: "
                                f"Prompt Audio: ${prompt_audio_token_cost:.6f} ({api_prompt_audio_tokens} tokens), "
                                f"Prompt Text: ${prompt_text_token_cost:.6f} ({api_prompt_text_tokens} tokens), "
                                f"Response Audio: ${response_audio_token_cost:.6f} ({api_response_audio_tokens} tokens). "
                                f"Turn Total: ${total_gemini_cost:.6f}"
                            )
                            logger.info(
                                f"GeminiRealtime Costs: "
                                f"Prompt Audio: ${prompt_audio_token_cost:.6f} ({api_prompt_audio_tokens} tokens), "
                                f"Prompt Text: ${prompt_text_token_cost:.6f} ({api_prompt_text_tokens} tokens), "
                                f"Response Audio: ${response_audio_token_cost:.6f} ({api_response_audio_tokens} tokens). "
                                f"Total: ${total_gemini_cost:.6f}"
                            )
                            logger.info(
                                f"GeminiRealtime Audio Durations (Informational): Input: {self._current_input_audio_duration_this_turn:.3f}s, Output: {self._current_output_audio_duration_this_turn:.3f}s"
                            )
                            logger.info( 
                                f"GeminiRealtime Char Counts (Informational): Input: {self._current_input_chars}, Output (TTS): {self._current_output_chars}."
                            )
                            
                            cost_data = {
                                "model": self.settings.current_llm_model or self.settings.gemini_model_arg,
                                "prompt_audio_tokens": api_prompt_audio_tokens,
                                "prompt_text_tokens": api_prompt_text_tokens,
                                "response_audio_tokens": api_response_audio_tokens,
                                # "response_text_tokens": api_response_text_tokens,
                                "prompt_audio_cost": prompt_audio_token_cost,
                                "prompt_text_cost": prompt_text_token_cost,
                                "response_audio_cost": response_audio_token_cost, # Cost for output audio (TTS)
                                # "response_text_cost": response_text_token_cost,
                                "total_cost": total_gemini_cost,
                                "input_audio_duration_seconds": round(self._current_input_audio_duration_this_turn, 3),
                                "output_audio_duration_seconds": round(self._current_output_audio_duration_this_turn, 3),
                                "input_chars": self._current_input_chars, 
                                "output_chars": self._current_output_chars,
                                "note": "Costs are based on API-provided token counts per modality from usage_metadata."
                            }
                            logger.debug(f"[BACKEND_COST_DEBUG_GEMINI] cost_data to be sent: {json.dumps(cost_data)}")
                            await self.output_queue.put(AdditionalOutputs({"type": "cost_update", "data": cost_data}))
                            
                            assistant_metadata = ChatMessageMetadata(
                                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                llm_model=self.settings.current_llm_model or self.settings.gemini_model_arg,
                                cost=cost_data,
                                usage={ 
                                    "prompt_audio_tokens": api_prompt_audio_tokens,
                                    "prompt_text_tokens": api_prompt_text_tokens,
                                    "response_audio_tokens": api_response_audio_tokens,
                                    # "response_text_tokens": api_response_text_tokens,
                                    "input_audio_duration_seconds": round(self._current_input_audio_duration_this_turn, 3),
                                    "output_audio_duration_seconds": round(self._current_output_audio_duration_this_turn, 3),
                                    "input_chars": self._current_input_chars, 
                                    "output_chars": self._current_output_chars, 
                                }
                            )
                            assistant_message = ChatMessage(role="assistant", content=full_output_text, metadata=assistant_metadata)
                            await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": assistant_message.model_dump()}))
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "idle", "message": "Ready"}))
                            self._reset_turn_usage_state()

                        # Process errors reported within server_content
                        error_obj_from_content = getattr(live_event_content, 'error', None)
                        if error_obj_from_content:
                            logger.error(f"GeminiRealtime API Error (from server_content): Code {error_obj_from_content.code}, Message: {error_obj_from_content.message}")
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Gemini Error: {error_obj_from_content.message}"}))
                            error_chat_message = ChatMessage(role="assistant", content=f"[Gemini Error: {error_obj_from_content.message}]")
                            await self.output_queue.put(AdditionalOutputs({"type": "chatbot_update", "message": error_chat_message.model_dump()}))
                            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "idle", "message": "Ready (after error in content)"}))
                            self._reset_turn_usage_state()

        except Exception as e:
            logger.error(f"GeminiRealtimeHandler: Connection failed or error during session: {e}", exc_info=True)
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Connection Error: {str(e)}"}))
        finally:
            logger.info("GeminiRealtimeHandler: start_up processing loop finished.")
            # The `async with` statement handles session closure.
            self.session = None
            # Signal end of stream to consumer if not already done by an error
            # Check if output_queue.put(None) is appropriate or if the loop ending does it.
            # The FastRTC stream expects None to terminate processing this handler instance.
            await self.output_queue.put(None)


    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.session:
            return

        _, array = frame # Input rate is self.input_sample_rate (16kHz)
        if array.ndim > 1:
            array = array.squeeze()
        
        if array.dtype != np.int16:
            array = array.astype(np.int16)

        audio_bytes = array.tobytes()
        try:
            await self._input_audio_queue.put(audio_bytes)
        except Exception as e:
            logger.error(f"GeminiRealtimeHandler: Error putting audio to input_queue: {e}")


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)


    async def shutdown(self) -> None:
        logger.info("GeminiRealtimeHandler: Shutting down...")
        # The `async with` context manager in start_up handles session closure.
        # Setting self.session to None here helps signal _audio_input_stream to terminate.
        self.session = None
        
        # Drain queues to prevent deadlocks
        while not self._input_audio_queue.empty():
            try:
                self._input_audio_queue.get_nowait()
                self._input_audio_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self.clear_output_queue() # Use separate method for output queue
        await self.output_queue.put(None) # Ensure emit gets None to terminate
        logger.info("GeminiRealtimeHandler: Shutdown complete.")

    def clear_output_queue(self): # Renamed from clear_queue to avoid confusion
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("GeminiRealtimeHandler: Output queue cleared.")


# --- OpenAI Realtime Handler ---
class OpenAIRealtimeHandler(AsyncStreamHandler):
    def __init__(self, app_settings: "AppSettings") -> None:
        super().__init__(
            expected_layout="mono", 
            output_sample_rate=OPENAI_REALTIME_SAMPLE_RATE,
            input_sample_rate=OPENAI_REALTIME_SAMPLE_RATE, 
        )
        self.settings = app_settings
        self.connection = None
        self.output_queue = asyncio.Queue()
        self.client: Optional[openai.AsyncOpenAI] = None 
        self.current_stt_language = self.settings.current_stt_language
        self.current_openai_voice = self.settings.current_openai_voice

        # State for cost calculation (OpenAI Backend)
        self.current_output_audio_duration_seconds: float = 0.0
        self.current_input_tokens: int = 0
        self.current_output_tokens: int = 0
        self.raw_usage_events_for_turn: List[Dict[str, Any]] = []


    def copy(self):
        return OpenAIRealtimeHandler(self.settings)

    def _reset_turn_usage_state(self):
        self.current_output_audio_duration_seconds = 0.0
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        self.raw_usage_events_for_turn = []
        logger.debug("OpenAIRealtimeHandler: Turn usage state reset.")

    async def start_up(self):
        # Removed import: from openai.types.beta.realtime.sessions import (
        #     SessionUpdateParams,
        #     TurnDetectionServerVad,
        #     InputAudioTranscriptionWhisper
        # )
        logger.info("OpenAIRealtimeHandler: Starting up and connecting to OpenAI...")
        if not self.settings.openai_api_key:
            logger.error("OpenAIRealtimeHandler: OpenAI API Key not configured. Cannot connect.")
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": "OpenAI API Key missing."}))
            await self.output_queue.put(None) 
            return

        self.client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key) 

        # Update language and voice if changed in settings
        if self.current_stt_language != self.settings.current_stt_language:
            self.current_stt_language = self.settings.current_stt_language
            logger.info(f"OpenAIRealtimeHandler: STT language updated to: {self.current_stt_language or 'auto-detect'}")
        
        if self.current_openai_voice != self.settings.current_openai_voice:
            self.current_openai_voice = self.settings.current_openai_voice
            logger.info(f"OpenAIRealtimeHandler: Output voice updated to: {self.current_openai_voice}")

        # --- Prepare session parameters using typed objects ---
        # turn_detection_config = TurnDetectionServerVad() # Replaced with dict

        transcription_model_args: Dict[str, Any] = {"model": "whisper-1"}
        if self.current_stt_language:
            transcription_model_args["language"] = self.current_stt_language
        # input_audio_transcription_config = InputAudioTranscriptionWhisper(**transcription_model_args) # Replaced with dict
        
        # Parameters for session.update()
        # Constructing session_params as a dictionary, similar to the reference.
        session_params = {
            "turn_detection": {"type": "server_vad"},
            "input_audio_transcription": transcription_model_args,
            # "output_audio_generation" and voice selection removed as it causes an error
            # with the default OpenAI Realtime model (e.g., gpt-4o-mini-realtime-preview-2024-12-17).
            # The API will likely use a default voice or one associated with the account/model.
        }
        # --- End session parameters preparation ---
        
        if self.current_openai_voice:
            logger.warning(
                f"OpenAIRealtimeHandler: A voice ('{self.current_openai_voice}') is configured for the OpenAI backend, "
                "but the current API model/version might not support explicit voice selection via client parameters. "
                "The default voice for the model will likely be used."
            )
        
        # Parameters for client.beta.realtime.connect()
        connect_kwargs: Dict[str, Any] = {
            "model": self.settings.openai_realtime_model_arg
        }
        # Voice parameter moved to session_params
        
        try:
            self._reset_turn_usage_state() # Reset usage for new connection/session
            async with self.client.beta.realtime.connect(**connect_kwargs) as conn:
                await conn.session.update(session=session_params) 
                self.connection = conn
                logger.info(f"OpenAIRealtimeHandler: Connection established with model {self.settings.openai_realtime_model_arg}, voice {self.current_openai_voice or 'default via API'}.")
                
                async for event in self.connection:
                    logger.debug(f"OpenAIRealtime Event: {event.type}")
                    if event.type == "input_audio_buffer.speech_started":
                        self.clear_queue() 
                        self._reset_turn_usage_state() # Reset for a new turn of speech
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "type": "status_update",
                                    "status": "stt_processing", 
                                    "message": "Listening...",
                                }
                            )
                        )
                    elif event.type == "conversation.item.input_audio_transcription.completed":
                        user_message = ChatMessage(
                            role="user", 
                            content=event.transcript, 
                            metadata=ChatMessageMetadata(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
                        )
                        await self.output_queue.put(
                            AdditionalOutputs({"type": "chatbot_update", "message": user_message.model_dump()})
                        )
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "type": "status_update",
                                    "status": "llm_waiting", 
                                    "message": "AI Responding...",
                                }
                            )
                        )
                    elif event.type == "response.audio_transcript.done":
                        logger.info(f"OpenAIRealtimeHandler: Assistant transcript received: '{event.transcript[:100]}...'")
                        # The final processing, cost calculation, and message creation will be handled by "response.done".
                        # We store the transcript if needed, or rely on "response.done" to contain it.
                        # For now, this event primarily serves as a log point.

                    elif event.type == "response.done":
                        logger.info(f"OpenAIRealtimeHandler: Response done event received.")
                        assistant_text = ""
                        if hasattr(event, 'response') and event.response and \
                           hasattr(event.response, 'output') and event.response.output and \
                           len(event.response.output) > 0 and \
                           hasattr(event.response.output[0], 'content') and event.response.output[0].content and \
                           len(event.response.output[0].content) > 0 and \
                           hasattr(event.response.output[0].content[0], 'text'):
                            assistant_text = event.response.output[0].content[0].text
                            logger.info(f"OpenAIRealtimeHandler: Assistant text from response.done: '{assistant_text[:100]}...'")
                        else:
                            logger.error("OpenAIRealtimeHandler: Could not extract assistant text from response.done event.")
                            # Potentially fallback to a stored transcript if we decide to implement that,
                            # or yield an error message. For now, proceed with empty text if not found.

                        # Extract usage from event.response.usage
                        if hasattr(event, 'response') and event.response and hasattr(event.response, 'usage'):
                            usage_data = event.response.usage
                            # Ensure tokens are integers, defaulting to 0 if None or attribute missing
                            raw_input_tokens = getattr(usage_data, 'input_tokens', 0)
                            raw_output_tokens = getattr(usage_data, 'output_tokens', 0)
                            
                            self.current_input_tokens = raw_input_tokens if raw_input_tokens is not None else 0
                            self.current_output_tokens = raw_output_tokens if raw_output_tokens is not None else 0
                            
                            logger.info(f"OpenAIRealtimeHandler: Final tokens from response.done: Input={self.current_input_tokens}, Output={self.current_output_tokens}")
                            logger.debug(f"[BACKEND_COST_DEBUG_OPENAI] Raw usage_data from event: {usage_data}")
                            logger.debug(f"[BACKEND_COST_DEBUG_OPENAI] Parsed tokens: Input={self.current_input_tokens}, Output={self.current_output_tokens}")

                            # Append this final usage to raw_usage_events_for_turn for completeness
                            raw_final_usage = dict(usage_data) if hasattr(usage_data, 'dict') else vars(usage_data)
                            self.raw_usage_events_for_turn.append({"type": event.type, "usage": raw_final_usage})
                        else:
                            logger.error("OpenAIRealtimeHandler: Could not extract usage data from response.done event. Token counts will be zero.")
                            self.current_input_tokens = 0
                            self.current_output_tokens = 0
                        
                        # --- Cost Calculation & Metadata (OpenAI Backend - Token Based) ---
                        input_cost = 0.0
                        output_cost = 0.0
                        total_cost = 0.0
                        model_name_for_pricing = self.settings.openai_realtime_model_arg.lower()
                        resolved_model_prices = None
                        
                        try:
                            resolved_model_prices = OPENAI_REALTIME_PRICING[model_name_for_pricing]
                            logger.info(f"OpenAIRealtimeHandler: Found direct pricing for model '{model_name_for_pricing}'.")
                        except KeyError:
                            logger.info(f"OpenAIRealtimeHandler: No direct pricing for '{model_name_for_pricing}'. Trying base model match...")
                            found_base_match = False
                            for base_model_key in OPENAI_REALTIME_PRICING.keys():
                                if model_name_for_pricing.startswith(base_model_key):
                                    resolved_model_prices = OPENAI_REALTIME_PRICING[base_model_key]
                                    logger.info(f"OpenAIRealtimeHandler: Found pricing for '{self.settings.openai_realtime_model_arg}' using base model key '{base_model_key}'.")
                                    found_base_match = True
                                    break
                            if not found_base_match:
                                logger.critical(
                                    f"OpenAIRealtimeHandler: No pricing for model '{model_name_for_pricing}' in OPENAI_REALTIME_PRICING."
                                )
                                # Set costs to 0 and proceed if pricing info is missing, but log critical error.
                                resolved_model_prices = None # Ensure it's None so costs remain 0
                        
                        if resolved_model_prices:
                            price_input_per_mil = resolved_model_prices.get("input", 0.0)
                            price_output_per_mil = resolved_model_prices.get("output", 0.0)

                            input_cost = (self.current_input_tokens / 1_000_000) * price_input_per_mil
                            output_cost = (self.current_output_tokens / 1_000_000) * price_output_per_mil
                            total_cost = input_cost + output_cost
                            logger.info(
                                f"OpenAIRealtime Token Costs: Input Tokens: {self.current_input_tokens}, Output Tokens: {self.current_output_tokens}. "
                                f"Input Cost: ${input_cost:.6f}, Output Cost: ${output_cost:.6f}, Total: ${total_cost:.6f}"
                            )
                            logger.debug(f"[BACKEND_COST_DEBUG_OPENAI] Calculated costs: Input: ${input_cost:.8f}, Output: ${output_cost:.8f}, Total: ${total_cost:.8f}")
                        else:
                            logger.error(f"OpenAIRealtimeHandler: Cost calculation skipped due to missing pricing for model '{model_name_for_pricing}'.")


                        cost_data = {
                            "input_cost": input_cost,
                            "output_cost": output_cost,
                            "total_cost": total_cost,
                            "input_tokens": self.current_input_tokens,
                            "output_tokens": self.current_output_tokens,
                            "model": self.settings.openai_realtime_model_arg,
                            "output_audio_duration_seconds": round(self.current_output_audio_duration_seconds, 2),
                            "note": "Costs are token-based. Audio duration is informational."
                        }
                        logger.debug(f"[BACKEND_COST_DEBUG_OPENAI] cost_data to be sent: {json.dumps(cost_data)}")
                        await self.output_queue.put(AdditionalOutputs({"type": "cost_update", "data": cost_data}))

                        assistant_metadata = ChatMessageMetadata(
                            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(), 
                            llm_model=self.settings.openai_realtime_model_arg,
                            cost=cost_data,
                            output_audio_duration_seconds=round(self.current_output_audio_duration_seconds, 2),
                            usage={"input_tokens": self.current_input_tokens, "output_tokens": self.current_output_tokens},
                            raw_openai_usage_events=self.raw_usage_events_for_turn.copy()
                        )
                        assistant_message = ChatMessage(
                            role="assistant", 
                            content=assistant_text, 
                            metadata=assistant_metadata
                        )
                        await self.output_queue.put(
                            AdditionalOutputs({"type": "chatbot_update", "message": assistant_message.model_dump()})
                        )
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "type": "status_update",
                                    "status": "idle",
                                    "message": "Ready",
                                }
                            )
                        )
                        self._reset_turn_usage_state() # Reset for the next full turn interaction after speech_started

                    elif event.type == "response.audio.delta":
                        audio_data_bytes = base64.b64decode(event.delta)
                        audio_data_np = np.frombuffer(audio_data_bytes, dtype=np.int16)
                        
                        num_samples = len(audio_data_np)
                        duration_seconds_chunk = num_samples / OPENAI_REALTIME_SAMPLE_RATE
                        self.current_output_audio_duration_seconds += duration_seconds_chunk

                        if audio_data_np.ndim == 1: 
                            audio_data_np = audio_data_np.reshape(1, -1)
                        await self.output_queue.put(
                            (
                                OPENAI_REALTIME_SAMPLE_RATE,
                                audio_data_np,
                            ),
                        )
                    elif event.type == "conversation.item.usage.completed" or event.type == "response.usage.completed":
                        # Log the raw usage data for debugging. vars() is a good way to see attributes.
                        logger.info(f"OpenAIRealtime Usage Event: {event.type} - Raw Usage Data: {vars(event.usage) if hasattr(event, 'usage') and event.usage is not None else 'Usage object missing or None'}")

                        # Ensure event.usage exists. If not, direct access below will raise AttributeError.
                        if not hasattr(event, 'usage') or event.usage is None:
                            logger.warning(f"OpenAIRealtime Usage Event ({event.type}) is missing the 'usage' object. Raw logging might be incomplete.")
                            # Do not raise, just log and potentially skip adding to raw_usage_events_for_turn
                            raw_usage_data = {"error": "Usage object missing or None"}
                        else:
                            # Log the raw usage data for debugging. vars() is a good way to see attributes.
                            logger.info(f"OpenAIRealtime Usage Event: {event.type} - Raw Usage Data: {vars(event.usage)}")
                            # Store the raw usage object (or its dict representation)
                            raw_usage_data = dict(event.usage) if hasattr(event.usage, 'dict') else vars(event.usage)
                        
                        self.raw_usage_events_for_turn.append({"type": event.type, "usage": raw_usage_data})
                        logger.debug(f"OpenAIRealtimeHandler: Raw usage event '{event.type}' logged. Token accumulation is handled by 'response.done'.")
                        # Token accumulation (self.current_input_tokens += ...) is removed from here.
                        # Final token counts will be taken from the "response.done" event.

                    elif event.type == "error":
                        error_code = "N/A"
                        error_message_text = "Unknown error from OpenAI Realtime API."
                        if hasattr(event, 'error') and event.error:
                            if hasattr(event.error, 'code'):
                                error_code = event.error.code
                            if hasattr(event.error, 'message'):
                                error_message_text = event.error.message
                        else:
                            # Fallback if event.error structure is not as expected or missing
                            error_message_text = str(event) # Convert the event to string as a last resort

                        full_error_details = f"Code: {error_code}, Message: {error_message_text}"
                        logger.error(f"OpenAI Realtime API Error: {full_error_details}")
                        
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "type": "status_update",
                                    "status": "error",
                                    "message": f"OpenAI Error: {error_message_text}", # UI gets the message part
                                }
                            )
                        )
                        error_chat_message = ChatMessage(role="assistant", content=f"[OpenAI Error: {error_message_text}]")
                        await self.output_queue.put(
                            AdditionalOutputs({"type": "chatbot_update", "message": error_chat_message.model_dump()})
                        )
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "type": "status_update",
                                    "status": "idle",
                                    "message": "Ready (after error)",
                                }
                            )
                        )
                        # Removed problematic final_chatbot_state
        except openai.AuthenticationError as e:
            logger.error(f"OpenAIRealtimeHandler: Authentication Error: {e}. Check your --openai-api-key.")
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": "OpenAI Auth Error. Check API Key."}))
        except Exception as e:
            error_message_for_log = "Unknown error"
            error_message_for_ui = "Unknown error"
            try:
                error_message_for_log = str(e)
                # Try to get a more specific message for UI if available from e.message
                ui_msg_candidate = getattr(e, 'message', None)
                if isinstance(ui_msg_candidate, str) and ui_msg_candidate:
                    error_message_for_ui = ui_msg_candidate
                else:
                    error_message_for_ui = error_message_for_log # Fallback to the general str(e)
            except Exception as conversion_exception:
                logger.warning(
                    f"Could not convert original exception to string or extract message. "
                    f"Original exception type: {type(e).__name__}. "
                    f"Conversion/extraction exception: {type(conversion_exception).__name__} - {str(conversion_exception)}"
                )
                error_message_for_log = f"Unstringifiable error of type {type(e).__name__}"
                error_message_for_ui = f"A server error of type {type(e).__name__} occurred."

            logger.error(f"OpenAIRealtimeHandler: Connection failed or error during session: {error_message_for_log}", exc_info=True)
            await self.output_queue.put(AdditionalOutputs({"type": "status_update", "status": "error", "message": f"Connection Error: {error_message_for_ui}"}))
        finally:
            logger.info("OpenAIRealtimeHandler: start_up processing loop finished.")
            if self.connection:
                logger.info("OpenAIRealtimeHandler: Closing connection in start_up finally block.")
                try:
                    await self.connection.close()
                except Exception as e:
                    logger.warning(f"OpenAIRealtimeHandler: Error closing connection in start_up finally: {e}")
            self.connection = None 
            await self.output_queue.put(None) 


    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection: 
            return
        
        _, array = frame
        if array.ndim > 1: 
            array = array.squeeze()

        if array.dtype != np.int16:
            array = array.astype(np.int16)

        audio_bytes = array.tobytes()
        audio_message = base64.b64encode(audio_bytes).decode("utf-8")
        try:
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e: 
            logger.error(f"OpenAIRealtimeHandler: Error sending audio: {e}")


    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        logger.info("OpenAIRealtimeHandler: Shutting down...")
        if self.connection:
            logger.info("OpenAIRealtimeHandler: Closing connection in shutdown.")
            try:
                await self.connection.close()
            except Exception as e:
                logger.warning(f"OpenAIRealtimeHandler: Error closing connection in shutdown: {e}")
        self.connection = None 
        if self.client:
            await self.client.close() 
            self.client = None
        self.clear_queue()
        await self.output_queue.put(None) 
        logger.info("OpenAIRealtimeHandler: Shutdown complete.")

    def clear_queue(self):
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("OpenAIRealtimeHandler: Output queue cleared.")

# --- Endpoint Definitions ---
def register_endpoints(app: FastAPI, stream: Stream):
    """Registers FastAPI endpoints."""
    curr_dir = Path(__file__).parent

    @app.get("/")
    async def _():
        rtc_config = get_twilio_turn_credentials() if get_space() else None
        index_path = curr_dir / "index.html"
        if not index_path.exists():
            logger.error("index.html not found in the current directory!")
            return HTMLResponse(
                content="<html><body><h1>Error: index.html not found</h1></body></html>",
                status_code=500,
            )

        html_content = index_path.read_text()
        if rtc_config:
            html_content = html_content.replace(
                "__RTC_CONFIGURATION__", json.dumps(rtc_config)
            )
        else:
            html_content = html_content.replace("__RTC_CONFIGURATION__", "null")

        html_content = html_content.replace(
            "__SYSTEM_MESSAGE_JSON__", json.dumps(settings.system_message)
        )
        html_content = html_content.replace("__APP_VERSION__", APP_VERSION)
        html_content = html_content.replace(
            "__STT_LANGUAGE_JSON__", json.dumps(settings.current_stt_language)
        )
        # For TTS speed, OpenAI backend doesn't have a user-configurable speed via this app's settings.
        # It might be part of the voice model or a parameter not exposed here. Default to 1.0.
        tts_speed_to_inject = settings.current_tts_speed if settings.backend == "classic" else 1.0
        html_content = html_content.replace(
            "__TTS_SPEED_JSON__", json.dumps(tts_speed_to_inject)
        )
        html_content = html_content.replace(
            "__STARTUP_TIMESTAMP_STR__", json.dumps(settings.startup_timestamp_str)
        )
        html_content = html_content.replace(
            "__BACKEND_TYPE__", json.dumps(settings.backend)
        )
        return HTMLResponse(content=html_content, status_code=200)

    @app.post("/input_hook")
    async def _(body: InputData):
        chatbot_history = [msg.model_dump() for msg in body.chatbot]
        stream.set_input(body.webrtc_id, chatbot_history)
        return {"status": "ok"}

    @app.get("/outputs")
    def _(webrtc_id: str):
        async def output_stream():
            try:
                async for output in stream.output_stream(webrtc_id):
                    if isinstance(output, AdditionalOutputs):
                        data_payload = output.args[0]
                        if isinstance(data_payload, dict) and "type" in data_payload:
                            event_type = data_payload["type"]
                            try:
                                event_data_json = json.dumps(data_payload, ensure_ascii=False)
                                # if settings.verbose:
                                #     logger.debug(
                                #         f"Sending SSE event: type={event_type}, data={event_data_json}..."
                                #     )
                                # else:
                                logger.debug(
                                    f"Sending SSE event: type={event_type}, data={event_data_json[:100]}...{event_data_json[-100:]}"
                                    )
                                yield f"event: {event_type}\ndata: {event_data_json}\n\n"
                            except TypeError as e:
                                logger.error(
                                    f"Failed to serialize AdditionalOutputs payload to JSON: {e}. Payload: {data_payload}"
                                )
                        else:
                            logger.warning(
                                f"Received AdditionalOutputs with unexpected payload structure: {data_payload}"
                            )
                    elif (
                        isinstance(output, tuple)
                        and len(output) == 2
                        and isinstance(output[1], np.ndarray)
                    ):
                        logger.debug(
                            f"Output stream received audio tuple for webrtc_id {webrtc_id}, should be handled by track."
                        )
                        pass 
                    elif isinstance(output, bytes):
                        logger.warning(
                            "Received raw bytes directly in output stream, expected AdditionalOutputs or audio tuple via handler."
                        )
                    else:
                        logger.warning(
                            f"Received unexpected output type in stream: {type(output)}"
                        )
            except Exception as e:
                logger.error(f"Error in output stream for webrtc_id {webrtc_id}: {e}")
                try:
                    error_payload = {
                        "type": "error_event",
                        "message": f"Server stream error: {str(e)}",
                    }
                    yield f"event: error_event\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
                except Exception as send_err:
                    logger.error(
                        f"Failed to send error event to client {webrtc_id}: {send_err}"
                    )
        return StreamingResponse(output_stream(), media_type="text/event-stream")

    @app.get("/available_models")
    async def get_available_models_endpoint(): 
        if settings.backend == "openai":
            return JSONResponse(
                {"available": [settings.openai_realtime_model_arg], "current": settings.openai_realtime_model_arg}
            )
        elif settings.backend == "gemini": # Add Gemini case
            return JSONResponse(
                # For Gemini, model is also fixed at startup like OpenAI.
                # Use settings.current_llm_model which is set to the gemini model during startup.
                {"available": [settings.current_llm_model], "current": settings.current_llm_model}
            )
        else: # classic
            return JSONResponse(
                {"available": settings.available_models, "current": settings.current_llm_model}
            )

    @app.get("/available_voices_tts")
    async def get_available_voices_tts_endpoint(): 
        if settings.backend == "openai":
            # OpenAI realtime backend uses its own set of voices, potentially configurable
            response_data = prepare_available_voices_data(
                settings.current_openai_voice, OPENAI_REALTIME_VOICES # Use OPENAI_REALTIME_VOICES
            )
            response_data["is_openai_realtime"] = True # Add a flag for UI
            return JSONResponse(response_data)
        elif settings.backend == "gemini": # Add Gemini case
            response_data = prepare_available_voices_data(
                settings.current_gemini_voice, GEMINI_LIVE_VOICES
            )
            response_data["is_gemini_realtime"] = True # Add a specific flag for UI
            return JSONResponse(response_data)
        else: # classic
            response_data = prepare_available_voices_data(
                settings.current_tts_voice, settings.available_voices_tts
            )
            response_data["is_openai_realtime"] = False # existing flag for classic
            return JSONResponse(response_data)

    @app.post("/switch_voice")
    async def switch_voice(request: Request):
        try:
            data = await request.json()
            new_voice_name = data.get("voice_name")
            if not new_voice_name:
                logger.warning("Missing voice_name in switch request.")
                return JSONResponse(
                    {"status": "error", "message": "Missing 'voice_name' in request body."},
                    status_code=400,
                )

            if settings.backend == "openai":
                if new_voice_name != settings.current_openai_voice:
                    if new_voice_name in OPENAI_REALTIME_VOICES:
                        settings.current_openai_voice = new_voice_name
                        logger.info(f"Switched active OpenAI realtime voice to: {settings.current_openai_voice}")
                        # The OpenAIRealtimeHandler will pick this up on the next connection.
                        return JSONResponse(
                            {"status": "success", "voice": settings.current_openai_voice}
                        )
                    else:
                        logger.warning(
                            f"Attempted to switch OpenAI realtime voice to '{new_voice_name}' which is not in the available list: {OPENAI_REALTIME_VOICES}"
                        )
                        return JSONResponse(
                            {"status": "error", "message": f"Voice '{new_voice_name}' is not available for OpenAI realtime backend."},
                            status_code=400,
                        )
                else:
                    logger.info(f"OpenAI realtime voice already set to: {new_voice_name}")
                    return JSONResponse(
                        {"status": "success", "voice": settings.current_openai_voice}
                    )
            elif settings.backend == "gemini": # Add Gemini logic
                if new_voice_name != settings.current_gemini_voice:
                    if new_voice_name in GEMINI_LIVE_VOICES:
                        settings.current_gemini_voice = new_voice_name
                        logger.info(f"Switched active Gemini realtime voice to: {settings.current_gemini_voice}")
                        # The GeminiRealtimeHandler will pick this up on the next connection.
                        return JSONResponse(
                            {"status": "success", "voice": settings.current_gemini_voice}
                        )
                    else:
                        logger.warning(
                            f"Attempted to switch Gemini realtime voice to '{new_voice_name}' which is not in the available list: {GEMINI_LIVE_VOICES}"
                        )
                        return JSONResponse(
                            {"status": "error", "message": f"Voice '{new_voice_name}' is not available for Gemini realtime backend."},
                            status_code=400,
                        )
                else:
                    logger.info(f"Gemini realtime voice already set to: {new_voice_name}")
                    return JSONResponse(
                        {"status": "success", "voice": settings.current_gemini_voice}
                    )
            else: # Classic backend logic
                if new_voice_name != settings.current_tts_voice:
                    if new_voice_name in settings.available_voices_tts:
                        settings.current_tts_voice = new_voice_name
                        logger.info(f"Switched active classic TTS voice to: {settings.current_tts_voice}")
                        return JSONResponse(
                            {"status": "success", "voice": settings.current_tts_voice}
                        )
                    else:
                        logger.warning(
                            f"Attempted to switch classic TTS voice to '{new_voice_name}' which is not in the available list: {settings.available_voices_tts}"
                        )
                        return JSONResponse(
                            {"status": "error", "message": f"Voice '{new_voice_name}' is not available for classic backend."},
                            status_code=400,
                        )
                else:
                    logger.info(f"Classic TTS voice already set to: {new_voice_name}")
                    return JSONResponse(
                        {"status": "success", "voice": settings.current_tts_voice}
                    ) 
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON body in /switch_voice")
            return JSONResponse(
                {"status": "error", "message": "Invalid JSON format in request body"},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Error processing /switch_voice request: {e}")
            return JSONResponse(
                {"status": "error", "message": f"Internal server error: {str(e)}"},
                status_code=500,
            )

    @app.post("/switch_stt_language")
    async def switch_stt_language(request: Request):
        try:
            data = await request.json()
            new_language = data.get("stt_language", None) 

            if new_language is not None and not new_language.strip():
                new_language = None
            elif new_language is not None:
                new_language = new_language.strip()

            if new_language != settings.current_stt_language:
                settings.current_stt_language = new_language
                logger.info(
                    f"Switched active STT language to: '{settings.current_stt_language}' (None means auto-detect)"
                )
                # For OpenAI backend, the handler needs to pick up this change on next connection.
                return JSONResponse(
                    {"status": "success", "stt_language": settings.current_stt_language}
                )
            else:
                logger.info(
                    f"STT language already set to: '{settings.current_stt_language}'"
                )
                return JSONResponse(
                    {"status": "success", "stt_language": settings.current_stt_language}
                )
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON body in /switch_stt_language")
            return JSONResponse(
                {"status": "error", "message": "Invalid JSON format in request body"},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Error processing /switch_stt_language request: {e}")
            return JSONResponse(
                {"status": "error", "message": f"Internal server error: {str(e)}"},
                status_code=500,
            )

    @app.post("/switch_tts_speed")
    async def switch_tts_speed(request: Request):
        if settings.backend == "openai" or settings.backend == "gemini": # Add Gemini here
            logger.info(f"TTS speed adjustment requested for {settings.backend} backend, which is not currently user-configurable via this app's UI.")
            return JSONResponse(
                {"status": "info", "message": f"TTS speed adjustment for {settings.backend} realtime backend is handled by the provider or not user-configurable through this app."},
                status_code=200,
            )
        # Classic backend logic
        try:
            data = await request.json()
            new_speed = data.get("tts_speed") 

            if new_speed is None:
                logger.warning("Missing 'tts_speed' in switch request.")
                return JSONResponse(
                    {"status": "error", "message": "Missing 'tts_speed' in request body."},
                    status_code=400,
                )
            try:
                new_speed_float = float(new_speed)
                if not (0.1 <= new_speed_float <= 4.0):
                    raise ValueError("TTS speed must be between 0.1 and 4.0")
            except (ValueError, TypeError):
                logger.warning(f"Invalid TTS speed value received: {new_speed}")
                return JSONResponse(
                    {"status": "error", "message": "Invalid TTS speed value. Must be a number between 0.1 and 4.0."},
                    status_code=400,
                )

            if new_speed_float != settings.current_tts_speed:
                settings.current_tts_speed = new_speed_float
                logger.info(
                    f"Switched active TTS speed (classic backend) to: {settings.current_tts_speed:.1f}"
                )
                return JSONResponse(
                    {"status": "success", "tts_speed": settings.current_tts_speed}
                )
            else:
                logger.info(
                    f"TTS speed (classic backend) already set to: {settings.current_tts_speed:.1f}"
                )
                return JSONResponse(
                    {"status": "success", "tts_speed": settings.current_tts_speed}
                )
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON body in /switch_tts_speed")
            return JSONResponse(
                {"status": "error", "message": "Invalid JSON format in request body"},
                status_code=400,
            )
        except Exception as e:
            logger.error(f"Error processing /switch_tts_speed request: {e}")
            return JSONResponse(
                {"status": "error", "message": f"Internal server error: {str(e)}"},
                status_code=500,
            )

    @app.post("/switch_model")
    async def switch_model(request: Request):
        if settings.backend == "openai" or settings.backend == "gemini": # Add Gemini here
            logger.warning(f"Attempt to switch model for {settings.backend} backend via API, which is not supported post-startup.")
            return JSONResponse(
                {"status": "error", "message": f"Switching {settings.backend} realtime model via API is not supported. Model is fixed at startup."},
                status_code=400,
            )
        else: # classic backend
            try:
                data = await request.json()
                new_model_name = data.get("model_name")
                if new_model_name:
                    if new_model_name != settings.current_llm_model:
                        if new_model_name in settings.available_models:
                            settings.current_llm_model = new_model_name
                            logger.info(
                                f"Switched active LLM model (classic backend) to: {settings.current_llm_model}"
                            )
                            if (
                                new_model_name not in settings.model_cost_data
                                or settings.model_cost_data[new_model_name].get(
                                    "input_cost_per_token"
                                )
                                is None
                            ):
                                logger.warning(
                                    f"Cost data might be missing or incomplete for the newly selected model '{settings.current_llm_model}'."
                                )
                            return JSONResponse(
                                {"status": "success", "model": settings.current_llm_model}
                            )
                        else:
                            logger.warning(
                                f"Attempted to switch to model '{new_model_name}' which is not in the available list: {settings.available_models}"
                            )
                            return JSONResponse(
                                {"status": "error", "message": f"Model '{new_model_name}' is not available."},
                                status_code=400,
                            )
                    else:
                        logger.info(f"Model already set to: {new_model_name}")
                        return JSONResponse(
                            {"status": "success", "model": settings.current_llm_model}
                        )
                else: 
                    logger.warning(f"Missing model_name in switch request.")
                    return JSONResponse(
                        {"status": "error", "message": f"Missing 'model_name' in request body."},
                        status_code=400,
                    )
            except json.JSONDecodeError:
                logger.error("Failed to decode JSON body in /switch_model")
                return JSONResponse(
                    {"status": "error", "message": "Invalid JSON format in request body"},
                    status_code=400,
                )
            except Exception as e:
                logger.error(f"Error processing /switch_model request: {e}")
                return JSONResponse(
                    {"status": "error", "message": f"Internal server error: {str(e)}"},
                    status_code=500,
                )

    @app.post("/heartbeat")
    async def heartbeat(request: Request):
        global last_heartbeat_time
        try:
            last_heartbeat_time = datetime.datetime.now(datetime.timezone.utc)
            payload = await request.json()
            logger.debug(
                f"Heartbeat received at {last_heartbeat_time}. Payload: {payload}"
            )
            return {"status": "ok"}
        except json.JSONDecodeError:
            last_heartbeat_time = datetime.datetime.now(datetime.timezone.utc)
            logger.debug(
                f"Heartbeat received at {last_heartbeat_time} (no valid JSON payload)."
            )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @app.post("/reset_chat_log")
    async def reset_chat_log():
        try:
            new_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            settings.startup_timestamp_str = new_timestamp
            logger.info(f"Chat log timestamp reset to: {settings.startup_timestamp_str}")
            # Also reset the TTS audio directory for the new "session"
            if settings.tts_base_dir: # Only relevant for classic backend
                settings.tts_audio_dir = settings.tts_base_dir / settings.startup_timestamp_str
                settings.tts_audio_dir.mkdir(exist_ok=True)
                logger.info(f"TTS audio directory for this session reset to: {settings.tts_audio_dir}")
            else:
                logger.warning("tts_base_dir not set, cannot reset TTS audio directory (or not applicable for current backend).")

            return JSONResponse(
                {"status": "success", "new_timestamp": settings.startup_timestamp_str}
            )
        except Exception as e:
            logger.error(f"Error resetting chat log timestamp: {e}")
            return JSONResponse(
                {"status": "error", "message": "Failed to reset chat log timestamp"},
                status_code=500,
            )

    @app.get("/tts_audio/{run_timestamp}/{filename}")
    async def get_tts_audio(run_timestamp: str, filename: str):
        if settings.backend == "openai" or settings.backend == "gemini": # Add Gemini Here
            logger.warning(f"Request to /tts_audio for '{filename}' when using {settings.backend} backend. This endpoint is for classic backend.")
            raise HTTPException(status_code=404, detail=f"TTS audio file serving not applicable for {settings.backend} backend.")

        if not settings.tts_audio_dir: 
             logger.error("settings.tts_audio_dir not configured for classic backend, cannot serve audio.")
             raise HTTPException(status_code=500, detail="Server configuration error")

        if ".." in filename or "/" in filename or "\\" in filename:
            logger.warning(f"Attempt to access potentially unsafe filename: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        file_path = settings.tts_audio_dir / filename
        logger.debug(f"Attempting to serve TTS audio file (classic backend): {file_path}")

        if file_path.is_file():
            return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
        else:
            logger.warning(f"TTS audio file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Audio file not found")

# --- Pywebview API Class ---
class Api:
    def __init__(self, window):
        self._window = window

    def close(self):
        logger.info("API close method called.")
        if self._window:
            self._window.destroy()

# --- Heartbeat Monitoring Thread ---
def monitor_heartbeat_thread():
    global last_heartbeat_time, uvicorn_server, pywebview_window, shutdown_event
    logger.info("Heartbeat monitor thread started.")
    initial_wait_done = False

    while not shutdown_event.is_set():
        if last_heartbeat_time is None:
            if not initial_wait_done:
                logger.info(
                    f"Waiting for the first heartbeat (timeout check in {heartbeat_timeout * 2}s)..."
                )
                shutdown_event.wait(heartbeat_timeout * 2)
                initial_wait_done = True
                if shutdown_event.is_set():
                    break 
                continue 
            else:
                logger.debug("Still waiting for first heartbeat...")
                shutdown_event.wait(5) 
                if shutdown_event.is_set():
                    break
                continue

        time_since_last = (
            datetime.datetime.now(datetime.timezone.utc) - last_heartbeat_time
        )
        logger.debug(
            f"Time since last heartbeat: {time_since_last.total_seconds():.1f}s"
        )

        if time_since_last.total_seconds() > heartbeat_timeout:
            if not settings.disable_heartbeat:
                logger.warning(
                    f"Heartbeat timeout ({heartbeat_timeout}s exceeded). Initiating shutdown."
                )
                if uvicorn_server:
                    logger.info("Signaling Uvicorn server to stop...")
                    uvicorn_server.should_exit = True
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown."
                    )

                if pywebview_window:
                    logger.info("Destroying pywebview window...")
                    try:
                        pywebview_window.destroy()
                    except Exception as e:
                        logger.error(
                            f"Error destroying pywebview window from monitor thread: {e}"
                        )
                break # Break loop to terminate monitor thread after shutdown initiated
            else:
                logger.info(
                    f"Heartbeat timeout ({heartbeat_timeout}s exceeded), but heartbeat monitoring is disabled. Not shutting down."
                )
                # Reset last_heartbeat_time to prevent constant logging of this message if client truly disconnected badly
                # This means we'd only log this once, then wait for a new "first" heartbeat.
                last_heartbeat_time = None
                initial_wait_done = False # Re-trigger initial wait logic
                logger.info("Resetting heartbeat state to wait for a new initial heartbeat.")


        shutdown_event.wait(5)
    logger.info("Heartbeat monitor thread finished.")


@click.command(help="Run a simple voice chat interface using a configurable LLM provider, STT server, and TTS.")
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind the FastAPI server to.",
)
@click.option(
    "--port",
    type=click.INT,
    envvar="APP_PORT",
    default=int(APP_PORT_ENV), 
    show_default=True, 
    help="Preferred port to run the FastAPI server on. (Env: APP_PORT)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    show_default=True,
    help="Enable verbose logging (DEBUG level).",
)
@click.option(
    "--browser",
    is_flag=True,
    default=False,
    show_default=True,
    help="Launch the application in the default web browser instead of a dedicated GUI window.",
)
@click.option(
    "--system-message",
    type=str,
    envvar="SYSTEM_MESSAGE",
    default=SYSTEM_MESSAGE_ENV, 
    show_default=True,
    help="System message to prepend to the chat history. (Env: SYSTEM_MESSAGE)",
)
@click.option(
    "--disable-heartbeat",
    is_flag=True,
    envvar="DISABLE_HEARTBEAT",
    default=(DISABLE_HEARTBEAT_ENV.lower() == "true"), # Convert string "False" to bool False
    show_default=True,
    help="Disable heartbeat timeout check (application will not exit if browser tab is closed without proper shutdown). (Env: DISABLE_HEARTBEAT)",
)
@click.option(
    "--backend",
    type=click.Choice(["classic", "openai", "gemini"], case_sensitive=False), # Add "gemini"
    default="classic",
    show_default=True,
    help="Backend to use for voice processing. 'classic' uses separate STT/LLM/TTS. 'openai' uses OpenAI's realtime voice API. 'gemini' uses Google's Gemini Live Connect API (Alpha).",
)
@click.option(
    "--openai-realtime-model",
    type=str,
    envvar="OPENAI_REALTIME_MODEL", 
    default=OPENAI_REALTIME_MODEL_ENV, 
    show_default=True,
    help="OpenAI realtime API model to use (if --backend=openai). (Env: OPENAI_REALTIME_MODEL)",
)
@click.option(
    "--openai-realtime-voice",
    type=str,
    envvar="OPENAI_REALTIME_VOICE",
    default=OPENAI_REALTIME_VOICE_ENV,
    show_default=True,
    help="Default voice for OpenAI realtime backend (if --backend=openai). (Env: OPENAI_REALTIME_VOICE)",
)
@click.option(
    "--openai-api-key",
    type=str,
    envvar="OPENAI_API_KEY",
    default=OPENAI_API_KEY_ENV,
    show_default=True,
    help="API key for OpenAI services (REQUIRED if --backend=openai). (Env: OPENAI_API_KEY)",
)
# Add Gemini CLI options
@click.option(
    "--gemini-model",
    type=str,
    envvar="GEMINI_MODEL",
    default=GEMINI_MODEL_ENV,
    show_default=True,
    help="Gemini model to use (if --backend=gemini). (Env: GEMINI_MODEL)",
)
@click.option(
    "--gemini-voice",
    type=str,
    envvar="GEMINI_VOICE",
    default=GEMINI_VOICE_ENV,
    show_default=True,
    help="Default voice for Gemini backend (if --backend=gemini). (Env: GEMINI_VOICE)",
)
@click.option(
    "--gemini-api-key",
    type=str,
    envvar="GEMINI_API_KEY",
    default=GEMINI_API_KEY_ENV,
    show_default=True,
    help="API key for Google Gemini services (REQUIRED if --backend=gemini). (Env: GEMINI_API_KEY)",
)
@click.option(
    "--gemini-compression-threshold",
    type=click.INT,
    envvar="GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD",
    default=int(GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD_ENV),
    show_default=True,
    help="Context window compression threshold for Gemini backend (if --backend=gemini). (Env: GEMINI_CONTEXT_WINDOW_COMPRESSION_THRESHOLD)",
)
@click.option(
    "--llm-host",
    type=str,
    envvar="LLM_HOST",
    default=LLM_HOST_ENV, 
    show_default=True,
    help="Host address of the LLM proxy server (classic backend, optional). (Env: LLM_HOST)",
)
@click.option(
    "--llm-port",
    type=str, 
    envvar="LLM_PORT",
    default=LLM_PORT_ENV, 
    show_default=True,
    help="Port of the LLM proxy server (classic backend, optional). (Env: LLM_PORT)",
)
@click.option(
    "--llm-model",
    type=str,
    envvar="LLM_MODEL",
    default=DEFAULT_LLM_MODEL_ENV,
    show_default=True,
    help="Default LLM model to use for classic backend (e.g., 'gpt-4o', 'litellm_proxy/claude-3-opus'). (Env: LLM_MODEL)",
)
@click.option(
    "--llm-api-key",
    type=str,
    envvar="LLM_API_KEY",
    default=LLM_API_KEY_ENV, 
    show_default=True,
    help="API key for the LLM provider/proxy (classic backend, optional). (Env: LLM_API_KEY)",
)
@click.option(
    "--stt-host",
    type=str,
    envvar="STT_HOST",
    default=STT_HOST_ENV,
    show_default=True,
    help="Host address of the STT server (classic backend). (Env: STT_HOST)",
)
@click.option(
    "--stt-port",
    type=str, 
    envvar="STT_PORT",
    default=STT_PORT_ENV,
    show_default=True,
    help="Port of the STT server (classic backend). (Env: STT_PORT)",
)
@click.option(
    "--stt-model",
    type=str,
    envvar="STT_MODEL",
    default=STT_MODEL_ENV,
    show_default=True,
    help="STT model to use (classic backend). (Env: STT_MODEL)",
)
@click.option(
    "--stt-language",
    type=str,
    envvar="STT_LANGUAGE",
    default=STT_LANGUAGE_ENV, 
    show_default=True,
    help="Language code for STT (e.g., 'en', 'fr'). Used by both backends. If unset, Whisper usually auto-detects. (Env: STT_LANGUAGE)",
)
@click.option(
    "--stt-api-key",
    type=str,
    envvar="STT_API_KEY",
    default=STT_API_KEY_ENV, 
    show_default=True,
    help="API key for the STT server (classic backend, e.g., for OpenAI STT). (Env: STT_API_KEY)",
)
@click.option(
    "--stt-no-speech-prob-threshold",
    type=click.FLOAT,
    envvar="STT_NO_SPEECH_PROB_THRESHOLD",
    default=float(STT_NO_SPEECH_PROB_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence (classic backend): Reject if no_speech_prob > this. (Env: STT_NO_SPEECH_PROB_THRESHOLD)",
)
@click.option(
    "--stt-avg-logprob-threshold",
    type=click.FLOAT,
    envvar="STT_AVG_LOGPROB_THRESHOLD",
    default=float(STT_AVG_LOGPROB_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence (classic backend): Reject if avg_logprob < this. (Env: STT_AVG_LOGPROB_THRESHOLD)",
)
@click.option(
    "--stt-min-words-threshold",
    type=click.INT,
    envvar="STT_MIN_WORDS_THRESHOLD",
    default=int(STT_MIN_WORDS_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence (classic backend): Reject if word count < this. (Env: STT_MIN_WORDS_THRESHOLD)",
)
@click.option(
    "--tts-host",
    type=str,
    envvar="TTS_HOST",
    default=TTS_HOST_ENV,
    show_default=True,
    help="Host address of the TTS server (classic backend). (Env: TTS_HOST)",
)
@click.option(
    "--tts-port",
    type=str, 
    envvar="TTS_PORT",
    default=TTS_PORT_ENV,
    show_default=True,
    help="Port of the TTS server (classic backend). (Env: TTS_PORT)",
)
@click.option(
    "--tts-model",
    type=str,
    envvar="TTS_MODEL",
    default=TTS_MODEL_ENV,
    show_default=True,
    help="TTS model to use (classic backend). (Env: TTS_MODEL)",
)
@click.option(
    "--tts-voice",
    type=str,
    envvar="TTS_VOICE", # This is for CLASSIC backend TTS voice
    default=DEFAULT_VOICE_TTS_ENV,
    show_default=True,
    help="Default TTS voice to use (classic backend). (Env: TTS_VOICE)",
)
@click.option(
    "--tts-api-key",
    type=str,
    envvar="TTS_API_KEY",
    default=TTS_API_KEY_ENV, 
    show_default=True,
    help="API key for the TTS server (classic backend, e.g., for OpenAI TTS). (Env: TTS_API_KEY)",
)
@click.option(
    "--tts-speed",
    type=click.FLOAT,
    envvar="TTS_SPEED",
    default=float(DEFAULT_TTS_SPEED_ENV),
    show_default=True,
    help=f"Default TTS speed multiplier (classic backend). (Env: TTS_SPEED)",
)
@click.option(
    "--tts-acronym-preserve-list",
    type=str,
    envvar="TTS_ACRONYM_PRESERVE_LIST",
    default=TTS_ACRONYM_PRESERVE_LIST_ENV, 
    show_default=True,
    help=f"Comma-separated list of acronyms to preserve during TTS (classic backend, Kokoro TTS). (Env: TTS_ACRONYM_PRESERVE_LIST)",
)
def main(
    host: str,
    port: int,
    verbose: bool,
    browser: bool,
    system_message: Optional[str],
    disable_heartbeat: bool,
    backend: str,
    openai_realtime_model: str,
    openai_realtime_voice: str, # New CLI option for OpenAI backend voice
    openai_api_key: Optional[str],
    gemini_model: str,             # Add Gemini arg
    gemini_voice: str,             # Add Gemini arg
    gemini_api_key: Optional[str], # Add Gemini arg
    gemini_compression_threshold: int, # Add Gemini compression threshold arg
    llm_host: Optional[str],
    llm_port: Optional[str],
    llm_model: str,
    llm_api_key: Optional[str], 
    stt_host: str,
    stt_port: str,
    stt_model: str,
    stt_language: Optional[str], 
    stt_api_key: Optional[str], 
    stt_no_speech_prob_threshold: float,
    stt_avg_logprob_threshold: float,
    stt_min_words_threshold: int,
    tts_host: str,
    tts_port: str,
    tts_model: str,
    tts_voice: str, # This is for CLASSIC backend TTS voice
    tts_api_key: Optional[str], 
    tts_speed: float,
    tts_acronym_preserve_list: str,
) -> int:
    global uvicorn_server, pywebview_window

    startup_time = datetime.datetime.now()
    startup_timestamp_str_local = startup_time.strftime("%Y%m%d_%H%M%S")

    settings.startup_timestamp_str = startup_timestamp_str_local
    settings.backend = backend
    settings.openai_realtime_model_arg = openai_realtime_model
    settings.openai_realtime_voice_arg = openai_realtime_voice # Store initial arg
    settings.openai_api_key = openai_api_key 
    settings.gemini_model_arg = gemini_model       # Store Gemini arg
    settings.gemini_voice_arg = gemini_voice       # Store Gemini arg
    settings.gemini_api_key = gemini_api_key       # Store Gemini arg
    settings.gemini_context_window_compression_threshold = gemini_compression_threshold # Store Gemini threshold

    settings.preferred_port = port
    settings.host = host
    settings.verbose = verbose
    settings.browser = browser
    settings.system_message = system_message.strip() if system_message is not None else ""
    settings.disable_heartbeat = disable_heartbeat

    # --- Logging Setup (Early) ---
    console_log_level_str = "DEBUG" if settings.verbose else "INFO"
    log_file_path_for_setup: Optional[Path] = None
    log_dir_creation_error_details: Optional[str] = None 
    try:
        app_name = "SimpleVoiceChat"
        app_author = "Attila"
        log_base_dir_path_str: Optional[str] = None
        try:
            log_base_dir_path_str = platformdirs.user_log_dir(app_name, app_author)
        except Exception as e_log_dir:
            print(f"Warning: Could not find user log directory ({e_log_dir}), falling back to user data directory for logs.", file=sys.stderr)
            try:
                log_base_dir_path_str = platformdirs.user_data_dir(app_name, app_author)
            except Exception as e_data_dir:
                log_dir_creation_error_details = f"Could not find user data directory either ({e_data_dir})."
                print(f"Error: {log_dir_creation_error_details} File logging will be disabled.", file=sys.stderr)
        
        if log_base_dir_path_str:
            log_base_dir = Path(log_base_dir_path_str)
            settings.app_log_dir = log_base_dir / "logs"
            settings.app_log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path_for_setup = settings.app_log_dir / f"log_{settings.startup_timestamp_str}.log"
        elif not log_dir_creation_error_details: 
            log_dir_creation_error_details = "Failed to determine a valid base directory for logs."
            print(f"Error: {log_dir_creation_error_details} File logging will be disabled.", file=sys.stderr)
            
    except Exception as e:
        log_dir_creation_error_details = f"Failed to set up log directory structure: {e}."
        print(f"Error: {log_dir_creation_error_details} File logging will be disabled.", file=sys.stderr)

    setup_logging(console_log_level_str, log_file_path_for_setup, settings.verbose)
    if log_dir_creation_error_details:
        logger.error(f"Log directory setup failed: {log_dir_creation_error_details} File logging is disabled.")
    
    logger.info(f"Application Version: {APP_VERSION}")
    logger.info(f"Using backend: {settings.backend}")
    if settings.backend == "gemini":
        logger.warning("The Gemini Live Connect API backend is experimental (uses v1alpha).")

    # --- STT Language (Common to all backends that support it) ---
    settings.stt_language_arg = stt_language 
    if settings.stt_language_arg:
        logger.info(f"STT language specified: {settings.stt_language_arg}")
        settings.current_stt_language = settings.stt_language_arg
    else:
        logger.info("No STT language specified (or empty), STT will auto-detect.")
        settings.current_stt_language = None

    # --- Backend Specific Configuration ---
    if settings.backend == "openai":
        logger.info(f"Configuring for 'openai' backend.")
        
        settings.available_models = OPENAI_REALTIME_MODELS # Use the list from config

        if not settings.available_models:
            logger.critical(
                "OPENAI_REALTIME_MODELS list is empty in configuration. "
                "Cannot proceed with OpenAI backend. Exiting."
            )
            return 1

        # Validate the chosen model (from CLI/env, stored in settings.openai_realtime_model_arg)
        # against the available list.
        chosen_model = settings.openai_realtime_model_arg
        if chosen_model in settings.available_models:
            settings.current_llm_model = chosen_model
        else:
            logger.warning(
                f"OpenAI realtime model '{chosen_model}' (from --openai-realtime-model or env) "
                f"is not in the list of supported models: {settings.available_models}. "
                f"Defaulting to the first available model: '{settings.available_models[0]}'."
            )
            settings.current_llm_model = settings.available_models[0]
        
        logger.info(f"OpenAI Realtime Model set to: {settings.current_llm_model}")
        settings.model_cost_data = {} # Cost for OpenAI realtime is handled differently (per token via OPENAI_REALTIME_PRICING)

        if not settings.openai_api_key:
            logger.critical("OpenAI API Key (--openai-api-key or OPENAI_API_KEY env) is REQUIRED for 'openai' backend. Exiting.")
            return 1
        logger.info("Using dedicated OpenAI API Key for 'openai' backend.")

        # OpenAI Realtime Voice
        settings.available_voices_tts = OPENAI_REALTIME_VOICES # Populate for UI dropdown
        initial_openai_voice_preference = settings.openai_realtime_voice_arg
        if initial_openai_voice_preference and initial_openai_voice_preference in OPENAI_REALTIME_VOICES:
            settings.current_openai_voice = initial_openai_voice_preference
        elif OPENAI_REALTIME_VOICES:
            if initial_openai_voice_preference:
                 logger.warning(f"OpenAI realtime voice '{initial_openai_voice_preference}' not found. Using first available: {OPENAI_REALTIME_VOICES[0]}.")
            settings.current_openai_voice = OPENAI_REALTIME_VOICES[0]
        else: # Should not happen if OPENAI_REALTIME_VOICES is populated
            settings.current_openai_voice = initial_openai_voice_preference 
            logger.error(f"No OpenAI realtime voices available or specified voice '{settings.current_openai_voice}' is invalid. Voice may not work.")
        logger.info(f"Initial OpenAI realtime voice set to: {settings.current_openai_voice}")


        # Log warnings if classic backend STT/TTS params are provided unnecessarily
        if stt_host != STT_HOST_ENV or stt_port != STT_PORT_ENV or stt_model != STT_MODEL_ENV or stt_api_key is not None:
            logger.warning("STT host/port/model/api-key parameters are ignored when using 'openai' backend (except STT language).")
        if tts_host != TTS_HOST_ENV or tts_port != TTS_PORT_ENV or tts_model != TTS_MODEL_ENV or tts_voice != DEFAULT_VOICE_TTS_ENV or tts_api_key is not None or tts_speed != float(DEFAULT_TTS_SPEED_ENV):
            logger.warning("Classic TTS host/port/model/voice/api-key/speed parameters are ignored when using 'openai' backend.")
        
        settings.current_tts_speed = 1.0 # Not user-configurable for OpenAI backend via this app

    elif settings.backend == "gemini": # Add Gemini setup block
        logger.info("Configuring for 'gemini' backend.")
        settings.available_models = GEMINI_LIVE_MODELS # Use the list from config

        if not settings.available_models:
            logger.critical("GEMINI_LIVE_MODELS list is empty. Cannot proceed with Gemini. Exiting.")
            return 1
        
        chosen_model = settings.gemini_model_arg
        if chosen_model in settings.available_models:
            settings.current_llm_model = chosen_model # Store the chosen Gemini model here
        else:
            logger.warning(
                f"Gemini model '{chosen_model}' (from --gemini-model or env) "
                f"is not in the list of supported models: {settings.available_models}. "
                f"Defaulting to the first available model: '{settings.available_models[0]}'."
            )
            settings.current_llm_model = settings.available_models[0]
        logger.info(f"Gemini Live Model set to: {settings.current_llm_model}")
        settings.model_cost_data = {} # Cost for Gemini live is character-based, handled by handler.

        if not settings.gemini_api_key:
            logger.critical("Gemini API Key (--gemini-api-key or GEMINI_API_KEY env) is REQUIRED for 'gemini' backend. Exiting.")
            return 1
        logger.info("Using dedicated Gemini API Key for 'gemini' backend.")

        settings.available_voices_tts = GEMINI_LIVE_VOICES # Populate for UI dropdown
        initial_gemini_voice_preference = settings.gemini_voice_arg
        if initial_gemini_voice_preference and initial_gemini_voice_preference in GEMINI_LIVE_VOICES:
            settings.current_gemini_voice = initial_gemini_voice_preference
        elif GEMINI_LIVE_VOICES: # Check if list is not empty
            if initial_gemini_voice_preference: # Log warning only if a preference was given but not found
                 logger.warning(f"Gemini voice '{initial_gemini_voice_preference}' not found. Using first available: {GEMINI_LIVE_VOICES[0]}.")
            settings.current_gemini_voice = GEMINI_LIVE_VOICES[0]
        else: # Should not happen if GEMINI_LIVE_VOICES is populated
            settings.current_gemini_voice = initial_gemini_voice_preference # Fallback to user preference
            logger.error(f"No Gemini voices available or specified voice '{settings.current_gemini_voice}' is invalid. Voice may not work.")
        logger.info(f"Initial Gemini voice set to: {settings.current_gemini_voice}")

        # Log warnings if classic backend STT/TTS params are provided unnecessarily
        # (STT language is common, so not warned here if stt_language is provided)
        if stt_host != STT_HOST_ENV or stt_port != STT_PORT_ENV or stt_model != STT_MODEL_ENV or stt_api_key is not None:
            logger.warning("Classic STT host/port/model/api-key parameters are ignored when using 'gemini' backend (except STT language).")
        if tts_host != TTS_HOST_ENV or tts_port != TTS_PORT_ENV or tts_model != TTS_MODEL_ENV or tts_voice != DEFAULT_VOICE_TTS_ENV or tts_api_key is not None or tts_speed != float(DEFAULT_TTS_SPEED_ENV):
            logger.warning("Classic TTS host/port/model/voice/api-key/speed parameters are ignored when using 'gemini' backend.")
        
        settings.current_tts_speed = 1.0 # Not user-configurable for Gemini backend via this app


    elif settings.backend == "classic":
        logger.info(f"Configuring for 'classic' backend.")
        # --- LLM Configuration (Classic) ---
        settings.llm_host_arg = llm_host
        settings.llm_port_arg = llm_port
        settings.llm_model_arg = llm_model 
        settings.llm_api_key = llm_api_key 

        settings.use_llm_proxy = bool(settings.llm_host_arg and settings.llm_port_arg)
        if settings.use_llm_proxy:
            try:
                llm_port_int = int(settings.llm_port_arg) # type: ignore
                settings.llm_api_base = f"http://{settings.llm_host_arg}:{llm_port_int}/v1"
                logger.info(f"Using LLM proxy at: {settings.llm_api_base}")
                if settings.llm_api_key:
                    logger.info("Using LLM API key for proxy.")
                else:
                    logger.info("No LLM API key provided for proxy (assumed optional).")
            except (ValueError, TypeError):
                logger.error(
                    f"Error: Invalid LLM port specified: '{settings.llm_port_arg}'. Disabling proxy."
                )
                settings.use_llm_proxy = False
                settings.llm_api_base = None
        else:
            settings.llm_api_base = None
            logger.info("Not using LLM proxy (using default LLM routing).")
            if settings.llm_api_key:
                logger.info("Using LLM API key for direct routing.")
            else:
                logger.info("No LLM API key provided for direct routing (will use LiteLLM's environment config).")

        # --- STT Configuration (Classic) ---
        settings.stt_host_arg = stt_host
        settings.stt_port_arg = stt_port
        settings.stt_model_arg = stt_model
        settings.stt_api_key = stt_api_key 
        settings.stt_no_speech_prob_threshold = stt_no_speech_prob_threshold
        settings.stt_avg_logprob_threshold = stt_avg_logprob_threshold
        settings.stt_min_words_threshold = stt_min_words_threshold

        settings.is_openai_stt = settings.stt_host_arg == "api.openai.com"
        if settings.is_openai_stt:
            settings.stt_api_base = "https://api.openai.com/v1"
            logger.info(f"Using OpenAI STT at: {settings.stt_api_base} with model {settings.stt_model_arg}")
            if not settings.stt_api_key:
                logger.critical(
                    "STT_API_KEY (--stt-api-key) is required when using OpenAI STT (stt-host=api.openai.com) with classic backend. Exiting."
                )
                return 1
            logger.info("Using STT API key for OpenAI STT.")
        else: 
            try:
                stt_port_int = int(settings.stt_port_arg)
                scheme = "http" 
                settings.stt_api_base = f"{scheme}://{settings.stt_host_arg}:{stt_port_int}/v1"
                logger.info(f"Using Custom STT server at: {settings.stt_api_base} with model {settings.stt_model_arg}")
                if settings.stt_api_key:
                    logger.info("Using STT API key for custom STT server.")
                else:
                    logger.info("No STT API key provided for custom STT server (assumed optional).")
            except (ValueError, TypeError):
                logger.critical(
                    f"Invalid STT port specified for custom server: '{settings.stt_port_arg}'. Cannot connect. Exiting."
                )
                return 1
        logger.info(
            f"STT Confidence Thresholds: no_speech_prob > {settings.stt_no_speech_prob_threshold}, avg_logprob < {settings.stt_avg_logprob_threshold}, min_words < {settings.stt_min_words_threshold}"
        )

        # --- TTS Configuration (Classic) ---
        settings.tts_host_arg = tts_host
        settings.tts_port_arg = tts_port
        settings.tts_model_arg = tts_model
        settings.tts_voice_arg = tts_voice # Classic backend TTS voice
        settings.tts_api_key = tts_api_key 
        settings.tts_speed_arg = tts_speed 
        settings.tts_acronym_preserve_list_arg = tts_acronym_preserve_list

        settings.is_openai_tts = settings.tts_host_arg == "api.openai.com"
        if settings.is_openai_tts:
            settings.tts_base_url = "https://api.openai.com/v1"
            logger.info(f"Using OpenAI TTS at: {settings.tts_base_url} with model {settings.tts_model_arg}")
            if not settings.tts_api_key:
                logger.critical(
                    "TTS_API_KEY (--tts-api-key) is required when using OpenAI TTS (tts-host=api.openai.com) with classic backend. Exiting."
                )
                return 1
            logger.info("Using TTS API key for OpenAI TTS.")
            if settings.tts_model_arg in OPENAI_TTS_PRICING:
                logger.info(
                    f"OpenAI TTS pricing for '{settings.tts_model_arg}': ${OPENAI_TTS_PRICING[settings.tts_model_arg]:.2f} / 1M chars"
                )
        else: 
            try:
                tts_port_int = int(settings.tts_port_arg)
                scheme = "http" 
                settings.tts_base_url = f"{scheme}://{settings.tts_host_arg}:{tts_port_int}/v1"
                logger.info(f"Using Custom TTS server at: {settings.tts_base_url} with model {settings.tts_model_arg}")
                if settings.tts_api_key:
                    logger.info("Using TTS API key for custom TTS server.")
                else:
                    logger.info("No TTS API key provided for custom TTS server (assumed optional).")
            except (ValueError, TypeError):
                logger.critical(
                    f"Invalid TTS port specified for custom server: '{settings.tts_port_arg}'. Cannot connect. Exiting."
                )
                return 1
        
        settings.tts_acronym_preserve_set = {
            word.strip().upper()
            for word in settings.tts_acronym_preserve_list_arg.split(",")
            if word.strip()
        }
        logger.debug(f"Loaded TTS_ACRONYM_PRESERVE_SET: {settings.tts_acronym_preserve_set}")
        settings.current_tts_speed = settings.tts_speed_arg
        logger.info(f"Initial TTS speed (classic backend): {settings.current_tts_speed:.1f}")


        # --- Initialize Clients (Classic Backend) ---
        logger.info("Initializing clients for 'classic' backend...")
        try:
            settings.stt_client = OpenAI(
                base_url=settings.stt_api_base,
                api_key=settings.stt_api_key, 
            )
            logger.info(f"STT client initialized for classic backend (target: {settings.stt_api_base}).")
        except Exception as e:
            logger.critical(f"Failed to initialize STT client for classic backend: {e}. Exiting.")
            return 1
        
        try:
            settings.tts_client = OpenAI(
                base_url=settings.tts_base_url,
                api_key=settings.tts_api_key, 
            )
            logger.info(f"TTS client initialized for classic backend (target: {settings.tts_base_url}).")
        except Exception as e:
            logger.critical(f"Failed to initialize TTS client for classic backend: {e}. Exiting.")
            return 1

        # --- Model & Voice Availability (Classic Backend) ---
        if settings.use_llm_proxy:
            settings.available_models, settings.model_cost_data = get_models_and_costs_from_proxy(
                settings.llm_api_base, settings.llm_api_key 
            )
        else:
            settings.available_models, settings.model_cost_data = get_models_and_costs_from_litellm()

        if not settings.available_models:
            logger.warning("No LLM models found from proxy or litellm. Using fallback.")
            settings.available_models = ["fallback/unknown-model"]

        initial_llm_model_preference = settings.llm_model_arg
        if initial_llm_model_preference and initial_llm_model_preference in settings.available_models:
            settings.current_llm_model = initial_llm_model_preference
        elif settings.available_models and settings.available_models[0] != "fallback/unknown-model":
            if initial_llm_model_preference:
                logger.warning(f"LLM model '{initial_llm_model_preference}' not found. Using first available: {settings.available_models[0]}.")
            settings.current_llm_model = settings.available_models[0]
        else:
            settings.current_llm_model = initial_llm_model_preference or "fallback/unknown-model"
            logger.warning(f"Using specified or fallback LLM model: {settings.current_llm_model}. Availability/cost data might be missing.")
        logger.info(f"Initial LLM model (classic backend) set to: {settings.current_llm_model}")
        
        if settings.is_openai_tts:
            settings.available_voices_tts = OPENAI_TTS_VOICES
        else: 
            settings.available_voices_tts = get_voices(settings.tts_base_url, settings.tts_api_key)
            if not settings.available_voices_tts:
                logger.warning(f"Could not retrieve voices from custom TTS server at {settings.tts_base_url}.")
        logger.info(f"Available TTS voices (classic backend): {settings.available_voices_tts}")

        initial_tts_voice_preference = settings.tts_voice_arg # Classic TTS voice
        if initial_tts_voice_preference and initial_tts_voice_preference in settings.available_voices_tts:
            settings.current_tts_voice = initial_tts_voice_preference
        elif settings.available_voices_tts:
            if initial_tts_voice_preference:
                 logger.warning(f"Classic TTS voice '{initial_tts_voice_preference}' not found. Using first available: {settings.available_voices_tts[0]}.")
            settings.current_tts_voice = settings.available_voices_tts[0]
        else:
            settings.current_tts_voice = initial_tts_voice_preference
            logger.error(f"No classic TTS voices available or specified voice '{settings.current_tts_voice}' is invalid. TTS may fail.")
        logger.info(f"Initial classic TTS voice set to: {settings.current_tts_voice}")

    # --- Common Post-Backend-Specific Setup ---
    if settings.system_message:
        logger.info(f"Loaded SYSTEM_MESSAGE: '{settings.system_message[:50]}...'")
    else:
        logger.info("No SYSTEM_MESSAGE defined.")

    try:
        app_name = "SimpleVoiceChat"
        app_author = "Attila"
        user_data_dir = Path(platformdirs.user_data_dir(app_name, app_author))
        settings.chat_log_dir = user_data_dir / "chats"
        settings.chat_log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Chat log directory set to: {settings.chat_log_dir}")
    except Exception as e:
        logger.error(f"Failed to create chat log directory: {e}. Chat logging disabled.")
        settings.chat_log_dir = None

    # TTS audio directory setup (primarily for classic backend)
    if settings.backend == "classic":
        try:
            app_name = "SimpleVoiceChat"
            app_author = "Attila"
            try:
                cache_base_dir = Path(platformdirs.user_cache_dir(app_name, app_author))
            except Exception:
                logger.warning("Could not find user cache directory, falling back to user data directory for TTS audio.")
                cache_base_dir = Path(platformdirs.user_data_dir(app_name, app_author))

            settings.tts_base_dir = cache_base_dir / "tts_audio"
            settings.tts_base_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Base TTS audio directory: {settings.tts_base_dir}")

            settings.tts_audio_dir = settings.tts_base_dir / settings.startup_timestamp_str
            settings.tts_audio_dir.mkdir(exist_ok=True)
            logger.info(f"This run's TTS audio directory (classic backend): {settings.tts_audio_dir}")
        except Exception as e:
            logger.error(f"Failed to create temporary TTS audio directory for classic backend: {e}. TTS audio saving might fail.")
    elif settings.backend == "openai" or settings.backend == "gemini": # Add Gemini
        logger.info(f"TTS audio file saving to disk is not applicable for '{settings.backend}' backend.")
        settings.tts_audio_dir = None 


    logger.info(f"Application server host: {settings.host}")
    logger.info(
        f"Application server preferred port: {settings.preferred_port}"
    )

    # --- Stream Handler Setup ---
    stream_handler: Any 
    if settings.backend == "openai":
        logger.info("Initializing Stream with OpenAIRealtimeHandler.")
        stream_handler = OpenAIRealtimeHandler(app_settings=settings)
    elif settings.backend == "gemini": # Add Gemini case
        logger.info("Initializing Stream with GeminiRealtimeHandler.")
        stream_handler = GeminiRealtimeHandler(app_settings=settings)
    else: 
        logger.info("Initializing Stream with ReplyOnPause handler for classic backend.")
        stream_handler = ReplyOnPause(
            response, 
            algo_options=AlgoOptions(
                audio_chunk_duration=3.0,
                started_talking_threshold=0.2,
                speech_threshold=0.2,
            ),
            model_options=SileroVadOptions(
                threshold=0.6,
                min_speech_duration_ms=800,
                min_silence_duration_ms=3500,
            ),
            can_interrupt=True,
        )

    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=stream_handler,
        track_constraints={
            "echoCancellation": True,
            "noiseSuppression": {"exact": True}, 
            "autoGainControl": {"exact": True},  
            # Ideal sample rate should match what the handler expects as input.
            # OpenAI expects 24kHz. Gemini expects 16kHz. Classic is flexible.
            "sampleRate": {"ideal": GEMINI_REALTIME_INPUT_SAMPLE_RATE if settings.backend == "gemini" else (OPENAI_REALTIME_SAMPLE_RATE if settings.backend == "openai" else 16000)}, 
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        },
        rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
        concurrency_limit=5 if get_space() else None,
        time_limit=180 if get_space() else None, 
    )

    app = FastAPI()
    stream.mount(app)
    register_endpoints(app, stream)

    current_host = settings.host 
    preferred_port_val = settings.preferred_port
    actual_port = preferred_port_val
    max_retries = 10

    if is_port_in_use(actual_port, current_host):
        logger.warning(
            f"Preferred port {actual_port} on host {current_host} is in use. Searching for an available port..."
        )
        found_port = False
        for attempt in range(max_retries):
            new_port = random.randint(1024, 65535)
            logger.debug(f"Attempt {attempt+1}: Checking port {new_port} on {current_host}...")
            if not is_port_in_use(new_port, current_host):
                actual_port = new_port
                found_port = True
                logger.info(f"Found available port: {actual_port} on host {current_host}")
                break
        if not found_port:
            logger.error(
                f"Could not find an available port on host {current_host} after {max_retries} attempts. Exiting."
            )
            return 1
    else:
        logger.info(f"Using preferred port {actual_port} on host {current_host}")

    settings.port = actual_port 
    url = f"http://{current_host}:{actual_port}"

    def run_server():
        global uvicorn_server
        try:
            config = uvicorn.Config(
                app,
                host=current_host, 
                port=actual_port,   
                log_config=None,
            )
            uvicorn_server = uvicorn.Server(config)
            logger.info(f"Starting Uvicorn server on {current_host}:{actual_port}...")
            uvicorn_server.run()
            logger.info("Uvicorn server has stopped.")
        except Exception as e:
            logger.critical(f"Uvicorn server encountered an error: {e}")
        finally:
            uvicorn_server = None

    monitor_thread = threading.Thread(target=monitor_heartbeat_thread, daemon=True)
    monitor_thread.start()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logger.debug("Waiting for Uvicorn server to initialize...")
    time.sleep(3.0)

    if not server_thread.is_alive() or uvicorn_server is None:
        logger.critical(
            "Uvicorn server thread failed to start or initialize correctly. Exiting."
        )
        return 1
    else:
        logger.debug("Server thread appears to be running.")

    exit_code = 0
    try:
        if settings.browser:
            logger.info(f"Opening application in default web browser at: {url}")
            webbrowser.open(url, new=1)
            logger.info(
                "Application opened in browser. Server is running in the background."
            )
            logger.info("Press Ctrl+C to stop the server.")
            try:
                server_thread.join()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down.")
            finally:
                logger.info("Signaling heartbeat monitor thread to stop...")
                shutdown_event.set()
                if (
                    uvicorn_server
                    and server_thread.is_alive()
                    and not uvicorn_server.should_exit
                ):
                    logger.info("Signaling Uvicorn server to shut down...")
                    uvicorn_server.should_exit = True
                elif uvicorn_server and uvicorn_server.should_exit:
                    logger.info("Uvicorn server already signaled to shut down.")
                elif not server_thread.is_alive():
                    logger.info("Server thread already stopped.")
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown."
                    )
                if server_thread.is_alive():
                    logger.info("Waiting for Uvicorn server thread to join...")
                    server_thread.join(timeout=5.0)
                    if server_thread.is_alive():
                        logger.warning(
                            "Uvicorn server thread did not exit gracefully after 5 seconds."
                        )
                    else:
                        logger.info("Uvicorn server thread joined successfully.")
                logger.info("Waiting for heartbeat monitor thread to join...")
                monitor_thread.join(timeout=2.0)
                if monitor_thread.is_alive():
                    logger.warning(
                        "Heartbeat monitor thread did not exit gracefully after 2 seconds."
                    )
                else:
                    logger.info("Heartbeat monitor thread joined successfully.")
        else:
            logger.info(f"Creating pywebview window for URL: {url}")
            api = Api(None)
            webview.settings['OPEN_DEVTOOLS_IN_DEBUG'] = False
            logger.info(f"pywebview setting OPEN_DEVTOOLS_IN_DEBUG set to False.")
            
            pywebview_window = webview.create_window(
                f"Simple Voice Chat v{APP_VERSION}",
                url,
                width=800,
                height=800,
                js_api=api,
            )
            api._window = pywebview_window

            logger.info("Starting pywebview...")
            try:
                webview.start(debug=True, gui='qt')
            except Exception as e:
                logger.critical(f"Pywebview encountered an error: {e}")
                exit_code = 1
            finally:
                logger.info("Pywebview window closed or heartbeat timed out.")
                logger.info("Signaling heartbeat monitor thread to stop...")
                shutdown_event.set()
                if uvicorn_server and not uvicorn_server.should_exit:
                    logger.info("Signaling Uvicorn server to shut down...")
                    uvicorn_server.should_exit = True
                elif uvicorn_server and uvicorn_server.should_exit:
                    logger.info("Uvicorn server already signaled to shut down.")
                else:
                    logger.warning(
                        "Uvicorn server instance not found, cannot signal shutdown."
                    )
                logger.info("Waiting for Uvicorn server thread to join...")
                server_thread.join(timeout=5.0)
                if server_thread.is_alive():
                    logger.warning(
                        "Uvicorn server thread did not exit gracefully after 5 seconds."
                    )
                else:
                    logger.info("Uvicorn server thread joined successfully.")
                logger.info("Waiting for heartbeat monitor thread to join...")
                monitor_thread.join(timeout=2.0)
                if monitor_thread.is_alive():
                    logger.warning(
                        "Heartbeat monitor thread did not exit gracefully after 2 seconds."
                    )
                else:
                    logger.info("Heartbeat monitor thread joined successfully.")
    except Exception as e:
        logger.critical(
            f"An unexpected error occurred in the main execution block: {e}",
            exc_info=True,
        )
        exit_code = 1
        shutdown_event.set()
        if uvicorn_server and not uvicorn_server.should_exit:
            uvicorn_server.should_exit = True

    logger.info(f"Main function returning exit code: {exit_code}")
    return exit_code

