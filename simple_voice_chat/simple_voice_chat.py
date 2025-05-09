import json
import random
import sys
import threading
import time
import datetime
import uvicorn
import webbrowser
import tempfile
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional

from loguru import logger
import litellm
import numpy as np
import webview
import platformdirs
import click # Added click

from fastapi import FastAPI, Request, HTTPException
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
)
from gradio.utils import get_space
from openai import OpenAI, AuthenticationError
from pydantic import BaseModel
from pydub import AudioSegment

# --- Import Configuration ---
# This 'settings' instance will be populated in main() and used throughout.
from .utils.config import settings, APP_VERSION, OPENAI_TTS_PRICING, OPENAI_TTS_VOICES
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
from .logging_config import setup_logging

# --- Global Variables (Runtime State - Not Configuration) ---
# These are primarily for managing the server and UI state, not app settings.
last_heartbeat_time: datetime.datetime | None = None
heartbeat_timeout: int = 15  # Seconds before assuming client disconnected
shutdown_event = threading.Event()  # Used to signal monitor thread to stop
pywebview_window = None  # To hold the pywebview window object if created
uvicorn_server = None # Global variable to hold the Uvicorn server instance
# --- End Global Configuration & State ---


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


# --- Core Response Logic (Async Streaming with Background TTS) ---
async def response(
    audio: tuple[int, np.ndarray],
    chatbot: list[dict] | None = None,
) -> AsyncGenerator[Any, None]:
    """
    Handles audio input, performs STT, streams LLM response text chunks to UI,
    generates TTS concurrently, and yields final audio and updates.
    """
    # Access module-level variables set after arg parsing in main()
    # Ensure clients are initialized (should be, but good practice)
    if not settings.stt_client or not settings.tts_client:
        logger.error("STT or TTS client not initialized. Cannot process request.")
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
        if settings.llm_api_key:
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
                                settings.tts_client,
                                settings.tts_model_arg, # Use model from initial args/env
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
                settings.tts_client,
                settings.tts_model_arg, # Use model from initial args/env
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

        # Calculate TTS cost if applicable
        if settings.is_openai_tts and total_tts_chars > 0:
            tts_model_used = settings.tts_model_arg # Use model from initial args/env
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

        # Calculate LLM cost (if usage info available)
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
            )  # This might be misleading, maybe remove? Let's keep it for structure.

        # Yield combined cost update
        logger.info(f"Yielding combined cost update: {cost_result}")
        yield AdditionalOutputs({"type": "cost_update", "data": cost_result})
        logger.info("Cost update yielded.")

        # 2. Add Full Assistant Text Response to History (to the copy) with Metadata
        assistant_message_obj = None # Define outside the 'if'
        if not llm_error_occurred and full_response_text:
            # Create assistant message metadata
            assistant_metadata = ChatMessageMetadata(
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                llm_model=settings.current_llm_model,
                usage=final_usage_info, # This is the dict from LLM response
                cost=cost_result, # This is the dict calculated earlier {llm_cost..., tts_cost}
                tts_audio_file_paths=tts_audio_file_paths if tts_audio_file_paths else None # Add the list of audio file paths
            )
            # Create the full message object
            assistant_message_obj = ChatMessage(
                role="assistant",
                content=full_response_text,
                metadata=assistant_metadata
            )
            assistant_message_dict = assistant_message_obj.model_dump() # Convert to dict for storage/comparison

            # Check against the copy (comparing dicts) and append to the copy
            # Avoid appending duplicates if somehow the exact same dict exists at the end
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
        # This signals the end of the bot's turn and provides the final history.
        if response_completed_normally:
            logger.info("Yielding final chatbot state update...")
            # Send the modified copy of the history.
            yield AdditionalOutputs(
                {
                    "type": "final_chatbot_state",
                    "history": current_chatbot,
                }  # Yield the copy
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

    except AuthenticationError as e:  # Catch OpenAI specific auth errors
        logger.error(f"OpenAI Authentication Error during processing: {e}")
        response_completed_normally = False
        llm_error_occurred = True  # Treat as LLM/TTS error

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
        # --- Error Handling Path ---
        # Log type and message separately to avoid formatting errors with complex exception strings
        logger.error(
            f"Error during LLM streaming or TTS processing: {type(e).__name__} - {e}", exc_info=True
        )
        response_completed_normally = False  # Ensure this is false on exception
        llm_error_occurred = True  # Ensure error is flagged

        # Yield an error message to the UI
        error_content = f"\n[LLM/TTS Error: {type(e).__name__}]"  # Show error type
        if not first_chunk_yielded:
            # If no text yielded, send as a full message
            llm_error_msg = {"role": "assistant", "content": error_content.strip()}
            yield AdditionalOutputs(
                {"type": "chatbot_update", "message": llm_error_msg}
            )
            # Note: We don't add this error message to current_chatbot to keep history clean
        else:
            # If text chunks were already sent, append error info via chunk update
            yield AdditionalOutputs(
                {"type": "text_chunk_update", "content": error_content}
            )

        # Yield the final chatbot state (as it was when the error occurred)
        # This includes the user message but not the failed assistant response.
        logger.warning("Yielding final chatbot state (after exception)...")
        yield AdditionalOutputs(
            {
                "type": "final_chatbot_state",
                "history": current_chatbot,
            }  # Yield state at time of error
        )
        logger.warning("Final chatbot state (after exception) yielded.")

        # Ensure final status update is yielded last in the error path too
        logger.warning("Yielding final status update (idle, after exception)...")
        yield AdditionalOutputs(
            {
                "type": "status_update",
                "status": "idle",
                "message": "Ready (after error)",
            }
        )
        logger.warning("Final status update (idle, after exception) yielded.")

    # This log executes *after* the try/except/finally block completes and generator exits
    logger.info(
        f"--- Response function generator finished (Completed normally: {response_completed_normally}) ---"
    )


# --- Pydantic Models ---

class ChatMessageMetadata(BaseModel):
    """Optional metadata associated with a chat message."""
    timestamp: Optional[str] = None # ISO format timestamp
    llm_model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None # Allow flexible structure for usage data from LLM
    cost: Optional[Dict[str, Any]] = None # e.g., {'input_cost': 0.0001, 'output_cost': 0.0002, 'total_cost': 0.0003, 'tts_cost': 0.00005}
    stt_details: Optional[Dict[str, Any]] = None # e.g., {'no_speech_prob': 0.1, 'avg_logprob': -0.2}
    tts_audio_file_paths: Optional[List[str]] = None # List of FILENAMES of saved TTS audio files for this message

class ChatMessage(BaseModel):
    """Represents a single message in the chat history."""
    role: str
    content: str
    metadata: Optional[ChatMessageMetadata] = None

# --- FastAPI Setup ---

class InputData(BaseModel):
    """Model for data received by the /input_hook endpoint."""
    webrtc_id: str
    chatbot: list[ChatMessage] # Use the new ChatMessage model


# --- Endpoint Definitions ---
# These need access to the 'app' and 'stream' objects created in main()
# We can define them here but register them with the app inside main()


def register_endpoints(app: FastAPI, stream: Stream):
    """Registers FastAPI endpoints."""
    # Get current directory relative to this file
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
        # Inject RTC config
        if rtc_config:
            html_content = html_content.replace(
                "__RTC_CONFIGURATION__", json.dumps(rtc_config)
            )
        else:
            html_content = html_content.replace("__RTC_CONFIGURATION__", "null")

        # Inject the system message (using the final SYSTEM_MESSAGE string)
        html_content = html_content.replace(
            "__SYSTEM_MESSAGE_JSON__", json.dumps(settings.system_message)
        )
        # Inject the application version
        html_content = html_content.replace("__APP_VERSION__", APP_VERSION)
        # Inject the initial STT language (using the final current_stt_language string)
        html_content = html_content.replace(
            "__STT_LANGUAGE_JSON__", json.dumps(settings.current_stt_language)
        )
        # Inject the initial TTS speed
        html_content = html_content.replace(
            "__TTS_SPEED_JSON__", json.dumps(settings.current_tts_speed)
        )
        # Inject the startup timestamp string
        html_content = html_content.replace(
            "__STARTUP_TIMESTAMP_STR__", json.dumps(settings.startup_timestamp_str)
        )


        return HTMLResponse(content=html_content, status_code=200)

    @app.post("/input_hook")
    async def _(body: InputData):
        # The body.chatbot is now a list of ChatMessage objects thanks to Pydantic validation
        # Convert them to dictionaries for internal use (e.g., passing to stream handler)
        # Ensure metadata is preserved if present. model_dump(exclude_none=True) might be useful
        # if we want to keep the JSON clean, but let's keep all fields for now.
        chatbot_history = [msg.model_dump() for msg in body.chatbot]

        # Assuming fastrtc handler `response` expects a list of dictionaries matching ChatMessage structure.
        # If issues arise, we might need to store/retrieve state differently.
        stream.set_input(body.webrtc_id, chatbot_history)  # Keep as is for now
        return {"status": "ok"}

    @app.get("/outputs")
    def _(webrtc_id: str):
        async def output_stream():
            try:
                async for output in stream.output_stream(webrtc_id):
                    # The async handler `response` now yields audio tuples directly
                    # and AdditionalOutputs for SSE events.
                    if isinstance(output, AdditionalOutputs):
                        data_payload = output.args[0]
                        if isinstance(data_payload, dict) and "type" in data_payload:
                            event_type = data_payload["type"]
                            try:
                                event_data_json = json.dumps(data_payload)
                                logger.debug(
                                    f"Sending SSE event: type={event_type}, data={event_data_json[:100]}..."
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
                    # Audio chunks are handled by WebRTC track, not sent via SSE.
                    # The handler yields them, fastrtc puts them on the track.
                    elif (
                        isinstance(output, tuple)
                        and len(output) == 2
                        and isinstance(output[1], np.ndarray)
                    ):
                        # This case should be handled by fastrtc internally for the audio track.
                        # We don't need to send it via SSE. Log if it appears here unexpectedly.
                        logger.debug(
                            f"Output stream received audio tuple for webrtc_id {webrtc_id}, should be handled by track."
                        )
                        pass  # Audio is sent via WebRTC track
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
                # Optionally send an error event to the client via SSE if possible
                try:
                    error_payload = {
                        "type": "error_event",
                        "message": f"Server stream error: {e}",
                    }
                    yield f"event: error_event\ndata: {json.dumps(error_payload)}\n\n"
                except Exception as send_err:
                    logger.error(
                        f"Failed to send error event to client {webrtc_id}: {send_err}"
                    )

        return StreamingResponse(output_stream(), media_type="text/event-stream")

    # --- Endpoint to Get Available Models ---
    @app.get("/available_models")
    async def get_available_models():
        # Access from settings object
        return JSONResponse(
            {"available": settings.available_models, "current": settings.current_llm_model}
        )

    # --- Endpoint to Get Available Voices ---
    @app.get("/available_voices_tts")
    async def get_available_voices():
        # Access from settings object
        response_data = prepare_available_voices_data(
            settings.current_tts_voice, settings.available_voices_tts
        )
        return JSONResponse(response_data)

    # --- End Voice Endpoint ---

    # --- Endpoint to Switch Voice ---
    @app.post("/switch_voice")
    async def switch_voice(request: Request):
        # Modify settings.current_tts_voice
        try:
            data = await request.json()
            new_voice_name = data.get("voice_name")
            if new_voice_name:
                if new_voice_name != settings.current_tts_voice:
                    if new_voice_name in settings.available_voices_tts:
                        settings.current_tts_voice = new_voice_name
                        logger.info(f"Switched active TTS voice to: {settings.current_tts_voice}")
                        return JSONResponse(
                            {"status": "success", "voice": settings.current_tts_voice}
                        )
                    else:
                        # Voice not found in the available list
                        logger.warning(
                            f"Attempted to switch to voice '{new_voice_name}' which is not in the available list: {settings.available_voices_tts}"
                        )
                        return JSONResponse(
                            {"status": "error", "message": f"Voice '{new_voice_name}' is not available."},
                            status_code=400,
                        )
                else:
                    logger.info(f"Voice already set to: {new_voice_name}")
                    return JSONResponse(
                        {"status": "success", "voice": settings.current_tts_voice}
                    )  # Still success
            else: # No voice_name provided
                logger.warning(f"Missing voice_name in switch request.")
                return JSONResponse(
                    {
                        "status": "error",
                        "message": f"Missing 'voice_name' in request body.",
                    },
                    status_code=400,
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
                {"status": "error", "message": f"Internal server error: {e}"},
                status_code=500,
            )

    # --- End Switch Voice Endpoint ---

    # --- Endpoint to Switch STT Language ---
    @app.post("/switch_stt_language")
    async def switch_stt_language(request: Request):
        # Modify settings.current_stt_language
        try:
            data = await request.json()
            # Allow empty string to clear the language setting (use auto-detect)
            new_language = data.get("stt_language", None) # Get value, None if key missing

            # Normalize empty string or None to None for internal consistency
            if new_language is not None and not new_language.strip():
                new_language = None
            elif new_language is not None:
                new_language = new_language.strip() # Use stripped value if not empty

            if new_language != settings.current_stt_language:
                settings.current_stt_language = new_language
                logger.info(
                    f"Switched active STT language to: '{settings.current_stt_language}' (None means auto-detect)"
                )
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
                {"status": "error", "message": f"Internal server error: {e}"},
                status_code=500,
            )
    # --- End Switch STT Language Endpoint ---

    # --- Endpoint to Switch TTS Speed ---
    @app.post("/switch_tts_speed")
    async def switch_tts_speed(request: Request):
        # Modify settings.current_tts_speed
        try:
            data = await request.json()
            new_speed = data.get("tts_speed") # Get value

            if new_speed is None:
                logger.warning("Missing 'tts_speed' in switch request.")
                return JSONResponse(
                    {"status": "error", "message": "Missing 'tts_speed' in request body."},
                    status_code=400,
                )

            try:
                new_speed_float = float(new_speed)
                # Validate range
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
                    f"Switched active TTS speed to: {settings.current_tts_speed:.1f}"
                )
                return JSONResponse(
                    {"status": "success", "tts_speed": settings.current_tts_speed}
                )
            else:
                logger.info(
                    f"TTS speed already set to: {settings.current_tts_speed:.1f}"
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
                {"status": "error", "message": f"Internal server error: {e}"},
                status_code=500,
            )
    # --- End Switch TTS Speed Endpoint ---

    # --- Endpoint to Switch Model ---
    @app.post("/switch_model")
    async def switch_model(request: Request):
        # Modify settings.current_llm_model
        try:
            data = await request.json()
            new_model_name = data.get(
                "model_name"
            )  # This name comes from the frontend dropdown (already prefixed if from proxy)

            if new_model_name:
                if new_model_name != settings.current_llm_model:
                    # Check if the new model exists in our loaded list (which contains prefixed names if applicable)
                    if new_model_name in settings.available_models:
                        settings.current_llm_model = new_model_name
                        logger.info(
                            f"Switched active LLM model to: {settings.current_llm_model}"
                        )
                        # Check if cost data is available for the switched model (using the potentially prefixed name)
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
                        # Model not found in the available list
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
                    )  # Still success
            else: # No model_name provided
                logger.warning(f"Missing model_name in switch request.")
                return JSONResponse(
                    {
                        "status": "error",
                        "message": f"Missing 'model_name' in request body.",
                    },
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
                {"status": "error", "message": f"Internal server error: {e}"},
                status_code=500,
            )

    # --- Heartbeat Endpoint ---
    @app.post("/heartbeat")
    async def heartbeat(request: Request):
        """Receives heartbeat pings from the frontend."""
        global last_heartbeat_time
        try:
            # Update the last heartbeat time using timezone-aware datetime
            last_heartbeat_time = datetime.datetime.now(datetime.timezone.utc)
            # Log received heartbeat and payload for debugging
            payload = await request.json()
            logger.debug(
                f"Heartbeat received at {last_heartbeat_time}. Payload: {payload}"
            )
            return {"status": "ok"}
        except json.JSONDecodeError:
            # Handle cases where the body might not be valid JSON (e.g., empty from sendBeacon)
            last_heartbeat_time = datetime.datetime.now(datetime.timezone.utc)
            logger.debug(
                f"Heartbeat received at {last_heartbeat_time} (no valid JSON payload)."
            )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    # --- Endpoint to Reset Chat Log Timestamp ---
    @app.post("/reset_chat_log")
    async def reset_chat_log():
        """Resets the timestamp used for the chat log filename."""
        # Modify settings.startup_timestamp_str
        try:
            new_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            settings.startup_timestamp_str = new_timestamp
            logger.info(f"Chat log timestamp reset to: {settings.startup_timestamp_str}")
            return JSONResponse(
                {"status": "success", "new_timestamp": settings.startup_timestamp_str}
            )
        except Exception as e:
            logger.error(f"Error resetting chat log timestamp: {e}")
            return JSONResponse(
                {"status": "error", "message": "Failed to reset chat log timestamp"},
                status_code=500,
            )

    # --- End Reset Chat Log Endpoint ---

    # --- Endpoint to Serve TTS Audio Files ---
    @app.get("/tts_audio/{run_timestamp}/{filename}")
    async def get_tts_audio(run_timestamp: str, filename: str):
        """Serves a specific TTS audio file from the cache."""
        if not settings.tts_audio_dir:
             logger.error("settings.tts_audio_dir not configured, cannot serve audio.")
             raise HTTPException(status_code=500, detail="Server configuration error")

        # Basic security check: Ensure filename looks safe (e.g., no path traversal)
        # A more robust check might involve ensuring it matches the expected UUID format.
        if ".." in filename or "/" in filename or "\\" in filename:
            logger.warning(f"Attempt to access potentially unsafe filename: {filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")

        # Construct the expected path based on the *current run's* TTS_AUDIO_DIR
        # We ignore the run_timestamp from the URL path for security, always serving from the current run's dir.
        # This prevents accessing audio from previous runs.
        # If access to previous runs is needed, the logic here would need to change significantly
        # and potentially involve searching through subdirectories based on the timestamp, using settings.tts_base_dir.
        file_path = settings.tts_audio_dir / filename
        logger.debug(f"Attempting to serve TTS audio file: {file_path} (URL timestamp '{run_timestamp}' ignored for security)")

        if file_path.is_file():
            return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
        else:
            logger.warning(f"TTS audio file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Audio file not found")
    # --- End TTS Audio Endpoint ---


# --- Pywebview API Class ---
class Api:
    def __init__(self, window):
        self._window = window

    def close(self):
        """Close the pywebview window."""
        logger.info("API close method called.")
        if self._window:
            self._window.destroy()


# --- Heartbeat Globals ---
# These are already defined at the top of the file.
# last_heartbeat_time: datetime.datetime | None = None
# heartbeat_timeout: int = 15  # Seconds before assuming client disconnected
# shutdown_event = threading.Event()  # Used to signal monitor thread to stop
# pywebview_window = None  # To hold the pywebview window object if created
# --- End Heartbeat Globals ---

# Global variable to hold the Uvicorn server instance
# This is already defined at the top of the file.
# uvicorn_server = None

# Global variable to hold parsed args (needed by response and endpoints)
# This will be removed. Settings object will be used instead.
# args: Optional[argparse.Namespace] = None # REMOVED


# --- Heartbeat Monitoring Thread ---
def monitor_heartbeat_thread():
    """Monitors the time since the last heartbeat and triggers shutdown if timeout occurs."""
    global last_heartbeat_time, uvicorn_server, pywebview_window, shutdown_event
    logger.info("Heartbeat monitor thread started.")
    initial_wait_done = False

    while not shutdown_event.is_set():
        if last_heartbeat_time is None:
            if not initial_wait_done:
                logger.info(
                    f"Waiting for the first heartbeat (timeout check in {heartbeat_timeout * 2}s)..."
                )
                # Wait longer initially before the first check
                shutdown_event.wait(heartbeat_timeout * 2)
                initial_wait_done = True
                if shutdown_event.is_set():
                    break  # Exit if shutdown requested during initial wait
                continue  # Re-check condition after initial wait
            else:
                # If still None after initial wait, maybe log periodically?
                logger.debug("Still waiting for first heartbeat...")
                shutdown_event.wait(5)  # Check every 5 seconds after initial wait
                if shutdown_event.is_set():
                    break
                continue

        # Calculate time since last heartbeat
        time_since_last = (
            datetime.datetime.now(datetime.timezone.utc) - last_heartbeat_time
        )
        logger.debug(
            f"Time since last heartbeat: {time_since_last.total_seconds():.1f}s"
        )

        if time_since_last.total_seconds() > heartbeat_timeout:
            logger.warning(
                f"Heartbeat timeout ({heartbeat_timeout}s exceeded). Initiating shutdown."
            )
            # 1. Signal Uvicorn server to shut down
            if uvicorn_server:
                logger.info("Signaling Uvicorn server to stop...")
                uvicorn_server.should_exit = True
            else:
                logger.warning(
                    "Uvicorn server instance not found, cannot signal shutdown."
                )

            # 2. If in pywebview mode, destroy the window to unblock the main thread
            if pywebview_window:
                logger.info("Destroying pywebview window...")
                try:
                    # Schedule the destroy call on the main GUI thread if necessary
                    # For simplicity, try direct call first, might work depending on pywebview version/OS
                    pywebview_window.destroy()
                except Exception as e:
                    logger.error(
                        f"Error destroying pywebview window from monitor thread: {e}"
                    )
                    # Fallback: Try to signal main thread differently if needed, or rely on Uvicorn shutdown

            # 3. Exit the monitor thread
            break

        # Wait for a short interval before checking again
        shutdown_event.wait(5)  # Check every 5 seconds

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
    default=int(APP_PORT_ENV), # Resolved default from env.py or actual env
    show_default=True, # Shows the resolved default
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
    default=SYSTEM_MESSAGE_ENV, # Can be None or empty string
    show_default=True,
    help="System message to prepend to the chat history. (Env: SYSTEM_MESSAGE)",
)
@click.option(
    "--llm-host",
    type=str,
    envvar="LLM_HOST",
    default=LLM_HOST_ENV, # Can be None
    show_default=True,
    help="Host address of the LLM proxy server (optional). (Env: LLM_HOST)",
)
@click.option(
    "--llm-port",
    type=str, 
    envvar="LLM_PORT",
    default=LLM_PORT_ENV, # Can be None
    show_default=True,
    help="Port of the LLM proxy server (optional). (Env: LLM_PORT)",
)
@click.option(
    "--llm-model",
    type=str,
    envvar="LLM_MODEL",
    default=DEFAULT_LLM_MODEL_ENV,
    show_default=True,
    help="Default LLM model to use (e.g., 'gpt-4o', 'litellm_proxy/claude-3-opus'). (Env: LLM_MODEL)",
)
@click.option(
    "--llm-api-key",
    type=str,
    envvar="LLM_API_KEY",
    default=LLM_API_KEY_ENV, # Can be None
    show_default=True,
    help="API key for the LLM provider/proxy (optional, depends on setup). (Env: LLM_API_KEY)",
)
@click.option(
    "--stt-host",
    type=str,
    envvar="STT_HOST",
    default=STT_HOST_ENV,
    show_default=True,
    help="Host address of the STT server (e.g., 'api.openai.com' or 'localhost'). (Env: STT_HOST)",
)
@click.option(
    "--stt-port",
    type=str, 
    envvar="STT_PORT",
    default=STT_PORT_ENV,
    show_default=True,
    help="Port of the STT server (e.g., 443 for OpenAI, 8002 for local). (Env: STT_PORT)",
)
@click.option(
    "--stt-model",
    type=str,
    envvar="STT_MODEL",
    default=STT_MODEL_ENV,
    show_default=True,
    help="STT model to use (e.g., 'whisper-1' for OpenAI, 'deepdml/faster-whisper-large-v3-turbo-ct2' for local). (Env: STT_MODEL)",
)
@click.option(
    "--stt-language",
    type=str,
    envvar="STT_LANGUAGE",
    default=STT_LANGUAGE_ENV, # Can be None or empty string
    show_default=True,
    help="Language code for STT (e.g., 'en', 'fr'). If unset, Whisper usually auto-detects. (Env: STT_LANGUAGE)",
)
@click.option(
    "--stt-api-key",
    type=str,
    envvar="STT_API_KEY",
    default=STT_API_KEY_ENV, # Can be None
    show_default=True,
    help="API key for the STT server (REQUIRED for OpenAI STT). (Env: STT_API_KEY)",
)
@click.option(
    "--stt-no-speech-prob-threshold",
    type=click.FLOAT,
    envvar="STT_NO_SPEECH_PROB_THRESHOLD",
    default=float(STT_NO_SPEECH_PROB_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence: Reject if no_speech_prob > this. (Env: STT_NO_SPEECH_PROB_THRESHOLD)",
)
@click.option(
    "--stt-avg-logprob-threshold",
    type=click.FLOAT,
    envvar="STT_AVG_LOGPROB_THRESHOLD",
    default=float(STT_AVG_LOGPROB_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence: Reject if avg_logprob < this. (Env: STT_AVG_LOGPROB_THRESHOLD)",
)
@click.option(
    "--stt-min-words-threshold",
    type=click.INT,
    envvar="STT_MIN_WORDS_THRESHOLD",
    default=int(STT_MIN_WORDS_THRESHOLD_ENV),
    show_default=True,
    help=f"STT confidence: Reject if word count < this. (Env: STT_MIN_WORDS_THRESHOLD)",
)
@click.option(
    "--tts-host",
    type=str,
    envvar="TTS_HOST",
    default=TTS_HOST_ENV,
    show_default=True,
    help="Host address of the TTS server (e.g., 'api.openai.com' or 'localhost'). (Env: TTS_HOST)",
)
@click.option(
    "--tts-port",
    type=str, 
    envvar="TTS_PORT",
    default=TTS_PORT_ENV,
    show_default=True,
    help="Port of the TTS server (e.g., 443 for OpenAI, 8880 for local). (Env: TTS_PORT)",
)
@click.option(
    "--tts-model",
    type=str,
    envvar="TTS_MODEL",
    default=TTS_MODEL_ENV,
    show_default=True,
    help="TTS model to use (e.g., 'tts-1', 'tts-1-hd' for OpenAI, 'kokoro' for local). (Env: TTS_MODEL)",
)
@click.option(
    "--tts-voice",
    type=str,
    envvar="TTS_VOICE",
    default=DEFAULT_VOICE_TTS_ENV,
    show_default=True,
    help="Default TTS voice to use (e.g., 'alloy', 'ash' for OpenAI, 'ff_siwis' for local). (Env: TTS_VOICE)",
)
@click.option(
    "--tts-api-key",
    type=str,
    envvar="TTS_API_KEY",
    default=TTS_API_KEY_ENV, # Can be None
    show_default=True,
    help="API key for the TTS server (REQUIRED for OpenAI TTS). (Env: TTS_API_KEY)",
)
@click.option(
    "--tts-speed",
    type=click.FLOAT,
    envvar="TTS_SPEED",
    default=float(DEFAULT_TTS_SPEED_ENV),
    show_default=True,
    help=f"Default TTS speed multiplier. (Env: TTS_SPEED)",
)
@click.option(
    "--tts-acronym-preserve-list",
    type=str,
    envvar="TTS_ACRONYM_PRESERVE_LIST",
    default=TTS_ACRONYM_PRESERVE_LIST_ENV, # Can be empty string
    show_default=True,
    help=f"Comma-separated list of acronyms to preserve during TTS (Kokoro TTS). (Env: TTS_ACRONYM_PRESERVE_LIST)",
)
def main(
    host: str,
    port: int,
    verbose: bool,
    browser: bool,
    system_message: Optional[str],
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
    tts_voice: str,
    tts_api_key: Optional[str],
    tts_speed: float,
    tts_acronym_preserve_list: str,
) -> int:
    """Main function to parse arguments, set up, and run the application."""
    # Access global runtime state variables (not config)
    global uvicorn_server, pywebview_window

    # --- Record Startup Time ---
    startup_time = datetime.datetime.now()
    startup_timestamp_str_local = startup_time.strftime("%Y%m%d_%H%M%S")

    # --- Argument Parsing is now handled by Click decorators ---

    # --- Apply Argument Values to Global Configuration ---
    settings.startup_timestamp_str = startup_timestamp_str_local

    # General
    settings.preferred_port = port
    settings.host = host
    settings.verbose = verbose
    settings.browser = browser
    settings.system_message = system_message.strip() if system_message is not None else ""

    # LLM Configuration
    settings.llm_host_arg = llm_host
    settings.llm_port_arg = llm_port
    settings.llm_model_arg = llm_model
    settings.llm_api_key = llm_api_key

    settings.use_llm_proxy = bool(settings.llm_host_arg and settings.llm_port_arg)
    if settings.use_llm_proxy:
        try:
            llm_port_int = int(settings.llm_port_arg)
            settings.llm_api_base = f"http://{settings.llm_host_arg}:{llm_port_int}/v1"
        except (ValueError, TypeError):
            print(
                f"Error: Invalid LLM port specified: '{settings.llm_port_arg}'. Disabling proxy.", file=sys.stderr
            )
            settings.use_llm_proxy = False
            settings.llm_api_base = None
    else:
        settings.llm_api_base = None

    # STT Configuration
    settings.stt_host_arg = stt_host
    settings.stt_port_arg = stt_port
    settings.stt_model_arg = stt_model
    settings.stt_language_arg = stt_language
    settings.stt_api_key = stt_api_key
    settings.stt_no_speech_prob_threshold = stt_no_speech_prob_threshold
    settings.stt_avg_logprob_threshold = stt_avg_logprob_threshold
    settings.stt_min_words_threshold = stt_min_words_threshold

    settings.is_openai_stt = settings.stt_host_arg == "api.openai.com"
    if settings.is_openai_stt:
        settings.stt_api_base = "https://api.openai.com/v1"
        if not settings.stt_api_key:
            print(
                "Critical Error: STT_API_KEY is required when using OpenAI STT (stt-host=api.openai.com). "
                "Set the STT_API_KEY environment variable or provide --stt-api-key argument. Exiting.", file=sys.stderr
            )
            return 1
    else:
        try:
            stt_port_int = int(settings.stt_port_arg)
            scheme = "http"
            settings.stt_api_base = f"{scheme}://{settings.stt_host_arg}:{stt_port_int}/v1"
            if not settings.stt_api_key:
                print(
                    f"Warning: No STT API key provided for custom server at {settings.stt_api_base}. Assuming it's not needed.", file=sys.stderr
                )
        except (ValueError, TypeError):
            print(
                f"Critical Error: Invalid STT port specified for custom server: '{settings.stt_port_arg}'. Cannot connect. Exiting.", file=sys.stderr
            )
            return 1

    # TTS Configuration
    settings.tts_host_arg = tts_host
    settings.tts_port_arg = tts_port
    settings.tts_model_arg = tts_model
    settings.tts_voice_arg = tts_voice
    settings.tts_api_key = tts_api_key
    settings.tts_speed_arg = tts_speed
    settings.tts_acronym_preserve_list_arg = tts_acronym_preserve_list

    settings.is_openai_tts = settings.tts_host_arg == "api.openai.com"
    if settings.is_openai_tts:
        settings.tts_base_url = "https://api.openai.com/v1"
        if not settings.tts_api_key:
            print(
                "Critical Error: TTS_API_KEY is required when using OpenAI TTS (tts-host=api.openai.com). "
                "Set the TTS_API_KEY environment variable or provide --tts-api-key argument. Exiting.", file=sys.stderr
            )
            return 1
    else:
        try:
            tts_port_int = int(settings.tts_port_arg)
            scheme = "http"
            settings.tts_base_url = f"{scheme}://{settings.tts_host_arg}:{tts_port_int}/v1"
            if not settings.tts_api_key:
                print(
                    f"Warning: No TTS API key provided for custom server at {settings.tts_base_url}. Assuming it's not needed.", file=sys.stderr
                )
        except (ValueError, TypeError):
            print(
                f"Critical Error: Invalid TTS port specified for custom server: '{settings.tts_port_arg}'. Cannot connect. Exiting.", file=sys.stderr
            )
            return 1

    settings.tts_acronym_preserve_set = {
        word.strip().upper()
        for word in settings.tts_acronym_preserve_list_arg.split(",")
        if word.strip()
    }
    settings.current_tts_speed = settings.tts_speed_arg

    # --- Logging Configuration using external setup function ---
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

    logger.debug(
        f"Loaded TTS_ACRONYM_PRESERVE_SET: {settings.tts_acronym_preserve_set}"
    )

    # --- Setup Chat Log Directory ---
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

    # --- Setup Temporary TTS Audio Directory ---
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
        logger.info(f"This run's TTS audio directory: {settings.tts_audio_dir}")

    except Exception as e:
        logger.error(f"Failed to create temporary TTS audio directory: {e}. TTS audio saving might fail.")
        return 1

    # --- Log Final Configuration ---
    logger.info(f"Logging level set to: {console_log_level_str}")
    logger.info(f"Application Version: {APP_VERSION}")
    logger.info(f"Application server host: {settings.host}")
    logger.info(
        f"Application server preferred port: {settings.preferred_port}"
    )
    if settings.use_llm_proxy:
        logger.info(f"Using LLM proxy at: {settings.llm_api_base}")
        if settings.llm_api_key:
            logger.info("Using LLM API key provided.")
        else:
            logger.info("No LLM API key provided.")
    else:
        logger.info(
            "Not using LLM proxy (using default LLM routing)."
        )
        if settings.llm_api_key:
            logger.info(
                "Using LLM API key provided (for direct routing)."
            )

    if settings.is_openai_stt:
        logger.info(f"Using OpenAI STT at: {settings.stt_api_base}")
        logger.info(f"Using STT model: {settings.stt_model_arg}")
        logger.info("Using STT API key provided (Required for OpenAI).")
    else:
        logger.info(f"Using Custom STT server at: {settings.stt_api_base}")
        logger.info(f"Using STT model: {settings.stt_model_arg}")
        if settings.stt_api_key:
            logger.info("Using STT API key provided.")
        else:
            logger.info("No STT API key provided (assumed optional for custom server).")

    if settings.stt_language_arg:
        logger.info(f"Using STT language: {settings.stt_language_arg}")
        settings.current_stt_language = settings.stt_language_arg
    else:
        logger.info("No STT language specified (or empty), Whisper will auto-detect.")
        settings.current_stt_language = None
    logger.info(
        f"STT Confidence Thresholds: no_speech_prob > {settings.stt_no_speech_prob_threshold}, avg_logprob < {settings.stt_avg_logprob_threshold}, min_words < {settings.stt_min_words_threshold}"
    )

    if settings.is_openai_tts:
        logger.info(f"Using OpenAI TTS at: {settings.tts_base_url}")
        logger.info(f"Using TTS model: {settings.tts_model_arg}")
        logger.info(f"Default TTS voice: {settings.tts_voice_arg}")
        logger.info(f"Initial TTS speed: {settings.current_tts_speed:.1f}")
        logger.info("Using TTS API key provided (Required for OpenAI).")
        if settings.tts_model_arg in OPENAI_TTS_PRICING:
            logger.info(
                f"OpenAI TTS pricing for '{settings.tts_model_arg}': ${OPENAI_TTS_PRICING[settings.tts_model_arg]:.2f} / 1M chars"
            )
        else:
            logger.warning(
                f"OpenAI TTS pricing not defined for model '{settings.tts_model_arg}'. Cost calculation will be $0."
            )
    else:
        logger.info(f"Using Custom TTS server at: {settings.tts_base_url}")
        logger.info(f"Using TTS model: {settings.tts_model_arg}")
        logger.info(f"Default TTS voice: {settings.tts_voice_arg}")
        logger.info(f"Initial TTS speed: {settings.current_tts_speed:.1f}")
        if settings.tts_api_key:
            logger.info("Using TTS API key provided.")
        else:
            logger.info("No TTS API key provided (assumed optional for custom server).")
    logger.debug(f"Loaded TTS_ACRONYM_PRESERVE_SET: {settings.tts_acronym_preserve_set}")

    if settings.system_message:
        logger.info(f"Loaded SYSTEM_MESSAGE: '{settings.system_message[:50]}...'")
    else:
        logger.info("No SYSTEM_MESSAGE defined.")

    # --- Populate Models and Costs ---
    if settings.use_llm_proxy:
        settings.available_models, settings.model_cost_data = get_models_and_costs_from_proxy(
            settings.llm_api_base, settings.llm_api_key
        )
    else:
        settings.available_models, settings.model_cost_data = get_models_and_costs_from_litellm()

    if not settings.available_models:
        logger.warning(
            "No models found from proxy or litellm.model_cost. Using fallback."
        )
        settings.available_models = ["fallback/unknown-model"]

    initial_model_preference = settings.llm_model_arg
    if initial_model_preference and initial_model_preference in settings.available_models:
        settings.current_llm_model = initial_model_preference
        logger.info(
            f"Using LLM model from --llm-model argument (or env default): {settings.current_llm_model}"
        )
    elif settings.available_models and settings.available_models[0] != "fallback/unknown-model":
        if initial_model_preference:
            logger.warning(
                f"LLM model '{initial_model_preference}' from --llm-model (or env default) not found in available list {settings.available_models}. Trying first available model."
            )
        settings.current_llm_model = settings.available_models[0]
        logger.info(f"Using first available model: {settings.current_llm_model}")
    elif initial_model_preference:
        settings.current_llm_model = initial_model_preference
        logger.warning(
            f"Model '{settings.current_llm_model}' from --llm-model (or env default) not found in available list, but using it as requested. Cost calculation might fail."
        )
    else:
        settings.current_llm_model = "fallback/unknown-model"
        logger.error(
            "No valid LLM models available or specified. Functionality may be impaired."
        )

    logger.info(f"Initial LLM model set to: {settings.current_llm_model}")

    # --- Client Initialization ---
    try:
        settings.tts_client = OpenAI(
            base_url=settings.tts_base_url,
            api_key=settings.tts_api_key,
        )
        if settings.is_openai_tts:
            try:
                logger.info(
                    "OpenAI TTS client initialized. API key will be validated on first use."
                )
            except AuthenticationError as e:
                logger.critical(
                    f"OpenAI API key is invalid: {e}. Please check TTS_API_KEY. Exiting."
                )
                return 1
            except Exception as e:
                logger.warning(
                    f"Could not perform initial validation of OpenAI API key: {e}"
                )
    except Exception as e:
        logger.critical(f"Failed to initialize TTS client: {e}. Exiting.")
        return 1

    try:
        settings.stt_client = OpenAI(
            base_url=settings.stt_api_base,
            api_key=settings.stt_api_key,
        )
        if settings.is_openai_stt:
            try:
                logger.info(
                    "OpenAI STT client initialized. API key will be validated on first use."
                )
            except AuthenticationError as e:
                logger.critical(
                    f"OpenAI API key is invalid: {e}. Please check STT_API_KEY. Exiting."
                )
                return 1
            except Exception as e:
                logger.warning(
                    f"Could not perform initial validation of OpenAI API key: {e}"
                )
    except Exception as e:
        logger.critical(f"Failed to initialize STT client: {e}. Exiting.")
        return 1

    # --- Populate Available Voices (Revised Logic) ---
    if settings.is_openai_tts:
        settings.available_voices_tts = OPENAI_TTS_VOICES
        logger.info(f"Using predefined OpenAI TTS voices: {settings.available_voices_tts}")
    else:
        logger.info(
            f"Querying custom TTS server ({settings.tts_base_url}) for available voices..."
        )
        settings.available_voices_tts = get_voices(settings.tts_base_url, settings.tts_api_key)
        if not settings.available_voices_tts:
            logger.warning(
                f"Could not retrieve voices from custom TTS server at {settings.tts_base_url}. TTS might fail."
            )
        else:
            logger.info(
                f"Available voices from custom TTS server: {settings.available_voices_tts}"
            )

    initial_voice_preference = settings.tts_voice_arg
    if initial_voice_preference and initial_voice_preference in settings.available_voices_tts:
        settings.current_tts_voice = initial_voice_preference
        logger.info(
            f"Using TTS voice from --tts-voice argument (or env default): {settings.current_tts_voice}"
        )
    elif settings.available_voices_tts:
        if initial_voice_preference:
            logger.warning(
                f"TTS voice '{initial_voice_preference}' from --tts-voice (or env default) not found in available voices: {settings.available_voices_tts}. Trying first available voice."
            )
        settings.current_tts_voice = settings.available_voices_tts[0]
        logger.info(f"Using first available voice instead: {settings.current_tts_voice}")
    else:
        settings.current_tts_voice = initial_voice_preference
        logger.error(
            f"No voices available from TTS engine, or specified voice '{settings.current_tts_voice}' is invalid. TTS will likely fail."
        )
        if settings.is_openai_tts and settings.current_tts_voice not in OPENAI_TTS_VOICES:
            logger.critical(
                f"Specified OpenAI voice '{settings.current_tts_voice}' is not valid. Valid options: {OPENAI_TTS_VOICES}. Exiting."
            )
            return 1

    logger.info(f"Initial TTS voice set to: {settings.current_tts_voice}")

    # --- FastAPI Setup ---
    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
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
        ),
        track_constraints={
            "echoCancellation": True,
            "noiseSuppression": {"exact": True},
            "autoGainControl": {"exact": True},
            "sampleRate": {"ideal": 24000},
            "sampleSize": {"ideal": 16},
            "channelCount": {"exact": 1},
        },
        rtc_configuration=get_twilio_turn_credentials() if get_space() else None,
        concurrency_limit=5 if get_space() else None,
        time_limit=90 if get_space() else None,
    )

    app = FastAPI()
    stream.mount(app)
    register_endpoints(app, stream)

    # --- Server and UI Launch ---
    # Use host and port from settings (which came from click parameters)
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

    settings.port = actual_port # Update settings with the actual port being used
    url = f"http://{current_host}:{actual_port}"

    def run_server():
        global uvicorn_server
        try:
            config = uvicorn.Config(
                app,
                host=current_host, # Use determined host
                port=actual_port,   # Use determined port
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

