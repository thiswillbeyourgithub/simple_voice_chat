# Simple Voice Chat

This project provides a flexible voice chat interface that connects to various Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) services.

![Screenshot](screenshot.png)

**Acknowledgement:** This project heavily relies on the fantastic [fastrtc](https://github.com/gradio-app/fastrtc) library, which simplifies real-time audio streaming over WebRTC and provided crucial examples for setting up the various supported backends, making this application possible.

## Motivation

This project aims to provide a versatile and cost-effective voice chat interface. While initially driven by the desire for alternatives to OpenAI's real-time voice API, it has evolved to offer multiple backend options, including direct integration with OpenAI's real-time services. This allows users to choose the best STT, LLM, and TTS combination for their needs, whether prioritizing cost, performance, self-hosting, or specific provider features.

## Features

*   🚀 **Multiple Backends:** The application supports three primary backend types for voice processing:
    *   **Classic Backend:** This is the most flexible option, offering a modular approach where you connect separate services for:
        *   🗣️ **STT (Speech-to-Text):** Supports API-based services like OpenAI Whisper or self-hosted engines such as [Speaches](https://github.com/speaches-ai/speaches) (which utilizes Faster Whisper).
        *   🧠 **LLM (Large Language Model):** Integrates with [LiteLLM](https://github.com/BerriAI/litellm), providing access to a vast array of models including OpenAI, Anthropic, Google, Mistral, Cohere, Azure, and local models run via services like [Ollama](https://ollama.com/), LiteLLM proxy, vLLM, and more.
        *   🔊 **TTS (Text-to-Speech):** Supports API-based services like OpenAI TTS or alternatives such as [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (which can use [KokoroTTS](https://github.com/kokorotts/)).
        *   *This backend allows for a fully local setup if desired, using local STT, LLM (e.g., via Ollama), and TTS engines.*
    *   **OpenAI Backend:** Utilizes OpenAI's real-time voice API for a streamlined, all-in-one voice interaction experience, requiring an OpenAI API key.
    *   **Gemini Backend:** Leverages Google's Gemini Live Connect API for real-time voice interactions, requiring a Google Gemini API key.
*   ⚙️ **Highly Configurable:** Adjust backend type, STT/LLM/TTS hosts, ports, models, API keys, STT confidence thresholds (classic backend), TTS voice/speed (classic backend), system messages, and more via CLI arguments or `.env` file.
*   🌐 **Web Interface:** Simple and responsive UI built with HTML, CSS, and JavaScript.
*   📊 **Cost Tracking:**
    *   **Classic Backend:** Real-time cost estimation for OpenAI LLM and TTS usage.
    *   **OpenAI Backend:** Real-time cost estimation based on token usage for the selected OpenAI real-time model.
*   ⚡ **Real-time Interaction:** Low-latency voice communication powered by [fastrtc](https://github.com/gradio-app/fastrtc) (WebRTC).
*   👂 **STT Confidence Filtering (Classic Backend):** Automatically reject low-confidence transcriptions based on configurable thresholds (no speech probability, average log probability, minimum word count).
*   🎤 **Dynamic Settings Adjustment:**
    *   **Classic Backend:** Change LLM model, TTS voice, TTS speed, and STT language on-the-fly.
    *   **OpenAI Backend:** Change STT language and output voice (if supported by the model/API) on-the-fly.
*   🔍 **Fuzzy Search:** Quickly find models and voices using fuzzy search in the UI dropdowns.
*   💬 **System Message Support:** Define a custom system message to guide the LLM's behavior.
*   📝 **Chat History Logging:** Automatically saves conversation history to timestamped JSON files.
*   🔄 **TTS Audio Replay (Classic Backend):** Replay the audio for any assistant message directly from the chat interface.
*   ⌨️ **Keyboard Shortcuts:** Control mute (M), clear chat (Ctrl+R), and toggle options (Shift+S) using keyboard shortcuts.
*   💓 **Connection Monitoring:** Uses a heartbeat mechanism to detect disconnected clients and potentially shut down the server.
*   🖥️ **Cross-Platform GUI:** Runs as a standalone desktop application using `pywebview` (default) or in a standard web browser (`--browser` flag). The application explicitly uses the QT backend for `pywebview` as the GTK backend lacks necessary WebRTC support.

## Known Issues

*   ⚠️ **Cost Calculation:** The cost calculation for the OpenAI real-time API and Gemini API is currently not functional.

## Installation


1.  Clone the repository:

    ```bash

    git clone https://github.com/thiswillbeyourgithub/simple_voice_chat

    cd simple_voice_chat

    ```

2.  Install the Python packages:

    ```bash

    uv pip install -e .

    ```

3.  (Optional) Configure services using environment variables. You can create a `.env` file based on the available options (see `--help` or `utils/env.py`).



## Usage



Run the main script using Python:


```bash
simple-voice-chat --help
```

The application will start a web server and attempt to open the interface in a dedicated window (or browser tab if `--browser` is specified).

### Running from a Python Script

You can also run the application directly from a Python script by importing and calling the `main` function from `simple_voice_chat.simple_voice_chat`. This allows you to pass arguments programmatically.

Here's an example:

```python
from simple_voice_chat.simple_voice_chat import main

if __name__ == "__main__":
    # Example arguments:
    # Replace these with your desired configuration
    args = [
        "--backend", "classic",
        "--llm-model", "gpt-4o",
        "--tts-voice", "alloy",
        "--stt-language", "en",
        "--browser",  # Launch in browser instead of pywebview GUI
        # Add other arguments as needed, like:
        # "--openai-api-key", "YOUR_OPENAI_KEY_HERE", # If using OpenAI backend
        # "--llm-api-key", "YOUR_LLM_KEY_HERE",    # If classic backend needs a key for LLM
        # "--stt-api-key", "YOUR_STT_KEY_HERE",    # If classic backend STT needs a key
        # "--tts-api-key", "YOUR_TTS_KEY_HERE",    # If classic backend TTS needs a key
    ]
    
    # The main function expects a list of strings, similar to sys.argv
    # It's decorated with @click.command(), so we call it with .main(args)
    # or by directly invoking it if click handles parsing internally when called this way.
    # For programmatic invocation with click, it's often easier to let click parse:
    import os
    # To ensure LiteLLM runs in production mode if not already set by the main script early enough
    os.environ['LITELLM_MODE'] = 'PRODUCTION' 
    
    # Call the click command directly
    # Note: click commands usually expect to be called as if from the command line.
    # To pass arguments programmatically to a click command, you typically invoke `main.main(args=args_list, standalone_mode=False)`.
    # However, since `main` is already a click command, we can try to directly invoke it.
    # If `main()` is defined as `def main(): @click.pass_context def cli(ctx, ...)` then `main(args)` works.
    # If `def main(...)` is the click command itself, it consumes args from sys.argv by default.
    # The `main` function in simple_voice_chat.py is a click command itself: `@click.command(...) def main(...)`
    # So, to run it programmatically as if from CLI:
    try:
        # sys.argv needs to be manipulated if click is to parse it automatically,
        # or use the programmatic API if available.
        # The simplest way with click
        main.main(args=args, standalone_mode=False) 
    except SystemExit as e:
        # Click commands often call sys.exit(). We can catch this if running in a script.
        print(f"Application exited with status: {e.code}")

```

When calling programmatically, `main.main(args=your_list_of_args, standalone_mode=False)` is the recommended way to invoke a Click command and pass arguments. The `standalone_mode=False` flag prevents Click from trying to exit the entire Python interpreter.

You can find all available command-line arguments and their corresponding environment variables by running `simple-voice-chat --help`.

You can choose the backend using the `--backend` option:
*   `--backend classic` (default): Uses separate STT, LLM, and TTS services.
*   `--backend openai`: Uses OpenAI's real-time voice API. Requires `--openai-api-key`.

**For a detailed list of all configuration options, please use the `--help` flag:**

```bash
simple-voice-chat --help
```

This will provide the most up-to-date information on available arguments and their corresponding environment variables, including options specific to each backend.

---



*This README was generated with assistance from [aider.chat](https://aider.chat).*
