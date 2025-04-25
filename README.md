# Simple Voice Chat

This project provides a flexible voice chat interface that connects to various Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) services.

![Screenshot](screenshot.png)

**Acknowledgement:** This project heavily relies on the fantastic [fastrtc](https://github.com/gradio-app/fastrtc) library, which simplifies real-time audio streaming over WebRTC, making this application possible.

## Motivation

The primary motivation for creating this project was the high cost associated with OpenAI's real-time voice API. This application allows you to leverage potentially more cost-effective or self-hosted alternatives for STT, LLM, and TTS, while still providing a near real-time voice interaction experience.


## Features

*   **Modular:** Connect to different STT, LLM (including local models via proxies like LiteLLM), and TTS providers.
    *   **STT:** Defaults to using [Speaches](https://github.com/speaches-ai/speaches) (which allows self-hosting Faster Whisper), but can also connect directly to OpenAI's Whisper API.
    *   **TTS:** Defaults to OpenAI TTS, but also supports alternatives like [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI).
    *   **LLM:** Supports virtually any LLM provider (OpenAI, Anthropic, Google, Mistral, Cohere, Azure, local models, etc.) thanks to its integration with [LiteLLM](https://github.com/BerriAI/litellm/). You can connect to any OpenAI-compatible API endpoint, including local models served via proxies like LiteLLM itself, vLLM, or Ollama.

*   **Configurable:** Fine-tune various parameters for STT confidence, TTS voice/speed, LLM model selection, and more via command-line arguments or environment variables.

*   **Web Interface:** Provides a simple web-based UI for interaction.

*   **Cost Tracking:** Includes basic cost estimation for supported LLM and TTS providers (like OpenAI).


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



**For a detailed list of all configuration options (STT/LLM/TTS hosts, ports, models, API keys, etc.), please use the `--help` flag:**


```bash
simple-voice-chat --help
```




This will provide the most up-to-date information on available arguments and their
corresponding environment variables.

<details>
<summary>Command-Line Options (--help)</summary>
<pre><code>
usage: simple_voice_chat.py [-h] [--host HOST] [--port PORT] [-v]
                               [--auto-start | --no-auto-start] [--browser]
                               [--system-message SYSTEM_MESSAGE]
                               [--llm-host LLM_HOST] [--llm-port LLM_PORT]
                               [--llm-model LLM_MODEL]
                               [--llm-api-key LLM_API_KEY]
                               [--stt-host STT_HOST] [--stt-port STT_PORT]
                               [--stt-model STT_MODEL]
                               [--stt-language STT_LANGUAGE]
                               [--stt-api-key STT_API_KEY]
                               [--stt-no-speech-prob-threshold STT_NO_SPEECH_PROB_THRESHOLD]
                               [--stt-avg-logprob-threshold STT_AVG_LOGPROB_THRESHOLD]
                               [--stt-min-words-threshold STT_MIN_WORDS_THRESHOLD]
                               [--tts-host TTS_HOST] [--tts-port TTS_PORT]
                               [--tts-model TTS_MODEL] [--tts-voice TTS_VOICE]
                               [--tts-api-key TTS_API_KEY]
                               [--tts-speed TTS_SPEED]
                               [--tts-acronym-preserve-list TTS_ACRONYM_PRESERVE_LIST]

Run a simple voice chat interface using a configurable LLM provider, STT
server, and TTS.

options:
  -h, --help            show this help message and exit
  --host HOST           Host address to bind the FastAPI server to. Default:
                        127.0.0.1
  --port PORT           Preferred port to run the FastAPI server on. Default:
                        7860. (Env: APP_PORT)
  -v, --verbose         Enable verbose logging (DEBUG level)
  --auto-start, --no-auto-start
                        Automatically start the connection when the
                        application loads. Default: True
  --browser             Launch the application in the default web browser
                        instead of a dedicated GUI window. Default: False
  --system-message SYSTEM_MESSAGE
                        System message to prepend to the chat history.
                        Default: (from SYSTEM_MESSAGE env var, empty if
                        unset).
  --llm-host LLM_HOST   Host address of the LLM proxy server (optional).
                        Default: None. (Env: LLM_HOST)
  --llm-port LLM_PORT   Port of the LLM proxy server (optional). Default:
                        None. (Env: LLM_PORT)
  --llm-model LLM_MODEL
                        Default LLM model to use (e.g., 'gpt-4o',
                        'litellm_proxy/claude-3-opus'). Default:
                        'litellm_proxy/litellm_best'. (Env: LLM_MODEL)
  --llm-api-key LLM_API_KEY
                        API key for the LLM provider/proxy (optional, depends
                        on setup). Default: None. (Env: LLM_API_KEY)
  --stt-host STT_HOST   Host address of the STT server (e.g., 'api.openai.com'
                        or 'localhost'). Default: 'localhost'. (Env: STT_HOST)
  --stt-port STT_PORT   Port of the STT server (e.g., 443 for OpenAI, 8002 for
                        local). Default: '8002'. (Env: STT_PORT)
  --stt-model STT_MODEL
                        STT model to use (e.g., 'whisper-1' for OpenAI,
                        'deepdml/faster-whisper-large-v3-turbo-ct2' for
                        local). Default: 'deepdml/faster-whisper-
                        large-v3-turbo-ct2'. (Env: STT_MODEL)
  --stt-language STT_LANGUAGE
                        Language code for STT (e.g., 'en', 'fr'). If unset,
                        Whisper usually auto-detects. Default: None. (Env:
                        STT_LANGUAGE)
  --stt-api-key STT_API_KEY
                        API key for the STT server (REQUIRED for OpenAI STT).
                        Default: None. (Env: STT_API_KEY)
  --stt-no-speech-prob-threshold STT_NO_SPEECH_PROB_THRESHOLD
                        STT confidence threshold: Reject if no_speech_prob is
                        higher than this. Default: 0.6. (Env:
                        STT_NO_SPEECH_PROB_THRESHOLD)
  --stt-avg-logprob-threshold STT_AVG_LOGPROB_THRESHOLD
                        STT confidence threshold: Reject if avg_logprob is
                        lower than this. Default: -0.7. (Env:
                        STT_AVG_LOGPROB_THRESHOLD)
  --stt-min-words-threshold STT_MIN_WORDS_THRESHOLD
                        STT confidence threshold: Reject if the number of
                        words is less than this. Default: 5. (Env:
                        STT_MIN_WORDS_THRESHOLD)
  --tts-host TTS_HOST   Host address of the TTS server (e.g., 'api.openai.com'
                        or 'localhost'). Default: 'api.openai.com'. (Env:
                        TTS_HOST)
  --tts-port TTS_PORT   Port of the TTS server (e.g., 443 for OpenAI, 8880 for
                        local). Default: '443'. (Env: TTS_PORT)
  --tts-model TTS_MODEL
                        TTS model to use (e.g., 'tts-1', 'tts-1-hd' for
                        OpenAI, 'kokoro' for local). Default: 'tts-1'. (Env:
                        TTS_MODEL)
  --tts-voice TTS_VOICE
                        Default TTS voice to use (e.g., 'alloy', 'ash', 'echo'
                        for OpenAI, 'ff_siwis' for local). Default: 'nova'.
                        (Env: TTS_VOICE)
  --tts-api-key TTS_API_KEY
                        API key for the TTS server (REQUIRED for OpenAI TTS).
                        Default: None. (Env: TTS_API_KEY)
  --tts-speed TTS_SPEED
                        Default TTS speed multiplier. Default: 1.00. (Env:
                        TTS_SPEED)
  --tts-acronym-preserve-list TTS_ACRONYM_PRESERVE_LIST
                        Comma-separated list of acronyms to preserve during
                        TTS (currently only used for Kokoro TTS). Default: ''.
                        (Env: TTS_ACRONYM_PRESERVE_LIST)
</code></pre>
</details>

---



*This README was generated with assistance from [aider.chat](https://aider.chat).*
