import setuptools
from pathlib import Path

# --- Read README for Long Description ---
def _read_readme(filename="README.md"):
    """Reads the README file for the long description."""
    readme_path = Path(__file__).parent / filename
    try:
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Long description will be empty.")
        return "" # CHANGEME: Consider adding a default description if README is missing


setuptools.setup(
    name="simple-voice-chat", # CHANGEME: Verify or change the package name
    version="4.0.1",
    author="thiswillbeyourgithub",
    description="A simple voice chat interface using configurable LLM, STT, and TTS providers.",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thiswillbeyourgithub/simple_voice_chat",
    license="GPLv3",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"] # Exclude test directories if any
    ),

    # Include non-code files specified in MANIFEST.in (if you create one)
    # Or use include_package_data=True and package_data dictionary
    include_package_data=True,
    package_data={
        # If 'simple_voice_chat' contains non-python files needed at runtime
        # Example: 'simple_voice_chat': ['index.html', 'static/*']
        'simple_voice_chat': ['index.html'], # Include index.html
        # Add other necessary non-code files here
    },

    # Dependencies are now listed directly here
    install_requires=[
        "click>=8.0", # Added click
        "qtpy>=2.4.3",
        "filelock>=3.18.0",
        "fastrtc[vad,tts]>=0.0.23",
        "openai>=1.76.0",
        "twilio>=9.5.2",
        "python-dotenv>=1.1.0",
        "pywebview>=5.4",
        "PyQt6>=6.9.0",
        "PyQt6-WebEngine>=6.9.0",
        "psutil>=7.0.0",
        "numpy>=2.2.5",
        "fastapi>=0.115.12",
        "uvicorn[standard]>=0.34.2",
        "litellm>=1.67.2",
        "loguru>=0.7.3",
        "platformdirs>=4.3.7",
        "google-genai >= 1.14.0",
    ],

    # Define entry points, e.g., console scripts
    entry_points={
        "console_scripts": [
            # Correctly point to the main function within the simple_voice_chat module
            "simple-voice-chat=simple_voice_chat.simple_voice_chat:main",
        ],
    },

    python_requires=">=3.9",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Communications :: Chat",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Add your license classifier here, e.g.:
        # "License :: OSI Approved :: MIT License",
    ],

    keywords="voice chat, llm, stt, tts, ai, chatbot, fastrtc, openai",
    project_urls={
        "Bug Reports": "https://github.com/thiswillbeyourgithub/simple_voice_chat/issues",
        "Source": "https://github.com/thiswillbeyourgithub/simple_voice_chat",
    },
)
