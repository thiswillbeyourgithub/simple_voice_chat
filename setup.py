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

# --- Package Version (from main script) ---
# Attempt to read version from the main script to keep it DRY
# CHANGEME: Verify this path and variable name are correct
VERSION = "1.0.0" # Default fallback
try:
    # Read version from the main script within the package
    main_script_path = Path(__file__).parent / "simple_voice_chat" / "simple_voice_chat.py"
    with open(main_script_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("APP_VERSION ="):
                # Extract version string (handles quotes)
                VERSION = line.split("=")[1].strip().strip('"').strip("'")
                break
except Exception as e:
    print(f"Warning: Could not read version from main script ({e}). Using default: {VERSION}")


setuptools.setup(
    name="simple-voice-chat", # CHANGEME: Verify or change the package name
    version=VERSION,
    author="CHANGEME: Your Name or Organization",
    author_email="CHANGEME: your.email@example.com",
    description="A simple voice chat interface using configurable LLM, STT, and TTS providers.", # CHANGEME: Improve description if needed
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    url="CHANGEME: https://github.com/your_username/your_repo", # CHANGEME: Add project URL
    license="CHANGEME: Specify License (e.g., MIT, Apache 2.0)", # CHANGEME: Add license

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
        "qtpy",
        "filelock",
        "fastrtc[vad,tts]>=0.0.20.rc2",
        "openai",
        "twilio",
        "python-dotenv",
        "pywebview", # Removed [qt] extra
        "PyQt6", # Added explicit Qt binding
        "PyQt6-WebEngine", # Added explicit Qt WebEngine binding
        "psutil",
        "numpy",
        "fastapi",
        "uvicorn[standard]",
        "litellm",
        "loguru", # Already present, ensuring it's here
    ],

    # Define entry points, e.g., console scripts
    entry_points={
        "console_scripts": [
            # Correctly point to the main function within the simple_voice_chat module
            "simple-voice-chat=simple_voice_chat.simple_voice_chat:main",
        ],
    },

    python_requires=">=3.9", # CHANGEME: Specify the minimum Python version required

    classifiers=[
        # CHANGEME: Choose appropriate classifiers from https://pypi.org/classifiers/
        "Development Status :: 4 - Beta", # Example
        "Intended Audience :: Developers", # Example
        "Intended Audience :: End Users/Desktop", # Example
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent", # Or specify OS if needed
        "Topic :: Communications :: Chat",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Add your license classifier here, e.g.:
        # "License :: OSI Approved :: MIT License",
    ],

    keywords="voice chat, llm, stt, tts, ai, chatbot, fastrtc, openai", # CHANGEME: Add relevant keywords
    project_urls={ # CHANGEME: Add relevant project links
        "Bug Reports": "CHANGEME: https://github.com/your_username/your_repo/issues",
        "Source": "CHANGEME: https://github.com/your_username/your_repo/",
    },
)

print("\n--- setup.py finished ---")
print("Remember to review and update all 'CHANGEME' placeholders.")
# Note: The console_scripts entry point now correctly points to the main() function.
# The application can also be run using `python -m simple_voice_chat`.

