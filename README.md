### Installation Instructions for Personal Machine

Before running the Python script on your personal machine, ensure you have the following system dependencies and Python packages installed:

**1. System Dependencies:**
   *   **PortAudio**: Required by `sounddevice`.
        *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y libportaudio2`
        *   **macOS (with Homebrew):** `brew install portaudio`
        *   **Windows:** You might need to download pre-compiled binaries or use a package manager like `Chocolatey` (`choco install portaudio`). Refer to the `python-sounddevice` documentation for details.
   *   **FFmpeg**: Required by `pydub` for various audio formats.
        *   **Debian/Ubuntu:** `sudo apt-get update && sudo apt-get install -y ffmpeg`
        *   **macOS (with Homebrew):** `brew install ffmpeg`
        *   **Windows:** Download from the official FFmpeg website and add it to your system's PATH.

**2. Python Packages:**
   Use `pip` to install these packages:
   ```bash
   pip install --upgrade cerebras_cloud_sdk faster-whisper pydub sounddevice soundfile numpy torch
   ```

**3. Cerebras API Key Environment Variable:**
   Set your Cerebras API key as an environment variable named `CEREBRAS_API_KEY`. Replace `your_api_key_here` with your actual key.
   *   **Linux/macOS:**
      ```bash
      export CEREBRAS_API_KEY='your_api_key_here'
      # Or add to ~/.bashrc or ~/.zshrc for permanent setting
      ```
   *   **Windows (Command Prompt):**
      ```cmd
      set CEREBRAS_API_KEY=your_api_key_here
      # For permanent setting, use System Properties -> Environment Variables
      ```

*NOTE: This early version is most preformative when hearing English and translating to other languages. Current theory is that the transcriber isn't as optimized for other languages as it is for English.*

*IMPORTANT: During testing, some systems take longer to close out of the program during exit. It's likely due to CUDA use taking longer to remove the model from memory, but rest assured it will clear out after a couple of minutes.*