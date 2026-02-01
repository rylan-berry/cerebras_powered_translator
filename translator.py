import os
import numpy as np
import threading
from collections import deque
import time
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from cerebras.cloud.sdk import Cerebras

# --- Cerebras API Key and Client Setup ---
# Retrieve API key from environment variable for personal machine
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not cerebras_api_key:
    raise ValueError("CEREBRAS_API_KEY environment variable not set. Please set it before running.")

client = Cerebras(api_key=cerebras_api_key)
target_lang = "English"

# Initialize the conversation history with the system message
conversation_history = [
    {
        "role": "system",
        "content": f"You are a helpful language translator assistant that must determine the language of origin and translate it to {target_lang}. You should only respond with the final translation, much like translation apps, feel free to use the reasoning space to aid in this. In case of the text already being in the {target_lang}, just repeate the message back to the user."
    }
]

# --- Faster-Whisper Model Loading ---
# Check for GPU availability and set compute type
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"  # Use float16 for GPU for faster inference
    print("GPU is available. Using CUDA with float16 compute type.")
else:
    device = "cpu"
    compute_type = "int8"  # Use int8 for CPU for optimized performance
    print("GPU is not available. Using CPU with int8 compute type.")

# Define the desired model size
model_size = "base"

# Instantiate the WhisperModel
print(f"Loading Faster-Whisper model '{model_size}'...")
model = WhisperModel(model_size, device=device, compute_type=compute_type)
print(f"Faster-Whisper model '{model_size}' loaded successfully on {device} with {compute_type} compute type.")

# --- Global Variables for Audio Buffer and State ---
global_audio_buffer = deque() # Use deque for efficient appends and pops
recording_active = True # Controls both audio capture and transcription loops
samplerate = 16000 # Matching faster-whisper model's expected sample rate
lock = threading.Lock() # Protects access to the global_audio_buffer

# Define transcription settings
CHUNK_SIZE_SECONDS = 1 # Process audio in 1-second chunks
AUDIO_CHUNK_LENGTH = int(samplerate * CHUNK_SIZE_SECONDS)

# Define sliding window size for conversation history
TRANSLATION_WINDOW_SIZE = 20 # Keep last 20 user/assistant message pairs

# --- Audio Callback Function (for sounddevice) ---
def audio_callback(indata, frames, time_info, status):
    """
    This function is called by sounddevice for each audio chunk.
    It processes the incoming audio data and adds it to the global buffer.
    """
    if status:
        print(f"Sounddevice warning: {status}")

    if recording_active:
        # Convert indata to a 1D float32 array, assuming mono input (first channel)
        audio_chunk = indata[:, 0].flatten().astype(np.float32)
        with lock: # Acquire lock before modifying global_audio_buffer
            global_audio_buffer.extend(audio_chunk)

# --- Transcription Worker Function ---
def transcription_worker():
    global recording_active, conversation_history, client, target_lang
    print("Transcription worker started.")
    while recording_active:
        audio_to_process = None
        with lock:
            # Check if there's enough audio for a full chunk
            if len(global_audio_buffer) >= AUDIO_CHUNK_LENGTH:
                audio_to_process = np.array(list(global_audio_buffer)[:AUDIO_CHUNK_LENGTH], dtype=np.float32)
                # Clear the consumed portion from the buffer
                for _ in range(AUDIO_CHUNK_LENGTH):
                    global_audio_buffer.popleft()

        if audio_to_process is not None and len(audio_to_process) > 0:
            try:
                # Transcribe the audio segment using the loaded faster-whisper model
                segments, info = model.transcribe(audio_to_process, beam_size=5)
                transcribed_text = " ".join([segment.text for segment in segments])

                if transcribed_text.strip(): # Only process if meaningful text is transcribed
                    print(f"Original Transcription: {transcribed_text.strip()}")

                    # 1. Append the transcribed text as a 'user' message to conversation_history
                    user_message = {"role": "user", "content": transcribed_text.strip()}
                    conversation_history.append(user_message)

                    # 2. Make an API call to client.chat.completions.create
                    chat_completion = client.chat.completions.create(
                        messages=conversation_history,
                        model="llama-3.3-70b",
                    )

                    # 3. Extract and append the translated response as an 'assistant' message
                    assistant_response = chat_completion.choices[0].message.content
                    conversation_history.append({"role": "assistant", "content": assistant_response})

                    # 4. Print the translated response
                    print(f"Translation: {assistant_response}")

                    # 5. Implement the sliding window logic
                    # The system message is at index 0. Each user/assistant pair adds 2 messages.
                    # We want to keep 1 system message + TRANSLATION_WINDOW_SIZE pairs = 1 + 2*TRANSLATION_WINDOW_SIZE messages
                    while len(conversation_history) > (1 + TRANSLATION_WINDOW_SIZE * 2):
                        # Remove the oldest user message (index 1) and its corresponding assistant message (index 1 after user message removed)
                        conversation_history.pop(1) # Remove oldest user message
                        conversation_history.pop(1) # Remove oldest assistant message

            except Exception as e:
                print(f"An error occurred during translation or API call: {e}")
                # Remove the last user message if the API call fails to keep history consistent
                if len(conversation_history) > 1 and conversation_history[-1]["role"] == "user":
                    conversation_history.pop() # Remove the user message that caused the error
        else:
            # Small delay to prevent the loop from busy-waiting if the buffer is empty or not full enough
            time.sleep(0.1)
    print("Transcription worker stopped.")

# --- Main Execution Block ---
print("Starting real-time speech-to-text and translation system...")

# Initialize and Start sounddevice InputStream
stream = None # Initialize stream to None
try:
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    print("Audio input stream started. Capturing microphone audio...")
except Exception as e:
    print(f"Error initializing or starting sounddevice stream: {e}")
    print("Please ensure your microphone is connected and PortAudio is correctly installed.")
    # If stream fails, set recording_active to False to prevent transcription_worker from running indefinitely
    recording_active = False

# Start the Transcription Worker Thread
transcription_thread = threading.Thread(target=transcription_worker)
transcription_thread.daemon = True # Allows the program to exit even if this thread is running
transcription_thread.start()
print("Transcription worker thread initialized and started.")

print("System ready. Press Enter to stop recording and exit.")
# Keep the main thread alive, waiting for user input to stop
input()

# --- Clean Up: Stop Recording and Threads ---
print("Stopping recording and transcription processes...")
recording_active = False # Signal both the sounddevice callback and transcription worker to stop

if stream and stream.is_active():
    stream.stop()
    stream.close()
    print("Sounddevice stream stopped and closed.")

# Ensure the transcription thread has finished before proceeding
if transcription_thread.is_alive():
    transcription_thread.join() # Wait for the transcription thread to finish gracefully
    print("Transcription worker thread joined.")

with lock:
    global_audio_buffer.clear()
    print("Global audio buffer cleared.")

print("All real-time speech-to-text processes stopped and resources cleaned up. Exiting.")