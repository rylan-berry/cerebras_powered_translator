import os
import numpy as np
import threading
from collections import deque
import time
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from cerebras.cloud.sdk import Cerebras
import subprocess


# --- Cerebras API Key and Client Setup ---
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not cerebras_api_key:
    try:
        from google.colab import userdata
        cerebras_api_key = userdata.get("CEREBRAS_API_KEY")
    except (ImportError, KeyError):
        pass

if not cerebras_api_key:
    print("Warning: CEREBRAS_API_KEY environment variable or Colab secret not found. Translation API calls will fail.")
    client = None
else:
    client = Cerebras(api_key=cerebras_api_key)

target_lang = "Hindi"
conversation_history = [
    {
        "role": "system",
        "content": f"You are a helpful language translator assistant that must determine the language of origin and translate it to {target_lang}. You should only respond with the final translation, much like translation apps, feel free to use the reasoning space to aid in this. In case of the text already being in the {target_lang}, just repeate the message back to the user."
    }
]

# --- Faster-Whisper Model Loading ---
if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
    print("GPU is available. Using CUDA with float16 compute type.")
else:
    device = "cpu"
    compute_type = "int8"
    print("GPU is not available. Using CPU with int8 compute type.")

model_size = "base"
print(f"Loading Faster-Whisper model '{model_size}'...")
model = WhisperModel(model_size, device=device, compute_type=compute_type)
print(f"Faster-Whisper model '{model_size}' loaded successfully on {device} with {compute_type} compute type.")

# --- Global Variables for Audio Buffer and State ---
global_audio_buffer = deque()
recording_active = True
samplerate = 16000
lock = threading.Lock()

CHUNK_SIZE_SECONDS = 1
AUDIO_CHUNK_LENGTH = int(samplerate * CHUNK_SIZE_SECONDS)

TRANSLATION_WINDOW_SIZE = 20
silence_threshold = 1.5

# --- Audio Callback Function (for sounddevice) ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Sounddevice warning: {status}")

    if recording_active:
        audio_chunk = indata[:, 0].flatten().astype(np.float32)
        with lock:
            global_audio_buffer.extend(audio_chunk)

# --- Transcription Worker Function ---
def transcription_worker():
    global recording_active, conversation_history, client, target_lang, CHUNK_SIZE_SECONDS, AUDIO_CHUNK_LENGTH, silence_threshold
    print("Transcription worker started with word timestamps enabled and utterance completion logic.")
    current_user_utterance_words = []
    last_full_utterance_end_time = 0.0
    last_word_time = 0.0
    tentative_translation = ""

    while recording_active:
        audio_to_process = None
        with lock:
            if len(global_audio_buffer) >= AUDIO_CHUNK_LENGTH:
                audio_to_process = np.array(list(global_audio_buffer)[:AUDIO_CHUNK_LENGTH], dtype=np.float32)
                for _ in range(AUDIO_CHUNK_LENGTH):
                    global_audio_buffer.popleft()

        if len(current_user_utterance_words) > 0 and time.time() - last_word_time > silence_threshold:
            final_utterance = " ".join(current_user_utterance_words).strip()
            if final_utterance:
                user_message = {"role": "user", "content": final_utterance}
                conversation_history.append(user_message)
                assistant_message = {"role": "assistant", "content": tentative_translation}
                conversation_history.append(assistant_message)
                print(f"\nFINAL Original: {final_utterance}")
                print(f"FINAL Translation: {tentative_translation}\n")

                current_user_utterance_words = []
                last_full_utterance_end_time = last_word_time

                while len(conversation_history) > (1 + TRANSLATION_WINDOW_SIZE * 2):
                    conversation_history.pop(1)
                    conversation_history.pop(1)

        if audio_to_process is not None and len(audio_to_process) > 0:
            try:
                segments_generator, info = model.transcribe(
                    audio_to_process,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                segments = list(segments_generator)

                new_words_transcribed_in_chunk = False

                for segment in segments:
                    if segment.words:
                        for word_info in segment.words:
                            if word_info.end > last_full_utterance_end_time:
                                current_user_utterance_words.append(word_info.word.strip())
                                last_word_time = time.time()
                                new_words_transcribed_in_chunk = True

                if new_words_transcribed_in_chunk:
                    current_utterance = " ".join(current_user_utterance_words).strip()
                    if current_utterance:
                        print(f"\rOriginal (Partial): {current_utterance}", end='')

                        if client:
                            temp_conversation_history = list(conversation_history)
                            temp_conversation_history.append({"role": "user", "content": current_utterance})

                            chat_completion = client.chat.completions.create(
                                messages=temp_conversation_history,
                                model="llama-3.3-70b",
                                temperature=0.0
                            )
                            tentative_translation = chat_completion.choices[0].message.content
                            print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: {tentative_translation}", end='')
                        else:
                            print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: (Cerebras client not initialized)", end='')
            except Exception as e:
                print(f"\nAn error occurred during transcription or API call: {e}")
        else:
            time.sleep(0.1)
    print("\nTranscription worker stopped.")

# --- Main Execution Block ---
print("\nStarting real-time speech-to-text and translation system...")

stream = None
microphone_active = False
try:
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    print("Audio input stream started. Capturing microphone audio...")
    microphone_active = True
except Exception as e:
    print(f"Error initializing or starting sounddevice stream: {e}")
    print("This is expected in Google Colab without proper audio hardware access.")
    print("Falling back to simulated audio input.")

if not microphone_active:
    print("\nManually populating global_audio_buffer with simulated data for demonstration...")

    simulated_audio = np.concatenate([
        np.zeros(samplerate * 2, dtype=np.float32),
        np.random.rand(samplerate * 5).astype(np.float32) * 0.1
    ])

    chunk_size_samples = samplerate // 2
    for i in range(0, len(simulated_audio), chunk_size_samples):
        with lock:
            global_audio_buffer.extend(simulated_audio[i : i + chunk_size_samples])
        print(f"Added {min(chunk_size_samples, len(simulated_audio) - i)} samples to buffer. Current buffer size: {len(global_audio_buffer)}")
        time.sleep(0.25)
    print("Finished populating buffer with simulated data. Transcription and translation should now be processing.")

transcription_thread = threading.Thread(target=transcription_worker)
transcription_thread.daemon = True
transcription_thread.start()
print("Transcription worker thread initialized and started.")

print("System ready. Press Enter to stop recording and exit.")
input()

# --- Clean Up: Stop Recording and Threads ---
print("\nStopping recording and transcription processes...")
recording_active = False

if stream and stream.is_active():
    stream.stop()
    stream.close()
    print("Sounddevice stream stopped and closed.")
else:
    print("Sounddevice stream was not active or not initialized.")

if transcription_thread.is_alive():
    transcription_thread.join()
    print("Transcription worker thread joined.")
else:
    print("Transcription worker thread was not active or already stopped.")

with lock:
    global_audio_buffer.clear()
    print("Global audio buffer cleared.")

print("All real-time speech-to-text processes stopped and resources cleaned up. Exiting.")