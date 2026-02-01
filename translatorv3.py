import subprocess
import os
from cerebras.cloud.sdk import Cerebras
import numpy as np
import threading
from collections import deque
import time
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import soundfile as sf
import sys

# --- Cerebras API Key and Client Setup ---
cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
if not cerebras_api_key:
    try:
        from google.colab import userdata
        cerebras_api_key = userdata.get("CEREBRAS_API_KEY")
    except (ImportError, Exception):
        pass

if not cerebras_api_key:
    print("Warning: CEREBRAS_API_KEY environment variable not found. Translation API calls will fail.")
    client = None
else:
    client = Cerebras(api_key=cerebras_api_key)
    print("Cerebras client initialized.")

target_lang = input("Target Language: ")
if len(target_lang) == 0:
    target_lang = "English" #default to english


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
conversation_history_lock = threading.Lock()

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
        if len(audio_chunk) > 0:
            with lock:
                global_audio_buffer.extend(audio_chunk)
            # print(f"DBG: Callback added {len(audio_chunk)} samples. Current buf len: {len(global_audio_buffer)}")


# --- Transcription Worker Function (Modified for word-level and dynamic translation) ---
def transcription_worker():
    global recording_active, conversation_history, client, target_lang, CHUNK_SIZE_SECONDS, AUDIO_CHUNK_LENGTH, silence_threshold, conversation_history_lock
    print("Transcription worker started with word timestamps enabled and utterance completion logic.")
    current_user_utterance_words = []
    # last_full_utterance_end_time = 0.0 # Removed: This variable was causing incorrect filtering
    last_word_time = time.time()
    tentative_translation = ""
    last_api_call_time = 0.0

    try:
        while recording_active or len(global_audio_buffer) > 0:
            # print(f"\nDBG: Top loop. Buf: {len(global_audio_buffer)}")

            audio_to_process = None
            current_buffer_length = 0
            with lock:
                current_buffer_length = len(global_audio_buffer)

            if current_buffer_length >= AUDIO_CHUNK_LENGTH:
                with lock:
                    audio_to_process = np.array(list(global_audio_buffer)[:AUDIO_CHUNK_LENGTH], dtype=np.float32)
                    for _ in range(AUDIO_CHUNK_LENGTH):
                        global_audio_buffer.popleft()
                # print(f"DBG: Proc full chunk. Rem Buf: {len(global_audio_buffer)}")
            elif not recording_active and current_buffer_length > 0:
                with lock:
                    audio_to_process = np.array(list(global_audio_buffer), dtype=np.float32)
                    global_audio_buffer.clear()
                # print(f"DBG: Proc remaining audio ({len(audio_to_process)} samples) during shutdown. Buf cleared.")
            else:
                time.sleep(0.05)
                continue

            # print(f"DBG: Utterance check. Words: {len(current_user_utterance_words)}, Silence: {time.time() - last_word_time:.2f}s")
            if len(current_user_utterance_words) > 0 and (time.time() - last_word_time) > silence_threshold:
                final_utterance = " ".join(current_user_utterance_words).strip()
                if final_utterance:
                    # print(f"DBG: Utterance complete: '{final_utterance}'")
                    if client and not tentative_translation:
                        # print("DBG: Getting final translation.")
                        with conversation_history_lock:
                            temp_conversation_history = list(conversation_history)
                        temp_conversation_history.append({"role": "user", "content": final_utterance})
                        try:
                            chat_completion = client.chat.completions.create(
                                messages=temp_conversation_history,
                                model="llama-3.3-70b",
                                temperature=0.0
                            )
                            tentative_translation = chat_completion.choices[0].message.content
                            # print("DBG: Final translation obtained.")
                        except Exception as e:
                            print(f"\nError getting final translation: {e}")
                            tentative_translation = f"Translation failed: {e}"
                    elif not client:
                        # print("DBG: Cerebras client not init for final trans.")
                        tentative_translation = "(Cerebras client not initialized)"

                    with conversation_history_lock:
                        user_message = {"role": "user", "content": final_utterance}
                        conversation_history.append(user_message)
                        assistant_message = {"role": "assistant", "content": tentative_translation}
                        conversation_history.append(assistant_message)
                        print(f"\nFINAL Original: {final_utterance}")
                        print(f"FINAL Translation: {tentative_translation}\n")

                        current_user_utterance_words = []
                        # last_full_utterance_end_time = last_word_time # Removed: Not needed for filtering words from new chunks
                        tentative_translation = ""

                        while len(conversation_history) > (1 + TRANSLATION_WINDOW_SIZE * 2):
                            conversation_history.pop(1)
                            conversation_history.pop(1)
                        # print(f"DBG: Conv history updated. Len: {len(conversation_history)}")

            if audio_to_process is not None and len(audio_to_process) > 0:
                # print("DBG: Transcribing segment.")
                # print(f"DBG: Audio for trans: {len(audio_to_process)} samples.")
                try:
                    segments_generator, info = model.transcribe(
                        audio_to_process,
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    segments = list(segments_generator)
                    # print(f"DBG: Transcribed {len(segments)} segments.")

                    new_words_transcribed_in_chunk = False

                    for segment in segments:
                        if segment.words:
                            for word_info in segment.words:
                                # Removed condition: if word_info.end > last_full_utterance_end_time:
                                # The current_user_utterance_words list is cleared when an utterance is finalized,
                                # so new words from a new chunk should always be appended.
                                word_text = word_info.word.strip()
                                if word_text:
                                    current_user_utterance_words.append(word_text)
                                    last_word_time = time.time()
                                    new_words_transcribed_in_chunk = True
                    if new_words_transcribed_in_chunk:
                        # print("DBG: New words transcribed.")
                        pass
                    # else:
                    # print("DBG: No new words this chunk.")

                    if new_words_transcribed_in_chunk:
                        current_utterance = " ".join(current_user_utterance_words).strip()
                        if current_utterance:
                            # print(f"DBG: Partial: '{current_utterance}'")
                            if client and (time.time() - last_api_call_time > 0.5):
                                # print("DBG: Calling API for tentative trans.")
                                with conversation_history_lock:
                                    temp_conversation_history = list(conversation_history)
                                temp_conversation_history.append({"role": "user", "content": current_utterance})

                                try:
                                    chat_completion = client.chat.completions.create(
                                        messages=temp_conversation_history,
                                        model="llama-3.3-70b",
                                        temperature=0.0
                                    )
                                    tentative_translation = chat_completion.choices[0].message.content
                                    print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: {tentative_translation}", end='')
                                    last_api_call_time = time.time()
                                    # print("DBG: Tentative trans updated.")
                                except Exception as e:
                                    print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: (API Error: {e})", end='')
                                    print(f"Error during tentative translation API call: {e}")
                            elif not client:
                                 print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: (Cerebras client not initialized)", end='')
                                 # print("DBG: API skipped: client not init.")
                            elif client and (time.time() - last_api_call_time <= 0.5):
                                print(f"\rOriginal (Partial): {current_utterance}\nTentative Translation: {tentative_translation}", end='')
                                # print("DBG: API skipped: cooldown active.")


                except Exception as e:
                    print(f"\nAn error occurred during transcription or API call: {e}")
            else:
                time.sleep(0.05)
                # print("DBG: No audio to process, sleeping.")
    except Exception as e:
        print(f"\nCRITICAL: Transcription worker thread terminated due to unhandled exception: {e}")
    finally:
        print("\nTranscription worker stopped.")


# --- Main Execution Block ---
print("Starting real-time speech-to-text and translation system...")

# Initialize and Start sounddevice InputStream
stream = None
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
    print("This is expected in Google Colab without proper audio hardware access. Attempting simulated audio.")

    print("Manually populating global_audio_buffer with simulated data for demonstration...")

    audio_file_path = '/content/sample_audio.mp3'
    if not os.path.exists(audio_file_path):
        print(f"Warning: {audio_file_path} not found. Falling back to random noise.")
        simulated_audio = np.concatenate([
            np.zeros(samplerate * 2, dtype=np.float32),
            np.random.rand(samplerate * 5).astype(np.float32) * 0.1
        ])
    else:
        try:
            print(f"Loading audio from {audio_file_path}...")
            simulated_audio, file_samplerate = sf.read(audio_file_path, dtype='float32')
            if file_samplerate != samplerate:
                print(f"Resampling audio from {file_samplerate}Hz to {samplerate}Hz...")
                duration = len(simulated_audio) / file_samplerate
                num_samples_resampled = int(duration * samplerate)
                simulated_audio = np.interp(
                    np.arange(0, num_samples_resampled),
                    np.arange(0, len(simulated_audio)) * (num_samples_resampled / len(simulated_audio)),
                    simulated_audio
                ).astype(np.float32)

            if simulated_audio.ndim > 1:
                simulated_audio = simulated_audio.mean(axis=1)
            print(f"Loaded {len(simulated_audio)} samples from {audio_file_path}.")

        except Exception as e:
            print(f"Error loading and processing audio file {audio_file_path}: {e}")
            print("Falling back to random noise simulation.")
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


# Start the Transcription Worker Thread
transcription_thread = threading.Thread(target=transcription_worker)
transcription_thread.daemon = True
transcription_thread.start()
print("Transcription worker thread initialized and started.")

print("System ready. Press Enter to stop recording and exit.")
try:
    input()
except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected. Shutting down.")
    pass

# --- Clean Up: Stop Recording and Threads ---
print("Stopping recording and transcription processes...")
recording_active = False

if stream and stream.active:
    stream.stop()
    stream.close()
    stream = None
    print("Sounddevice stream stopped and closed.")

if transcription_thread.is_alive():
    transcription_thread.join()
    print("Transcription worker thread joined.")

with lock:
    global_audio_buffer.clear()
    print("Global audio buffer cleared.")

# Explicitly dereference the model to aid in faster resource cleanup
model = None
client = None

print("All real-time speech-to-text processes stopped and resources cleaned up. Exiting.")
sys.exit(0) # Ensure a clean exit