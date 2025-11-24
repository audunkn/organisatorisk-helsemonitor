import pandas as pd
from datasets import load_dataset, Audio
import os
import shutil 
from collections import defaultdict
from itertools import islice

# --- NEW: Google Gemini/API integration ---
import google.generativeai as genai
import time
from dotenv import load_dotenv

# --- Configuration ---
DATASET_NAME = "distil-whisper/earnings22"
SPLIT = "test" 
CONFIG_NAME = "chunked"
OUTPUT_DIR = "full_transcripts_output"

# Set this to control how many unique, full-length transcripts are reconstructed
NUM_CALLS_TO_PROCESS = 2 

# --- Progress Bar Configuration ---
PROGRESS_UPDATE_INTERVAL = 500 # Print status every X segments processed

# Suppress benign warnings/errors from datasets library on Windows
import datasets
datasets.logging.set_verbosity_error()

# --- Gemini Setup (FROM YOUR SECOND FILE) ---
load_dotenv() 

API_KEY = os.getenv("GEMINI_API_KEY") 
if not API_KEY:
    raise ValueError("GEMINI_API_KEY er ikke funnet. Vennligst sjekk .env-filen eller miljøvariablene.")

try:
    genai.configure(api_key=API_KEY)
    GEMINI_MODEL_NAME = 'gemini-2.5-flash' 
except Exception as e:
    print(f"Feil ved konfigurering av Gemini API: {e}")
    exit()

# --- Hjelpefunksjoner ---

def generate_content_with_retry(model, prompt, max_retries=5, initial_wait=30):
    """
    Prøver å kalle Gemini API-et. Hvis vi treffer Rate Limit (429),
    venter vi og prøver igjen.
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if response and response.text:
                return response
            elif response and response.prompt_feedback.block_reason:
                print(f"❌ API-forespørsel blokkert: {response.prompt_feedback.block_reason}")
                return None
            else:
                print(f"❌ Tomt svar fra API-et. Prøver igjen.")
                time.sleep(2) 
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = initial_wait * (attempt + 1)
                print(f"⚠️ Traff Rate Limit (429). Venter {wait_time} sekunder før nytt forsøk ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"❌ Uventet feil fra API: {e}. Prøver på nytt.")
                time.sleep(5)
    
    print("❌ Ga opp etter maksimale gjentakelser.")
    return None


def translate_to_norwegian(text: str) -> str:
    """
    Kaller Gemini LLM for å oversette teksten til profesjonell norsk, 
    og bruker retry-logikken.
    """
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    prompt = f"Oversett følgende transkripsjon av en earnings call nøyaktig til profesjonelt norsk. Behold alle tall, navn og tekniske termer som de er. Her er transkripsjonen:\n\n---\n{text}\n---"
    
    print("\n[LLM TRANSLATION] Kaller Gemini for oversettelse til norsk...")
    
    response = generate_content_with_retry(model, prompt)
    
    if response and response.text:
        return response.text
    
    return "TRANSLATION FAILED: Klarte ikke å hente en oversettelse fra API-et."


def clear_output_directory():
    """Removes the output directory and all its contents, then recreates it."""
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Clearing existing directory: '{OUTPUT_DIR}'...")
        try:
            shutil.rmtree(OUTPUT_DIR)
            print("   Directory successfully cleared.")
        except Exception as e:
            print(f"Error clearing directory: {e}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"   Recreated directory: '{OUTPUT_DIR}'")

def save_transcript_to_file(transcript: str, filename: str, directory: str = OUTPUT_DIR):
    """Saves the reconstructed or translated transcript to a plain text file."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"-> Successfully saved to '{filepath}'.")
    except Exception as e:
        print(f"Error saving file: {e}")


# =================================================================
# Modified: Core function 'explore_dataset'
# =================================================================
def explore_dataset(dataset_name: str, split: str, config_name: str, num_calls: int):
    """
    Loads, reconstructs, saves, translates, and saves the translated transcripts.
    """
    # 0. Clean up before starting
    clear_output_directory() 

    print(f"--- Loading Dataset: {dataset_name}, Config: {config_name}, Split: {split} (Streaming Mode) ---")
    
    try:
        dataset_stream = load_dataset(dataset_name, config_name, split=split, streaming=True)
        
        if 'audio' in dataset_stream.column_names:
            dataset_stream = dataset_stream.cast_column("audio", Audio(decode=False))
            dataset_stream = dataset_stream.remove_columns(['audio'])

        call_data = defaultdict(lambda: {'segments': []})
        processed_call_ids = set()
        
        total_segments = 0
        
        print(f"Streaming data and processing segments until {num_calls} calls are complete...")
        
        # 1. STREAMING AND RECONSTRUCTION (Collect Segments)
        for segment in dataset_stream:
            total_segments += 1
            call_id = segment['file_id']
            
            # --- Progress Bar Update (Segments) ---
            if total_segments % PROGRESS_UPDATE_INTERVAL == 0:
                print(f"   [STREAMING PROGRESS] Segments processed: {total_segments:,} | Calls found: {len(processed_call_ids)}/{num_calls}...", end='\r')

            segment_text = segment.get('transcription', segment.get('sentence', ''))
            start_ts = segment.get('start_ts', 0)
            end_ts = segment.get('end_ts', start_ts)

            if segment_text:
                call_data[call_id]['segments'].append({
                    'text': segment_text, 
                    'start_ts': start_ts,
                    'end_ts': end_ts
                })
            
            if call_id not in processed_call_ids:
                processed_call_ids.add(call_id)
                if len(processed_call_ids) >= num_calls:
                    print(f"\nStopping stream after collecting segments for the first {num_calls} calls. Total segments processed: {total_segments:,}")
                    break

        print(f"\nCalls successfully collected: {len(call_data)}")
        
        # 2. STITCHING, SAVING, AND TRANSLATING
        completed_calls_count = 0
        
        print("\n--- Starting Reconstruction and Translation ---")
        
        for call_id, data in call_data.items():
            
            # Sort and Reconstruct
            data['segments'].sort(key=lambda x: x['start_ts'])
            full_transcript = " ".join([s['text'] for s in data['segments']])
            
            # Summary
            max_end_ts = max(s['end_ts'] for s in data['segments'])
            total_duration = max_end_ts / 60
            word_count = len(full_transcript.split())
            
            print(f"\n[CALL {completed_calls_count + 1}/{num_calls}] ID: {call_id} | Duration: {total_duration:.2f} min | Words: {word_count:,}")

            # 2.1. (REMOVED: Saving the original English transcript)
            
            # 2.2. Translate the full transcript
            norwegian_transcript = translate_to_norwegian(full_transcript)
            
            # 2.3. Save Translated Transcript (Norwegian)
            translated_filename = f"transcript_NO_{call_id}.txt"
            save_transcript_to_file(norwegian_transcript, translated_filename)
            
            # 2.4. Update Completion Progress
            completed_calls_count += 1
            print(f"[✅ COMPLETED] Total completed calls: {completed_calls_count}/{num_calls}. Waiting 5 seconds...")
            
            # Pause to avoid immediate rate limit
            time.sleep(5) 

        print("\n--- Processing Complete ---")
        print(f"You can find the full, translated (NO) transcripts in the '{OUTPUT_DIR}' directory.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    explore_dataset(DATASET_NAME, SPLIT, CONFIG_NAME, NUM_CALLS_TO_PROCESS)