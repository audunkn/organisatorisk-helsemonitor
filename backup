import pandas as pd
from datasets import load_dataset, Audio
import os
from collections import defaultdict
from itertools import islice

# --- Configuration ---
DATASET_NAME = "distil-whisper/earnings22"
# FIXED: Changed split to 'test', as 'train' is not available for this dataset version.
SPLIT = "test" 
CONFIG_NAME = "chunked" # Required config name for this dataset version
OUTPUT_DIR = "full_transcripts_output"

# Set this to control how many unique, full-length transcripts are reconstructed
# Set higher to process more calls.
NUM_CALLS_TO_PROCESS = 30 

# --- New: Progress Bar Configuration ---
PROGRESS_UPDATE_INTERVAL = 500 # Print status every X segments processed

# Suppress benign warnings/errors from datasets library on Windows
import datasets
datasets.logging.set_verbosity_error()

def save_transcript_to_file(transcript: str, filename: str):
    """Saves the reconstructed transcript to a plain text file."""
    try:
        # Ensure the output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"-> Successfully saved full transcript to '{filepath}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

def explore_dataset(dataset_name: str, split: str, config_name: str, num_calls: int):
    """
    Loads the Earnings-22 dataset using streaming to minimize disk/memory usage,
    reconstructs full transcripts for a specified number of calls, and saves them.
    """
    print(f"--- Loading Dataset: {dataset_name}, Config: {config_name}, Split: {split} (Streaming Mode) ---")
    
    try:
        # 1. Load the dataset in streaming mode to avoid caching the entire large file
        # Streaming is key to preventing memory and disk space exhaustion.
        dataset_stream = load_dataset(dataset_name, config_name, split=split, streaming=True)
        
        # Disable audio decoding to avoid dependency issues
        if 'audio' in dataset_stream.column_names:
            dataset_stream = dataset_stream.cast_column("audio", Audio(decode=False))
            dataset_stream = dataset_stream.remove_columns(['audio'])

        # 2. Process the stream to group segments by call ID.
        call_data = defaultdict(lambda: {'segments': []})
        processed_call_ids = set()
        
        # --- Progress Bar Variables ---
        total_segments = 0
        
        print(f"Streaming data and processing segments until {num_calls} calls are complete...")
        
        # Iterate over the streaming dataset
        for segment in dataset_stream:
            total_segments += 1
            call_id = segment['file_id']
            
            # --- Progress Bar Update ---
            if total_segments % PROGRESS_UPDATE_INTERVAL == 0:
                 print(f"   [PROGRESS] Segments processed: {total_segments:,} | Calls found: {len(processed_call_ids)}/{num_calls}...", end='\r')

            # Extract relevant text and timing information
            segment_text = segment.get('transcription', segment.get('sentence', ''))
            start_ts = segment.get('start_ts', 0)
            end_ts = segment.get('end_ts', start_ts)

            if segment_text:
                call_data[call_id]['segments'].append({
                    'text': segment_text, 
                    'start_ts': start_ts,
                    'end_ts': end_ts
                })
            
            # Stop streaming once enough unique calls are fully collected
            if call_id not in processed_call_ids:
                processed_call_ids.add(call_id)
                if len(processed_call_ids) >= num_calls:
                    # Final progress bar update before breaking
                    print(f"Stopping stream after collecting segments for the first {num_calls} calls. Total segments processed: {total_segments:,}")
                    break

        print(f"Calls found: {len(call_data)}")
        
        # 3. Stitch and Save the reconstructed calls
        for call_id, data in call_data.items():
            # Sort segments by timestamp (start_ts) to ensure chronological order
            data['segments'].sort(key=lambda x: x['start_ts'])
            
            # Concatenate text segments into one full transcript
            full_transcript = " ".join([s['text'] for s in data['segments']])
            
            # Calculate call-specific statistics
            max_end_ts = max(s['end_ts'] for s in data['segments'])
            total_duration = max_end_ts / 60
            word_count = len(full_transcript.split())

            # Save the result
            output_filename = f"transcript_{call_id}.txt"
            save_transcript_to_file(full_transcript, output_filename)
            
            # Display summary and a snippet
            print(f"\nCALL ID: {call_id}")
            print(f"Total Duration: {total_duration:.2f} minutes")
            print(f"Total Word Count: {word_count:,}")
            print(f"Transcript Snippet (Start):\n{full_transcript[:700]}...")


        print("\n--- Processing Complete ---")
        print(f"You can find the full, individual transcripts in the '{OUTPUT_DIR}' directory.")
        
    except Exception as e:
        print(f"An error occurred while loading or processing the dataset: {e}")
        print("NOTE: Streaming requires specific dataset structures. If the issue persists, the dataset may not support streaming.")

if __name__ == "__main__":
    explore_dataset(DATASET_NAME, SPLIT, CONFIG_NAME, NUM_CALLS_TO_PROCESS)