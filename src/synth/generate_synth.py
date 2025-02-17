import soundfile as sf
import fluidsynth
import os
import gc
import warnings
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import pretty_midi
import numpy as np

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pretty_midi.instrument._HAS_FLUIDSYNTH = True
pretty_midi.instrument.fluidsynth = fluidsynth

def remove_duplicates(files):
    unique_files = {}
    for file in files:
        stripped_name = file[1:]
        if stripped_name not in unique_files:
            unique_files[stripped_name] = file
    return list(unique_files.values())

def extract_beats_and_downbeats(pm):
    try:
        beats = pm.get_beats()
        downbeats = pm.get_downbeats()
        
        beat_sequence = []
        measure_beat_counter = 1
        
        for beat in beats:
            if beat in downbeats:
                measure_beat_counter = 1
            
            beat_sequence.append((beat, measure_beat_counter))
            measure_beat_counter = measure_beat_counter % 4 + 1
        
        return beat_sequence
    except Exception as e:
        print(f"Beat extraction error: {e}")
        return []

def save_beat_sequence(beat_sequence, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for time, beat_number in beat_sequence:
                f.write(f"{time:.9f} {beat_number}\n")
    except Exception as e:
        print(f"Error saving beat sequence: {e}")

def generate_audio_safely(pm, sample_rate=44100):
    try:
        # Reduce quality to save memory
        audio = pm.fluidsynth(fs=sample_rate)
        return audio
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None

def run_synth(folder_path, file):
    filepath = os.path.join(folder_path, file)
    
    try:
        # Load MIDI with error handling
        pm = pretty_midi.PrettyMIDI(filepath)
        
        # Generate audio safely
        audio = generate_audio_safely(pm)
        
        if audio is not None:
            # Save WAV
            wav_output_path = os.path.join(folder_path, "wav", file.replace('.mid', '.wav'))
            os.makedirs(os.path.dirname(wav_output_path), exist_ok=True)
            sf.write(wav_output_path, audio, 44100)
        
        # Extract beats
        beat_sequence = extract_beats_and_downbeats(pm)
        
        # Save beat sequence
        beat_output_path = os.path.join(folder_path, "beats", file.replace('.mid', '.txt'))
        save_beat_sequence(beat_sequence, beat_output_path)
        
        # Memory cleanup
        del pm
        del audio
        gc.collect()
        
        return file
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def get_processed_files(path):
    beat_path = os.path.join(path, "beats")
    if not os.path.exists(beat_path):
        return set()
    
    processed_files = set()
    for filename in os.listdir(beat_path):
        if filename.endswith('.txt'):
            # Replace .txt with .mid to get the original MIDI file name
            midi_filename = filename.replace('.txt', '.mid')
            processed_files.add(midi_filename)
    return processed_files

def processing_pool(path):
    # Find MIDI files
    all_files = [f for f in os.listdir(path) if f.endswith(('.midi', '.mid', '.MIDI', '.MID'))]
    print(f"Found {len(all_files)} MIDI files")
    
    # Remove duplicates
    files = remove_duplicates(all_files)
    print(f"Removed {len(all_files) - len(files)} duplicates")
    
    # Determine which files have already been processed
    processed_files = get_processed_files(path)
    files_to_process = [f for f in files if f not in processed_files]
    print(f"Found {len(processed_files)} already processed files")
    print(f"{len(files_to_process)} files left to process")
    
    # Define number of workers - use all available CPU cores
    max_workers = os.cpu_count()-4
    
    print(f"Processing {len(files_to_process)} files using {max_workers} workers")
    
    # Prepare processing function
    process_func = partial(run_synth, path)
    
    # Parallel processing
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, files_to_process), total=len(files_to_process)))
    
    processed_files_count = len([r for r in results if r is not None])
    print(f"Successfully processed {processed_files_count} files")
    
    # Free memory
    gc.collect()

# Path to MIDI files
path = '/home/your_email/your_email/figaro/figaro-supplementary/selected_balance'
# Execute processing
processing_pool(path)