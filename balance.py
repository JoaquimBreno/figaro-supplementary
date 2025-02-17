import os
import pretty_midi
import pandas as pd
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

# Target time signatures to include in the analysis
TARGET_TIME_SIGNATURES = {'4/4', '3/4', '2/4', '6/8', '12/8', '5/4', '5/8'}

def process_midi_file(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    if filename.lower().endswith(('.midi', '.mid')):
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            time_sigs = midi_data.time_signature_changes
            time_signature = [
                f"{sig.numerator}/{sig.denominator}" for sig in time_sigs
                if f"{sig.numerator}/{sig.denominator}" in TARGET_TIME_SIGNATURES
            ] or ['NaN']  # Filter to include only target time signatures
            return {
                'filename': filename,
                'time_signature': time_signature
            }
        except Exception as e:
            return {'error': filename, 'time_signature': ['NaN']}
    return None

def analyze_midi_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.midi', '.mid'))]
    process_func = partial(process_midi_file, folder_path)
    num_processes = os.cpu_count() or 1
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, files, chunksize=10), total=len(files)))
    # Filter out None results and errors
    valid_results = [result for result in results if result and 'error' not in result]
    return pd.DataFrame(valid_results)

def balance_time_signatures(df):
    df_exploded = df.explode('time_signature').dropna()
    # Keep only rows with target time signatures
    df_filtered = df_exploded[df_exploded['time_signature'].isin(TARGET_TIME_SIGNATURES)]
    # Find the minimum count among target time signatures
    min_count = df_filtered['time_signature'].value_counts().min()
    balanced_df = pd.DataFrame()
    for ts in TARGET_TIME_SIGNATURES:
        ts_df = df_filtered[df_filtered['time_signature'] == ts].sample(
            n=min_count, random_state=1, replace=False)
        balanced_df = pd.concat([balanced_df, ts_df])
    # Group back by filename if needed
    balanced = balanced_df.groupby('filename').agg({'time_signature': lambda x: list(set(x))}).reset_index()
    return balanced

def main():
    folder_path = '/home/your_email/your_email/figaro/figaro-supplementary/data'  # Update this path to match your MIDI files directory
    midi_dataframe = analyze_midi_files(folder_path)
    balanced_df = balance_time_signatures(midi_dataframe)
    balanced_df.to_csv('balanced_significant_midi_instances.csv', index=False)

if __name__ == "__main__":
    main()