import os
import pretty_midi
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from multiprocessing import Pool
from functools import partial
import ast


def process_midi_file(folder_path, filename):
    # Assume 'folder_path' is accessible globally or passed as an additional argument
    file_path = os.path.join(folder_path, filename)
    data = dict.fromkeys(['filename', 'instrument', 'program_number', 'num_notes', 'time_signature', 'tempo', 'duration'], [])
    
    if filename.endswith(('.midi', '.mid', '.MIDI', '.MID')):
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            duration = midi_data.get_end_time()
            tempo_times, tempi = midi_data.get_tempo_changes()
            avg_tempo = tempi[0] if len(tempi) > 0 else None
            time_sigs = midi_data.time_signature_changes
            if(len(time_sigs)):
                time_signature = [f"{sigs.numerator}/{sigs.denominator}" for sigs in time_sigs ]
            else:
                time_signature = ['NaN']
            instruments = midi_data.instruments
            
            has_more_than_one_time_signature = len(set(time_signature)) > 1
            has_more_than_one_tempo = len(set(tempi)) > 1
            print(instruments)
            
            print(f"Processed {filename}")
            print({
                'filename': filename,
                'instrument': [instrument.name for instrument in instruments],
                'program_number': [ instrument.program for instrument in instruments],
                'num_notes': [ len(instrument.notes) for instrument in instruments] ,
                'time_signature': time_signature,
                'has_more_than_one_time_signature': has_more_than_one_time_signature,
                'has_more_than_one_tempo': has_more_than_one_tempo,
                'tempo': avg_tempo,
                'duration': duration
            })
            return {
                'filename': filename,
                'instrument': [instrument.name for instrument in instruments],
                'program_number': [ instrument.program for instrument in instruments],
                'num_notes': [ len(instrument.notes) for instrument in instruments] ,
                'time_signature': time_signature,
                'has_more_than_one_time_signature': has_more_than_one_time_signature,
                'has_more_than_one_tempo': has_more_than_one_tempo,
                'tempo': avg_tempo,
                'duration': duration
            }
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return {'error': filename}
    return None

def analyze_midi_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(('.midi', '.mid'))]  # Filter files ahead
    process_func = partial(process_midi_file, folder_path)
    
    # Decide on the number of processes
    num_processes = os.cpu_count()
    
    # Initialize Pool within 'with' statement for better resource management
    with Pool(processes=num_processes) as pool:
        # Here we use 'imap_unordered' for potentially better efficiency
        # 'chunksize' can be adjusted based on your use case
        # Wrap with 'tqdm' for progress bar, using 'total=len(files)' to know the total length
        results = list(tqdm(pool.imap_unordered(process_func, files, chunksize=10), total=len(files)))
    
    # Filter out None results and separate errors
    valid_results = []
    errors = []
    
    for result in results:
        if result is not None:
            if 'error' in result:
                errors.append(result['error'])
            else:
                valid_results.append(result)
    
    print(f"Number of errors: {len(errors)}")
    # Create DataFrame only if we have valid results
    if valid_results:
        df = pd.DataFrame(valid_results)
        return df
    else:
        print("No valid MIDI files were processed successfully")
        return pd.DataFrame()  # Return empty DataFrame if no valid results

    return df

def plot_visualizations(df):
            
        df['instrument'] = df['instrument'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df['time_signature'] = df['time_signature'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        # Explodindo a coluna de instrumentos
        df_exploded = df.explode('instrument')
        df_exploded = df_exploded.explode('time_signature')
        # Contando a ocorrência de cada instrumento
        instrument_counts = df_exploded['instrument'].value_counts().reset_index()
        instrument_counts.columns = ['Instrument', 'Count']
        # 1. Distribuição de Instrumentos (agora com todos os itens únicos)
        fig = px.bar(instrument_counts, x='Instrument', y='Count', title='Distribution of Instruments', color='Instrument')
        fig.update_layout(xaxis_title='Instrument', yaxis_title='Number of Occurrences')
        fig.show()

        # 2. Distribuição das Assinaturas de Tempo (todos os itens únicos)
        time_signature_counts = df_exploded['time_signature'].value_counts().reset_index()
        time_signature_counts.columns = ['Time Signature', 'Count']
        fig = px.bar(time_signature_counts, x='Time Signature', y='Count', title='Distribution of Time Signatures', color='Time Signature')
        fig.update_layout(xaxis_title='Time Signature', yaxis_title='Count')
        fig.show()
        
        # 3. Músicas com mais de uma assinatura de tempo e mais de uma mudança de tempo
        more_than_one_ts = sum(df['has_more_than_one_time_signature'])
        more_than_one_tempo = sum(df['has_more_than_one_tempo'])
        
        counts = pd.DataFrame({
            'Metric': ['More than one Time Signature', 'More than one Tempo'],
            'Count': [more_than_one_ts, more_than_one_tempo]
        })
        
        fig = px.bar(counts, x='Metric', y='Count', title='Songs with Multiple Time Signatures and Tempo Changes')
        fig.update_layout(xaxis_title='', yaxis_title='Number of Songs')
        fig.show()

def main():
    # Pasta com arquivos MIDI
    folder_path = '/home/your_email/your_email/figaro/figaro-supplementary/samples3/checkpoint/max_iter=8000,max_bars=32'
    
    # Analisar arquivos MIDI
    midi_dataframe = analyze_midi_files(folder_path)
    
    # # Salvar dados em CSV (opcional)
    midi_dataframe.to_csv('midi_analysis_check_results.csv', index=False)
    
    # # Plotar visualizações
    # plot_visualizations(midi_dataframe)

if __name__ == "__main__":
    main()
