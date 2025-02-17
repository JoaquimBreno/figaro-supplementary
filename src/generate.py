
import os
import glob
import time
import torch
import random
from torch.utils.data import DataLoader
import pandas as pd
from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi
import tqdm
import copy
import torch.nn as nn

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', '/home/your_email/your_email/figaro/figaro-supplementary/data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples4')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16000))
MAX_BARS = int(os.getenv('MAX_BARS', 200))

MAKE_MEDLEYS = False
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))
  
CHECKPOINT = os.getenv('CHECKPOINT', "/home/your_email/your_email/figaro/figaro-supplementary/outputs/figaro-expert/step=37999-valid_loss=0.94.ckpt")
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', '/home/your_email/your_email/checkpoints/vq-vae.ckpt')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 2))
VERBOSE = int(os.getenv('VERBOSE', 2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def trim_midi(midi_obj, target_duration):
    for instrumento in midi_obj.instruments:
        notas_filtradas = []
        for nota in instrumento.notes:
            if nota.start < target_duration:
                if nota.end > target_duration:
                    nota.end = target_duration
                notas_filtradas.append(nota)
        instrumento.notes = notas_filtradas
        instrumento.pitch_bends = [pb for pb in instrumento.pitch_bends if pb.time < target_duration]
        instrumento.control_changes = [cc for cc in instrumento.control_changes if cc.time < target_duration]

    return midi_obj
  
def get_midi_duration(midi_obj):
    total_time = 0
    for instrumento in midi_obj.instruments:
        # Calcula o final da última nota do instrumento, se houver
        if instrumento.notes:
            instrumento_duration = max(nota.end for nota in instrumento.notes)
            total_time = max(total_time, instrumento_duration)
    return total_time
  
def setup_model(checkpoint_path):
    model = Seq2SeqModule.load_from_checkpoint(checkpoint_path)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    actual_model = model.module if isinstance(model, nn.DataParallel) else model
    actual_model.to(device)
    actual_model.freeze()
    actual_model.eval()
    
    return actual_model
  
def reconstruct_sample(model, batch, 
  initial_context=1, 
  output_dir=None, 
  max_iter=-1, 
  max_bars=-1,
  target_duration= 40,
  verbose=0,
):
    
  batch_size, seq_len = batch['input_ids'].shape[:2]

  batch_ = { key: batch[key][:, :initial_context].to(device) for key in ['input_ids', 'bar_ids', 'position_ids'] }
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description'].to(device)
    batch_['desc_bar_ids'] = batch['desc_bar_ids'].to(device)
  if model.description_flavor in ['latent', 'both']:
    batch_['latents'] = batch['latents'].to(device)

  max_len = seq_len + 1024
  if max_iter > 0:
    max_len = min(max_len, initial_context + max_iter)
  if verbose:
    print(f"Generating sequence ({initial_context} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)")
    print(f"(target duration: {target_duration} seconds)")
  sample = model.sample(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose//2)

  xs = batch['input_ids'].detach().cpu()
  xs_hat = sample['sequences'].detach().cpu()
  events = [model.vocab.decode(x) for x in xs]
  events_hat = [model.vocab.decode(x) for x in xs_hat]

  pms, pms_hat = [], []
  n_fatal = 0
  for rec, rec_hat in zip(events, events_hat):
    try:
      pm = remi2midi(rec)
      pms.append(pm)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
    
    try:
      pm_hat = remi2midi(rec_hat)
      duration = get_midi_duration(pm_hat)
      
      # Se a duração for muito curta, tenta gerar novamente
      attempts = 0
      while duration < target_duration * 0.9 and attempts < 4:
          if verbose:
            print(f"Duration too short ({duration:.2f} < {target_duration * 0.9:.2f}), generating again...")
          sample = model.sample(batch_, max_length=max_len * 2, max_bars=max_bars * 2, verbose=verbose//2)
          new_xs_hat = sample['sequences'].detach().cpu()
          new_events_hat = [model.vocab.decode(x) for x in new_xs_hat]
          new_pm_hat = remi2midi(new_events_hat[0])
          for i, instrumento in enumerate(new_pm_hat.instruments):
              if i < len(pm_hat.instruments):
                  # Ajusta o tempo de início das novas notas para concaternar corretamente
                  offset = duration
                  for note in instrumento.notes:
                      note.start += offset
                      note.end += offset
                  pm_hat.instruments[i].notes.extend(instrumento.notes)
                  pm_hat.instruments[i].pitch_bends.extend(
                      [copy.deepcopy(pb) for pb in instrumento.pitch_bends]
                  )
                  pm_hat.instruments[i].control_changes.extend(
                      [copy.deepcopy(cc) for cc in instrumento.control_changes]
                  )
          duration = get_midi_duration(pm_hat)
          print(f"New duration: {duration:.2f}")
          attempts += 1
      
      # Se a duração for muito longa, corta
      # if duration > target_duration * 1.4:
          # Implementar lógica de corte do MIDI
          # pm_hat = trim_midi(pm_hat, target_duration)
      
      pms_hat.append(pm_hat)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
 

  if output_dir:
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    count = 0 
    for pm, pm_hat, file in zip(pms, pms_hat, batch['files']):
      count += 1
      if verbose:
        file.split()
        print(f"Saving to {output_dir}/{file}")
        pm.write(os.path.join(output_dir, 'gt', file))
      pm_hat.write(os.path.join(output_dir, str(count)+file))

  return events


def get_processed_files(path):
    print(path)
    if not os.path.exists(path):
        return set()
    
    processed_files = set()
    for filename in os.listdir(path):
      processed_files.add(filename[1:])
    print(processed_files)
    return processed_files
  
def main():
  print(f"Using device: {device}")
  
  if MAKE_MEDLEYS:
    max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
  else:
    max_bars = MAX_BARS

  if OUTPUT_DIR:
    params = []
    if MAKE_MEDLEYS:
      params.append(f"n_pieces={N_MEDLEY_PIECES}")
      params.append(f"n_bars={N_MEDLEY_BARS}")
    if MAX_ITER > 0:
      params.append(f"max_iter={MAX_ITER}")
    if MAX_BARS > 0:
      params.append(f"max_bars={MAX_BARS}")
    output_dir = os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
  else:
    raise ValueError("OUTPUT_DIR must be specified.")

  print(f"Saving generated files to: {output_dir}")

  if VAE_CHECKPOINT:
    vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT).to(device)
    vae_module.eval()
  else:
    vae_module = None
  if os.path.exists(CHECKPOINT):
      print(f"O caminho {CHECKPOINT} existe.")
  else:
      print(f"O caminho {CHECKPOINT} não existe.")

  # Verifica se o caminho especificado é de um arquivo
  if os.path.isfile(CHECKPOINT):
      print(f"O caminho {CHECKPOINT} é um arquivo.")
  else:
      print(f"O caminho {CHECKPOINT} não é um arquivo ou não existe.")

  model = setup_model(CHECKPOINT)
  # model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
  # model.to(device)
  # model.freeze()
  # model.eval()


  #midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  midis_names = pd.read_csv('/home/your_email/your_email/figaro/figaro-supplementary/balanced_significant_midi_instances_expanded.csv')
  midis_names = midis_names['filename'].tolist()
  processed_files = get_processed_files(output_dir)
  print(f"Found {len(midis_names)} MIDI files")
  print(f"Found {len(processed_files)} already processed files")
  files_to_process = [f for f in midis_names if f not in processed_files]
  print(f"Found {len(files_to_process)} files to process")
  
  midi_files = [os.path.join(ROOT_DIR, name) for name in files_to_process]
  
  dm = model.get_datamodule(midi_files, vae_module=vae_module)
  dm.setup('test')
  midi_files = dm.test_ds.files
  random.shuffle(midi_files)

  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]


  description_options = None
  if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )


  start_time = time.time()
  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)

  if MAKE_MEDLEYS:
    dl = medley_iterator(dl, 
      n_pieces=N_MEDLEY_BARS, 
      n_bars=N_MEDLEY_BARS, 
      description_flavor=model.description_flavor
    )
  
  with torch.no_grad():
    for batch in tqdm.tqdm(dl, total=len(dl)): 
      reconstruct_sample(model, batch, 
        output_dir=output_dir, 
        max_iter=MAX_ITER, 
        max_bars=max_bars,
        target_duration= 40,
        verbose=VERBOSE,
      )

if __name__ == '__main__':
  main()
