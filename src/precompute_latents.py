import os
import glob
import pickle
import random
import torch
import torch.multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

from models.vae import VqVaeModule
from constants import MASK_TOKEN
from datasets import MidiDataset, SeqCollator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
ROOT_DIR = os.getenv('ROOT_DIR', '/home/your_email/your_email/figaro/figaro-supplementary/data_pure')
MAX_N_FILES = int(os.getenv('MAX_N_FILES', '400000'))

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
  N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)
LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH', os.path.join('./temp', 'latent'))
os.makedirs(LATENT_CACHE_PATH, exist_ok=True)


def process_files(rank, world_size, files, vae_checkpoint):
    # Set the GPU for this process
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Load VAE model on this device
    vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=vae_checkpoint).to(device)
    vae_module.eval()
    vae_module.freeze()
    
    collator = SeqCollator(context_size=vae_module.context_size)
    
    # Split files among processes
    files_per_process = len(files) // world_size
    start_idx = rank * files_per_process
    end_idx = start_idx + files_per_process if rank < world_size - 1 else len(files)
    process_files = files[start_idx:end_idx]
    
    print(f"GPU {rank} processing {len(process_files)} files ({start_idx} to {end_idx-1})")
    
    for i, file in enumerate(process_files):
        print(f"GPU {rank} - {i:4d}/{len(process_files)}: {file} ", end='')
        cache_key = os.path.basename(file)
        cache_file = os.path.join(LATENT_CACHE_PATH, cache_key)

        try:
            latents, codes = pickle.load(open(cache_file, 'rb'))
            print(f'(already cached: {len(latents)} bars)')
            continue
        except:
            pass

        ds = MidiDataset([file], vae_module.context_size, 
            description_flavor='none',
            max_bars_per_context=1, 
            bar_token_mask=MASK_TOKEN,
            print_errors=True,
        )

        dl = DataLoader(ds, 
            collate_fn=collator, 
            batch_size=BATCH_SIZE, 
            num_workers=N_WORKERS // world_size, 
            pin_memory=True
        )

        latents, codes = [], []
        for batch in dl:
            x = batch['input_ids'].to(device)

            out = vae_module.encode(x)
            latents.append(out['z'])
            codes.append(out['codes'])
        
        if len(latents) == 0:
            continue
            
        latents = torch.cat(latents).cpu()
        codes = torch.cat(codes).cpu()
        print(f'(caching latents: {latents.size(0)} bars)')

        # Try to store the computed representation in the cache directory
        try:
            pickle.dump((latents, codes), open(cache_file, 'wb'))
        except Exception as err:
            print('Unable to cache file:', str(err))


if __name__ == "__main__":
    ### Create data loaders ###
    midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
    if MAX_N_FILES > 0:
        midi_files = midi_files[200000:MAX_N_FILES]

    # Shuffle files for approximate parallelizability
    random.shuffle(midi_files)

    VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', '/home/your_email/your_email/figaro/figaro-supplementary/outputs/vq-vae/step=9543-valid_loss=0.94.ckpt')
    
    print('***** PRECOMPUTING LATENT REPRESENTATIONS *****')
    print(f'Number of files: {len(midi_files)}')
    print(f'Using cache: {LATENT_CACHE_PATH}')
    
    # Use multiple GPUs if available
    n_gpus = torch.cuda.device_count()
    print(f'Number of available GPUs: {n_gpus}')
    print('***********************************************')
    
    if n_gpus > 1:
        mp.spawn(
            process_files,
            args=(n_gpus, midi_files, VAE_CHECKPOINT),
            nprocs=n_gpus,
            join=True
        )
    else:
        # Single GPU/CPU processing
        process_files(0, 1, midi_files, VAE_CHECKPOINT)