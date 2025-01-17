import os
import torch
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from input_representation import remi2midi

# Define paths and parameters
CHECKPOINT = '/home/your_email/figaro/figaro-supplementary/outputs/figaro-expert/step=6999-valid_loss=1.06.ckpt'
FILE_TO_COMPLETE = 'azz.mid'
OUTPUT_FILE = 'output_file.mid'

BATCH_SIZE = 1
MAX_BARS = 32
VERBOSE = 2

def reconstruct_sample(model, batch, 
                       max_bars=-1, 
                       verbose=0):
    # Similar implementation as before]
    
    batch_size, seq_len = batch['input_ids'].shape[:2]

    batch_ = {key: batch[key][:, :1] for key in ['input_ids', 'bar_ids', 'position_ids']}
    if model.description_flavor in ['description', 'both']:
        batch_['description'] = batch['description']
        batch_['desc_bar_ids'] = batch['desc_bar_ids']
    if model.description_flavor in ['latent', 'both']:
        batch_['latents'] = batch['latents']

    max_len = seq_len + 1024
    sample = model.sample(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose // 2)

    xs_hat = sample['sequences'].detach().cpu()
    events_hat = [model.vocab.decode(x) for x in xs_hat]

    for events in events_hat:
        try:
            pm_hat = remi2midi(events)
            pm_hat.write(OUTPUT_FILE)
            print(f"File completed and saved to: {OUTPUT_FILE}")
        except Exception as err:
            print("ERROR: Could not convert events to MIDI:", err)

def main():
    if CHECKPOINT is None:
        raise ValueError("Please specify a CHECKPOINT path.")

    # Load the model
    model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
    model.freeze()
    model.eval()

    # Prepare and load the dataset
    dataset = MidiDataset(
        [FILE_TO_COMPLETE],
        max_len=-1,
        description_flavor=model.description_flavor,
        max_bars=model.context_size
    )

    coll = SeqCollator(context_size=-1)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)
    print(f"Number of batches: {len(dataloader)}")
    with torch.no_grad():
        for batch in dataloader:
            print(f"Completing file: {FILE_TO_COMPLETE}")
            reconstruct_sample(model, batch, max_bars=MAX_BARS, verbose=VERBOSE)

if __name__ == '__main__':
    main()