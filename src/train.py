import torch

import os
import glob

import pytorch_lightning as pl

from models.seq2seq import Seq2SeqModule
from models.vae import VqVaeModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

progress_bar = TQDMProgressBar(refresh_rate=5)  # Set the refresh rate as needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_DIR = os.getenv('ROOT_DIR', '/home/your_email/your_email/figaro/figaro-supplementary/data_pure')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/home/your_email/your_email/figaro/figaro-supplementary/outputs')
LOGGING_DIR = os.getenv('LOGGING_DIR', './logs')
MAX_N_FILES = int(os.getenv('MAX_N_FILES', -1))

MODEL = os.getenv('MODEL', 'figaro')
MODEL_NAME = os.getenv('MODEL_NAME', 'figaro')
N_CODES = int(os.getenv('N_CODES', 2048))
N_GROUPS = int(os.getenv('N_GROUPS', 16))
D_MODEL = int(os.getenv('D_MODEL', 1024))
D_LATENT = int(os.getenv('D_LATENT', 1024))

CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', '/home/your_email/your_email/figaro/figaro-supplementary/outputs/vq-vae/step=9543-valid_loss=0.94.ckpt')

BATCH_SIZE = int(os.getenv('BATCH_SIZE', 3))
TARGET_BATCH_SIZE = int(os.getenv('TARGET_BATCH_SIZE', 16))

EPOCHS = int(os.getenv('EPOCHS', '10'))
WARMUP_STEPS = int(float(os.getenv('WARMUP_STEPS', 4000)))
MAX_STEPS = int(float(os.getenv('MAX_STEPS', 1e20)))
MAX_TRAINING_STEPS = int(float(os.getenv('MAX_TRAINING_STEPS', 100_000)))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 1e-4))
LR_SCHEDULE = os.getenv('LR_SCHEDULE', 'const')
CONTEXT_SIZE = int(os.getenv('CONTEXT_SIZE', 1024))

ACCUMULATE_GRADS = max(1, TARGET_BATCH_SIZE//BATCH_SIZE)

N_WORKERS = min(os.cpu_count(), float(os.getenv('N_WORKERS', 'inf')))
if device.type == 'cuda':
  N_WORKERS = min(N_WORKERS, 8*torch.cuda.device_count())
N_WORKERS = int(N_WORKERS)


def main():
  ### Define available models ###

  available_models = [
    'vq-vae',
    'figaro-learned',
    'figaro-expert',
    'figaro',
    'figaro-inst',
    'figaro-chord',
    'figaro-meta',
    'figaro-no-inst',
    'figaro-no-chord',
    'figaro-no-meta',
    'baseline',
  ]

  assert MODEL is not None, 'the MODEL needs to be specified'
  assert MODEL in available_models, f'unknown MODEL: {MODEL}'

  print(MODEL)

  ### Create data loaders ###
  midi_files = glob.glob(os.path.join('temp', 'latent', '*.mid'), recursive=True)
  
  print(midi_files[0])
  # Filter midi_files to only include those that are in the latent_midi_set
  midi_files = [file.replace('temp/latent', ROOT_DIR) for file in midi_files ]
  print("EXAMPLE",midi_files[0])
  print("LENGTH OF MIDI FILES", len(midi_files))
  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]

  if len(midi_files) == 0:
    print(f"WARNING: No MIDI files were found at '{ROOT_DIR}'. Did you download the dataset to the right location?")
    exit()


  MAX_CONTEXT = min(1024, CONTEXT_SIZE)

  if MODEL in ['figaro-learned', 'figaro'] and VAE_CHECKPOINT:
    vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=VAE_CHECKPOINT)
    vae_module.cpu()
    vae_module.freeze()
    vae_module.eval()

  else:
    vae_module = None


  ### Create and train model ###

  # load model from checkpoint if available

  if CHECKPOINT:
    model_class = {
      'vq-vae': VqVaeModule,
      'figaro-learned': Seq2SeqModule,
      'figaro-expert': Seq2SeqModule,
      'figaro': Seq2SeqModule,
      'figaro-inst': Seq2SeqModule,
      'figaro-chord': Seq2SeqModule,
      'figaro-meta': Seq2SeqModule,
      'figaro-no-inst': Seq2SeqModule,
      'figaro-no-chord': Seq2SeqModule,
      'figaro-no-meta': Seq2SeqModule,
      'baseline': Seq2SeqModule,
    }[MODEL]
    model = model_class.load_from_checkpoint(checkpoint_path=CHECKPOINT)
    model.freeze_layers([
        model.transformer.encoder
    ])
    
  else:
    seq2seq_kwargs = {
      'encoder_layers': 4,
      'decoder_layers': 6,
      'num_attention_heads': 8,
      'intermediate_size': 2048,
      'd_model': D_MODEL,
      'context_size': MAX_CONTEXT,
      'lr': LEARNING_RATE,
      'warmup_steps': WARMUP_STEPS,
      'max_steps': MAX_STEPS,
    }
    dec_kwargs = { **seq2seq_kwargs }
    dec_kwargs['encoder_layers'] = 0

    # use lambda functions for lazy initialization
    model = {
      'vq-vae': lambda: VqVaeModule(
        encoder_layers=4,
        decoder_layers=6,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        n_codes=N_CODES, 
        n_groups=N_GROUPS, 
        context_size=MAX_CONTEXT,
        lr=LEARNING_RATE,
        lr_schedule=LR_SCHEDULE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        d_model=D_MODEL,
        d_latent=D_LATENT,
      ),
      'figaro-learned': lambda: Seq2SeqModule(
        description_flavor='latent',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro': lambda: Seq2SeqModule(
        description_flavor='both',
        n_codes=vae_module.n_codes,
        n_groups=vae_module.n_groups,
        d_latent=vae_module.d_latent,
        **seq2seq_kwargs
      ),
      'figaro-expert': lambda: Seq2SeqModule(
        description_flavor='description',
        **seq2seq_kwargs
      ),
      'figaro-no-meta': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': True, 'meta': False },
        **seq2seq_kwargs
      ),
      'figaro-no-inst': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': False, 'chords': True, 'meta': True },
        **seq2seq_kwargs
      ),
      'figaro-no-chord': lambda: Seq2SeqModule(
        description_flavor='description',
        description_options={ 'instruments': True, 'chords': False, 'meta': True },
        **seq2seq_kwargs
      ),
      'baseline': lambda: Seq2SeqModule(
        description_flavor='none',
        **dec_kwargs
      ),
    }[MODEL]()

  datamodule = model.get_datamodule(
    midi_files,
    vae_module=vae_module,
    batch_size=BATCH_SIZE, 
    num_workers=N_WORKERS, 
    pin_memory=True
  )

  print(f"check path {os.path.join(OUTPUT_DIR, MODEL)}")
  checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
    monitor='valid_loss',
    dirpath=os.path.join(OUTPUT_DIR, MODEL),
    filename='{step}-{valid_loss:.2f}',
    save_last=True,
    save_top_k=2,
    mode='min'
  )

  lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

  wandb_logger = WandbLogger(
      project='figaro',
      log_model=True,
      save_dir=LOGGING_DIR,
      config={
          'batch_size': BATCH_SIZE,
          'learning_rate': LEARNING_RATE,
          'epochs': EPOCHS,
          # Add other hyperparameters you wish to track
      }
  )

  num_batches = len(midi_files) // BATCH_SIZE
  trainer = pl.Trainer(
    gpus=2,  # Use 2 GPUs
    strategy='ddp',  # Use Distributed Data Parallel
    precision=16,  # Use mixed precision
    profiler='simple',
    callbacks=[checkpoint_callback, lr_monitor, progress_bar],
    max_epochs=EPOCHS,
    max_steps=MAX_TRAINING_STEPS,
    log_every_n_steps=max(100, min(25*ACCUMULATE_GRADS, 200)),
    val_check_interval=max(500, min(300*ACCUMULATE_GRADS, 1000)),
    limit_val_batches=64,
    accumulate_grad_batches=ACCUMULATE_GRADS,
    gradient_clip_val=1.0, 
    terminate_on_nan=True,
    resume_from_checkpoint=CHECKPOINT,
    logger=wandb_logger
  )

  trainer.fit(model, datamodule)

if __name__ == '__main__':
  main()