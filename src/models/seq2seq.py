import pytorch_lightning as pl
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
from datasets import MidiDataModule
from vocab import RemiVocab, DescriptionVocab
from constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY

import torch.cuda.amp as amp
import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GroupEmbedding(nn.Module):
  def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
    super().__init__()
    self.n_tokens = n_tokens
    self.n_groups = n_groups
    self.inner_dim = inner_dim
    self.out_dim = out_dim

    self.embedding = nn.Embedding(n_tokens, inner_dim)
    self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

  def forward(self, x):
    shape = x.shape
    emb = self.embedding(x)
    return self.proj(emb.view(*shape[:-1], self.n_groups * self.inner_dim))

class Seq2SeqModule(pl.LightningModule):
  def __init__(self,
               d_model=512,
               d_latent=512,
               n_codes=512,
               n_groups=8,
               context_size=1024,  # Changed default from 512 to 1024
               lr=1e-4,
               lr_schedule='sqrt_decay',
               warmup_steps=None,
               max_steps=None,
               encoder_layers=6,
               decoder_layers=12,
               intermediate_size=2048,
               num_attention_heads=8,
               description_flavor='description',
               description_options=None,
               use_pretrained_latent_embeddings=True):
    super(Seq2SeqModule, self).__init__()

    self.description_flavor = description_flavor
    assert self.description_flavor in ['latent', 'description', 'none', 'both'], f"Unknown description flavor '{self.description_flavor}', expected one of ['latent', 'description', 'none', 'both]"
    self.description_options = description_options

    self.context_size = context_size
    self.d_model = d_model
    self.d_latent = d_latent

    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    self.vocab = RemiVocab()

    encoder_config = BertConfig(
      vocab_size=1,
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=encoder_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )
    decoder_config = BertConfig(
      vocab_size=1,
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=decoder_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    self.transformer = EncoderDecoderModel(config)
    self.transformer.config.decoder.is_decoder = True
    self.transformer.config.decoder.add_cross_attention = True


    self.max_bars = self.context_size
    self.max_positions = 1024  # Changed from 512 to 1024
    self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)
    self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)

    if self.description_flavor in ['latent', 'both']:
      if use_pretrained_latent_embeddings:
        self.latent_in = nn.Linear(self.d_latent, self.d_model, bias=False)
      else:
        self.latent_in = GroupEmbedding(n_codes, n_groups, self.d_model, inner_dim=self.d_latent//n_groups)
    if self.description_flavor in ['description', 'both']:
      desc_vocab = DescriptionVocab()
      self.desc_in = nn.Embedding(len(desc_vocab), self.d_model)
    
    if self.description_flavor == 'both':
      self.desc_proj = nn.Linear(2*self.d_model, self.d_model, bias=False)

    self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
    self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)
    
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))
        
    self.save_hyperparameters()

  def get_datamodule(self, midi_files, **kwargs):
    return MidiDataModule(
      midi_files, 
      self.context_size,
      description_flavor=self.description_flavor,
      max_bars=self.max_bars,
      max_positions=self.max_positions,
      description_options=self.description_options,
      **kwargs
    )

  def encode(self, z, desc_bar_ids=None):
    try:
      if self.description_flavor == 'both':
        desc = z['description']
        latent = z['latents']
        
        desc_emb = self.desc_in(desc)
        latent_emb = self.latent_in(latent)
        #print(f"desc_emb shape: {desc_emb.shape}, latent_emb shape: {latent_emb.shape}")
        # Simple approach: just truncate the longer sequence to match the shorter one
        if desc_emb.size(1) > latent_emb.size(1):
            desc_emb = desc_emb[:, :latent_emb.size(1), :]
        elif latent_emb.size(1) > desc_emb.size(1):
            latent_emb = latent_emb[:, :desc_emb.size(1), :]
        
        # Ensure desc_bar_ids matches desc_emb length
        if desc_bar_ids is not None:
            desc_bar_ids = desc_bar_ids[:, :desc_emb.size(1)]
            desc_emb = desc_emb + self.bar_embedding(desc_bar_ids)
        #print(f"desc_emb shape after bar embedding: {desc_emb.shape}, latent_emb shape after bar embedding: {latent_emb.shape}")
        # Now they have the same shape, concatenate along the feature dimension
        z_emb = self.desc_proj(torch.cat([desc_emb, latent_emb], dim=-1))

      elif self.description_flavor == 'description':
        z_emb = self.desc_in(z)
        if desc_bar_ids is not None:
          z_emb += self.bar_embedding(desc_bar_ids)

      elif self.description_flavor == 'latent':
        z_emb = self.latent_in(z)

      else:
        return None

      out = self.transformer.encoder(inputs_embeds=z_emb, output_hidden_states=True)
      encoder_hidden = out.hidden_states[-1]
      return encoder_hidden
    except Exception as e:
      # Get batch info to identify problematic files
      if hasattr(self, 'current_batch_files') and self.current_batch_files:
        print(f"\n\nERROR in encode method with files: {self.current_batch_files}")
      print(f"Error details: {str(e)}")
      
      # Re-raise the exception
      raise

  def decode(self, x, labels=None, bar_ids=None, position_ids=None, encoder_hidden_states=None, return_hidden=False):
    seq_len = x.size(1)

    # Shape of x_emb: (batch_size, seq_len, d_model)
    x_emb = self.in_layer(x)
    if bar_ids is not None:
      x_emb += self.bar_embedding(bar_ids)
    if position_ids is not None:
      x_emb += self.pos_embedding(position_ids)

    # # Add latent embedding to input embeddings
    # if bar_ids is not None:
    #   assert bar_ids.max() <= encoder_hidden.size(1)
    #   embs = torch.cat([torch.zeros(x.size(0), 1, self.d_model, device=self.device), encoder_hidden], dim=1)
    #   offset = (seq_len * torch.arange(bar_ids.size(0), device=self.device)).unsqueeze(1)
    #   # Use bar_ids to gather encoder hidden states s.t. latent_emb[i, j] == encoder_hidden[i, bar_ids[i, j]]
    #   latent_emb = F.embedding((bar_ids + offset).view(-1), embs.view(-1, self.d_model)).view(x_emb.shape)
    #   x_emb += latent_emb

    if encoder_hidden_states is not None:
      # Make x_emb and encoder_hidden_states match in sequence length. Necessary for relative positional embeddings
      padded = pad_sequence([x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)], batch_first=True)
      x_emb, encoder_hidden_states = padded.transpose(1, 2)

      out = self.transformer.decoder(
        inputs_embeds=x_emb, 
        encoder_hidden_states=encoder_hidden_states, 
        output_hidden_states=True
      )
      hidden = out.hidden_states[-1][:, :seq_len]
    else:
      out = self.transformer.decoder(inputs_embeds=x_emb, output_hidden_states=True)
      hidden = out.hidden_states[-1][:, :seq_len]

    # Shape of logits: (batch_size, seq_len, tuple_size, vocab_size)

    if return_hidden:
      return hidden
    else:
      return self.out_layer(hidden)


  def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, description_bar_ids=None, return_hidden=False):
    encoder_hidden = self.encode(z, desc_bar_ids=description_bar_ids)

    out = self.decode(x, 
      labels=labels, 
      bar_ids=bar_ids, 
      position_ids=position_ids, 
      encoder_hidden_states=encoder_hidden,
      return_hidden=return_hidden
    )

    return out 
    
  def get_loss(self, batch, return_logits=False):
    # Shape of x: (batch_size, seq_len, tuple_size)
    x = batch['input_ids'].to(self.device)
    bar_ids = batch['bar_ids'].to(self.device)
    position_ids = batch['position_ids'].to(self.device)
    # Shape of labels: (batch_size, tgt_len, tuple_size)
    labels = batch['labels']

    # Shape of z: (batch_size, context_size, n_groups, d_latent)
    if self.description_flavor == 'latent':
      z = batch['latents']
      desc_bar_ids = None
    elif self.description_flavor == 'description':
      z = batch['description']
      desc_bar_ids = batch['desc_bar_ids']
    elif self.description_flavor == 'both':
      z = { 'latents': batch['latents'], 'description': batch['description'] }
      desc_bar_ids = batch['desc_bar_ids']
    else:
      z, desc_bar_ids = None, None

    
    logits = self(x, z=z, labels=labels, bar_ids=bar_ids, position_ids=position_ids, description_bar_ids=desc_bar_ids)
    # Shape of logits: (batch_size, tgt_len, tuple_size, vocab_size)
    pred = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    
    loss = self.loss_fn(pred, labels)

    if return_logits:
      return loss, logits
    else:
      return loss
  
  def training_step(self, batch, batch_idx):
    if 'file_paths' in batch:
      self.current_batch_files = batch['file_paths']
    else:
      self.current_batch_files = None
    loss = self.get_loss(batch)
    self.log('train_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    return loss
  
  def validation_step(self, batch, batch_idx):
    loss, logits = self.get_loss(batch, return_logits=True)
    self.log('valid_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    y = batch['labels']
    pad_token_id = self.vocab.to_i(PAD_TOKEN)
    
    logits = logits.view(logits.size(0), -1, logits.size(-1))
    y = y.view(y.size(0), -1)

    log_pr = logits.log_softmax(dim=-1)
    log_pr[y == pad_token_id] = 0 # log(pr) = log(1) for padding
    log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)

    t = (y != pad_token_id).sum(dim=-1)
    ppl = (-log_pr.sum(dim=1) / t).exp().mean()
    self.log('valid_ppl', ppl.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    return self.get_loss(batch)
        
  def configure_optimizers(self):
    # set LR to 1, scale with LambdaLR scheduler
    optimizer = transformers.AdamW(self.parameters(), lr=1, weight_decay=0.01)

    if self.lr_schedule == 'sqrt_decay':
      # constant warmup, then 1/sqrt(n) decay starting from the initial LR
      lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1)/self.warmup_steps))
    elif self.lr_schedule == 'linear':
      # linear warmup, linear decay
      lr_func = lambda step: min(self.lr, self.lr*step/self.warmup_steps, self.lr*(1 - (step - self.warmup_steps)/self.max_steps))
    elif self.lr_schedule == 'cosine':
      # linear warmup, cosine decay to 10% of initial LR
      lr_func = lambda step: self.lr * min(step/self.warmup_steps, 0.55 + 0.45*math.cos(math.pi*(min(step, self.max_steps) - self.warmup_steps)/(self.max_steps - self.warmup_steps)))
    else:
      # Use no lr scheduling
      lr_func = lambda step: self.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    return [optimizer], [{
      'scheduler': scheduler,
      'interval': 'step',
    }]

  @torch.no_grad()
  def sample(self, batch, 
      max_length=256, 
      max_bars=-1,
      temp=0.8,
      pad_token=PAD_TOKEN, 
      eos_token=EOS_TOKEN,
      verbose=0,
  ):
    self.eval()  # Ensure the model is in eval mode

    pad_token_id = self.vocab.to_i(pad_token)
    eos_token_id = self.vocab.to_i(eos_token)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, curr_len = batch['input_ids'].shape
    x = batch['input_ids'].to(device)
    bar_ids = batch['bar_ids'].to(device)
    position_ids = batch['position_ids'].to(device)

    if self.description_flavor == 'both':
        z = { 'latents': batch['latents'].to(device), 'description': batch['description'].to(device) }
        desc_bar_ids = batch['desc_bar_ids'].to(device)
    elif self.description_flavor == 'latent':
        z, desc_bar_ids = batch['latents'].to(device), None
    elif self.description_flavor == 'description':
        z, desc_bar_ids = batch['description'].to(device), batch['desc_bar_ids'].to(device)
    else:
        z, desc_bar_ids = None, None

    is_done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Cache the encoder hidden states if applicable
    encoder_hidden_states = self.encode(z, desc_bar_ids) if self.description_flavor in ['latent', 'both'] else None

    curr_bars = torch.zeros(batch_size, device=device).fill_(-1)
    with amp.autocast():  # Use AMP for faster inference on supported GPUs
      for i in range(curr_len - 1, max_length):
          print(f"\r{i+1}/{max_length}", end='')

          x_ = x[:, -self.context_size:]
          bar_ids_ = bar_ids[:, -self.context_size:]
          position_ids_ = position_ids[:, -self.context_size:]
          
          if self.description_flavor in ['description', 'both']:
              if self.description_flavor == 'description':
                  desc = z
              else:
                  desc = z['description']

              next_bars = bar_ids_[:, 0]
              bars_changed = not (next_bars == curr_bars).all()
              curr_bars = next_bars

              if bars_changed:
                  z_ = torch.zeros(batch_size, self.context_size, dtype=torch.int, device=device)
                  desc_bar_ids_ = torch.zeros(batch_size, self.context_size, dtype=torch.int, device=device)
                  
                  for j in range(batch_size):
                      curr_bar = bar_ids_[j, 0]
                      indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
                      idx = indices[0, 0] if indices.size(0) > 0 else desc.size(1) - 1
                      offset = min(self.context_size, desc.size(1) - idx)
                      z_[j, :offset] = desc[j, idx:idx+offset]
                      desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx+offset]

                  if self.description_flavor == 'both':
                      z_ = { 'description': z_, 'latents': z['latents'] }

                  encoder_hidden_states = self.encode(z_, desc_bar_ids_)

        
          logits = self.decode(x_, bar_ids=bar_ids_, position_ids=position_ids_, encoder_hidden_states=encoder_hidden_states)
          idx = min(self.context_size - 1, i)
          logits = logits[:, idx] / temp
          pr = F.softmax(logits, dim=-1)
              
          pr = pr.contiguous().view(-1, pr.size(-1))  # Ensure contiguous memory for performance
          next_token_ids = torch.multinomial(pr, num_samples=1).view(-1)
          next_tokens = self.vocab.decode(next_token_ids.cpu())

          next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int, device=device)
          next_bar_ids = bar_ids[:, i].clone() + next_bars
          next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
          next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in next_positions]
          next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i].cpu().tolist(), next_positions)]
          next_position_ids = torch.tensor(next_positions, dtype=torch.int, device=device)

          is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
          next_token_ids[is_done] = pad_token_id
          
          if max_bars > 0:
              is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)
          
          x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
          bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
          position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)

          if is_done.all():
              break
            
    return {
        'sequences': x,
        'bar_ids': bar_ids,
        'position_ids': position_ids
    }
  def freeze_layers(self, layers):
    # Set requires_grad = False for all parameters in the specified layers
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad = False