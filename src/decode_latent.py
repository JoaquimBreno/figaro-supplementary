import os
import pickle
import torch
import argparse
import pretty_midi
from models.vae import VqVaeModule
from constants import (
    BOS_TOKEN, EOS_TOKEN, 
    DEFAULT_POS_PER_QUARTER, DEFAULT_VELOCITY_BINS, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_BINS,
    TIME_SIGNATURE_KEY, BAR_KEY, POSITION_KEY, INSTRUMENT_KEY, PITCH_KEY, VELOCITY_KEY, DURATION_KEY, TEMPO_KEY
)
from vocab import RemiVocab

def token_to_event(token_str):
    """Convert a token string to an event dictionary"""
    parts = token_str.split('_')
    
    if parts[0] == 'Pitch':
        return {'type': 'note', 'pitch': int(parts[1])}
    elif parts[0] == 'Velocity':
        return {'type': 'velocity', 'value': int(parts[1])}
    elif parts[0] == 'Duration':
        return {'type': 'duration', 'value': float(parts[1])}
    elif parts[0] == 'Position':
        return {'type': 'position', 'value': int(parts[1])}
    elif parts[0] == 'Instrument':
        is_drum = 'drum' in token_str
        if is_drum:
            return {'type': 'instrument', 'program': 0, 'is_drum': True}
        else:
            # Extract program number from instrument name
            for i in range(128):
                if pretty_midi.program_to_instrument_name(i) in token_str:
                    return {'type': 'instrument', 'program': i, 'is_drum': False}
            return {'type': 'instrument', 'program': 0, 'is_drum': False}
    elif parts[0] == 'Bar':
        return {'type': 'bar', 'value': int(parts[1])}
    else:
        return {'type': 'other', 'token': token_str}

def remi2midi(events, bpm=120, time_signature=(4, 4), polyphony_limit=16):
  vocab = RemiVocab()

  def _get_time(reference, bar, pos):
    time_sig = reference['time_sig']
    num, denom = time_sig.numerator, time_sig.denominator
    # Quarters per bar, assuming 4 quarters per whole note
    qpb = 4 * num / denom
    ref_pos = reference['pos']
    d_bars = bar - ref_pos[0]
    d_pos = (pos - ref_pos[1]) + d_bars*qpb*DEFAULT_POS_PER_QUARTER
    d_quarters = d_pos / DEFAULT_POS_PER_QUARTER
    # Convert quarters to seconds
    dt = d_quarters / reference['tempo'] * 60
    return reference['time'] + dt

  tempo_changes = [event for event in events if f"{TEMPO_KEY}_" in event]
  if len(tempo_changes) > 0:
    bpm = DEFAULT_TEMPO_BINS[int(tempo_changes[0].split('_')[-1])]

  pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
  num, denom = time_signature
  pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))
  current_time_sig = pm.time_signature_changes[0]

  instruments = {}

  # Use implicit timeline: keep track of last tempo/time signature change event
  # and calculate time difference relative to that
  last_tl_event = {
    'time': 0,
    'pos': (0, 0),
    'time_sig': current_time_sig,
    'tempo': bpm
  }

  bar = -1
  n_notes = 0
  polyphony_control = {}
  for i, event in enumerate(events):
    if event == EOS_TOKEN:
      break

    if not bar in polyphony_control:
      polyphony_control[bar] = {}

    if f"{BAR_KEY}_" in events[i]:
      # Next bar is starting
      bar += 1
      polyphony_control[bar] = {}

      if i+1 < len(events) and f"{TIME_SIGNATURE_KEY}_" in events[i+1]:
        num, denom = events[i+1].split('_')[-1].split('/')
        num, denom = int(num), int(denom)
        current_time_sig = last_tl_event['time_sig']
        if num != current_time_sig.numerator or denom != current_time_sig.denominator:
          time = _get_time(last_tl_event, bar, 0)
          time_sig = pretty_midi.TimeSignature(num, denom, time)
          pm.time_signature_changes.append(time_sig)
          last_tl_event['time'] = time
          last_tl_event['pos'] = (bar, 0)
          last_tl_event['time_sig'] = time_sig

    elif i+1 < len(events) and \
        f"{POSITION_KEY}_" in events[i] and \
        f"{TEMPO_KEY}_" in events[i+1]:
      position = int(events[i].split('_')[-1])
      tempo_idx = int(events[i+1].split('_')[-1])
      tempo = DEFAULT_TEMPO_BINS[tempo_idx]

      if tempo != last_tl_event['tempo']:
        time = _get_time(last_tl_event, bar, position)
        last_tl_event['time'] = time
        last_tl_event['pos'] = (bar, position)
        # don't change the tempo throughout the piece
        # last_tl_event['tempo'] = tempo

    elif i+4 < len(events) and \
        f"{POSITION_KEY}_" in events[i] and \
        f"{INSTRUMENT_KEY}_" in events[i+1] and \
        f"{PITCH_KEY}_" in events[i+2] and \
        f"{VELOCITY_KEY}_" in events[i+3] and \
        f"{DURATION_KEY}_" in events[i+4]:
      # get position
      position = int(events[i].split('_')[-1])
      if not position in polyphony_control[bar]:
        polyphony_control[bar][position] = {}

      # get instrument
      instrument_name = events[i+1].split('_')[-1]
      if instrument_name not in polyphony_control[bar][position]:
        polyphony_control[bar][position][instrument_name] = 0
      elif polyphony_control[bar][position][instrument_name] >= polyphony_limit:
        # If number of notes exceeds polyphony limit, omit this note
        continue

      if instrument_name not in instruments:
        if instrument_name == 'drum':
          instrument = pretty_midi.Instrument(0, is_drum=True)
        else:
          program = pretty_midi.instrument_name_to_program(instrument_name)
          instrument = pretty_midi.Instrument(program)
        instruments[instrument_name] = instrument
      else:
        instrument = instruments[instrument_name]

      # get pitch
      pitch = int(events[i+2].split('_')[-1])
      # get velocity
      velocity_index = int(events[i+3].split('_')[-1])
      velocity = min(127, DEFAULT_VELOCITY_BINS[velocity_index])
      # get duration
      duration_index = int(events[i+4].split('_')[-1])
      duration = DEFAULT_DURATION_BINS[duration_index]
      # create note and add to instrument
      start = _get_time(last_tl_event, bar, position)
      end = _get_time(last_tl_event, bar, position + duration)
      note = pretty_midi.Note(velocity=velocity,
                            pitch=pitch,
                            start=start,
                            end=end)
      instrument.notes.append(note)
      n_notes += 1
      polyphony_control[bar][position][instrument_name] += 1

  for instrument in instruments.values():
    pm.instruments.append(instrument)
  return pm

def combine_midis(midi_objects, bar_durations=None):
    """Combine multiple PrettyMIDI objects into a single one with sequential bars"""
    combined_midi = pretty_midi.PrettyMIDI()
    
    current_time_offset = 0.0
    
    for i, midi in enumerate(midi_objects):
        # Determine the duration of this bar
        if bar_durations and i < len(bar_durations):
            bar_duration = bar_durations[i]
        else:
            # If no explicit duration, calculate from the end time of the last note
            if midi.instruments and any(inst.notes for inst in midi.instruments):
                max_end_time = max(note.end for inst in midi.instruments for note in inst.notes) if any(note for inst in midi.instruments for note in inst.notes) else 4.0
                bar_duration = max_end_time
            else:
                bar_duration = 4.0  # Default bar length of 4 beats
        
        # Add each instrument with time-adjusted notes
        for src_inst in midi.instruments:
            # Find or create matching instrument in combined MIDI
            matching_inst = None
            for inst in combined_midi.instruments:
                if inst.program == src_inst.program and inst.is_drum == src_inst.is_drum:
                    matching_inst = inst
                    break
                
            if matching_inst is None:
                matching_inst = pretty_midi.Instrument(
                    program=src_inst.program,
                    is_drum=src_inst.is_drum
                )
                combined_midi.instruments.append(matching_inst)
            
            # Add time-shifted notes
            for note in src_inst.notes:
                shifted_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start + current_time_offset,
                    end=note.end + current_time_offset
                )
                matching_inst.notes.append(shifted_note)
        
        # Update the time offset for the next bar
        current_time_offset += bar_duration
    
    return combined_midi

def decode_latents_to_midi(latent_file, output_file, vae_checkpoint):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the VAE model
    vae_module = VqVaeModule.load_from_checkpoint(checkpoint_path=vae_checkpoint).to(device)
    vae_module.eval()
    
    # Load cached latents
    latents, codes = pickle.load(open(latent_file, 'rb'))
    print(f"Loaded {len(latents)} latents shape {latents.shape}")
    print(f"len codes: {codes.shape}")
    
    # Create vocabulary
    remi_vocab = RemiVocab()
    bos_token_id = remi_vocab.to_i(BOS_TOKEN)
    eos_token_id = remi_vocab.to_i(EOS_TOKEN)
    
    # List to store the midi objects for each bar
    midi_objects = []
    bar_durations = []
    
    # Process one bar at a time
    for i, latent in enumerate(latents):
        print(f"Processing latent {i+1}/{len(latents)}")
        
        # Prepare initial input with BOS token
        current_ids = torch.tensor([[bos_token_id]]).to(device)
        generated_ids = [bos_token_id]
        
        # Autoregressive generation - up to 1024 tokens or until EOS
        max_length = 1024
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions for next token
                logits = vae_module.decode(current_ids, latent.unsqueeze(0).to(device))
                
                # Get the last token prediction
                next_token_logits = logits[:, -1, :]
                
                # Sample or take argmax of the next token
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                generated_ids.append(next_token)
                
                # Break if we reach EOS token
                if next_token == eos_token_id:
                    break
                
                # Update the input ids for next iteration
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                
                # Print progress occasionally
                if len(generated_ids) % 100 == 0:
                    print(f"  Generated {len(generated_ids)} tokens so far")
        
        # Convert generated tokens to REMI events
        remi_events = remi_vocab.decode(generated_ids)
        print(f"Generated {len(remi_events)} events")
        print(f"First few events: {remi_events}")
        
        # Skip empty sequences
        if len(remi_events) <= 1:
            print(f"Warning: Bar {i+1} produced an empty sequence, skipping")
            continue
        
        # Convert REMI events to MIDI
        pm = remi2midi(remi_events)
        
        # Calculate bar duration (find the end time of the last note)
        if pm.instruments and any(inst.notes for inst in pm.instruments):
            max_end_time = max(note.end for inst in pm.instruments for note in inst.notes) if any(inst.notes for inst in pm.instruments) else 4.0
        else:
            max_end_time = 4.0  # Default duration for empty bars
        
        midi_objects.append(pm)
        bar_durations.append(max_end_time)
        
        # Optionally save individual bars as well
        # if len(latents) > 1:
        #     basename, ext = os.path.splitext(output_file)
        #     bar_path = f"{basename}_bar{i+1}{ext}"
        #     pm.write(bar_path)
        #     print(f"Saved bar to {bar_path}")
    
    # Combine all bars into a single MIDI file
    if midi_objects:
        combined_midi = combine_midis(midi_objects, bar_durations)
        combined_midi.write(output_file)
        print(f"Saved combined MIDI to {output_file}")
    else:
        print("No valid bars were generated. No combined MIDI file created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode latent vectors to MIDI files")
    parser.add_argument("--latent_file", type=str, required=True, help="Path to the cached latent file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output MIDI file")
    parser.add_argument("--vae_checkpoint", type=str, 
                       default="/home/your_email/your_email/figaro/figaro-supplementary/outputs/vq-vae/step=9543-valid_loss=0.94.ckpt",
                       help="Path to the VAE checkpoint")
    
    args = parser.parse_args()
    decode_latents_to_midi(args.latent_file, args.output_file, args.vae_checkpoint)