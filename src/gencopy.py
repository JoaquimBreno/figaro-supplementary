
def remi2midi(events, bpm=120, time_signature=(4, 4), polyphony_limit=16):
  vocab = RemiVocab()

  def _get_time(bar, position, bpm=120, positions_per_bar=48):
    abs_position = bar*positions_per_bar + position
    beat = abs_position / DEFAULT_POS_PER_QUARTER
    return beat/bpm*60

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

  # time_sigs = [event.split('_')[-1].split('/') for event in events if f"{TIME_SIGNATURE_KEY}_" in event]
  # time_sigs = [(int(num), int(denom)) for num, denom in time_sigs]

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
      # create not and add to instrument
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pl_ckpt = torch.load("checkpoints/figaro-expert.ckpt")

kwargs = pl_ckpt["hyper_parameters"]
del kwargs["flavor"]
model = Seq2SeqModule(**kwargs)

try:
  model.load_state_dict(pl_ckpt["state_dict"])
except RuntimeError:
  # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
  config = model.transformer.decoder.bert.config
  for layer in model.transformer.decoder.bert.encoder.layer:
    layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
  model.load_state_dict(pl_ckpt["state_dict"])

model.freeze()
model.eval()
model.to(device)


def generate_sample_from_description(description):
    # parse the given description to model input
    n_bars = len(re.findall(r"<Bar_[0-9]+>", description))
    example = make_example_from_description(description)
    batch = {key: tensor.unsqueeze(0) for key, tensor in example.items()}

    # print description and expected number of notes
    approx_n_tokens = estimate_number_of_tokens(description)
    print(f"The generated sample based on this description will contain approximately {approx_n_tokens:.0f} tokens (~{int((approx_n_tokens - 6*n_bars)/5)} notes)")
    print("=== Description ===")
    print(description.strip())
    print("===================")

    # generate a sample based on the description
    sample = model.sample(batch, max_bars=n_bars, max_length=int(approx_n_tokens * 1.2))

    # convert the generated tokens to MIDI and write it to disk
    remi_events = remi_vocab.decode(sample["sequences"][0])
    pm = remi2midi(remi_events)
    pm.write("sample.mid")

    # synthesize the generated MIDI and display it
    audio = pm.fluidsynth()
    soundfile.write("sample.wav", audio, 44100)
    return IPython.display.Audio("sample.wav")