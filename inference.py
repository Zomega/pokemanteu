import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import json
import numpy as np
import random
from tabulate import tabulate
import tensorflow as tf

print(f"TF Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print("Physical Devices:", tf.config.list_physical_devices())
print("Using oneDNN:", tf.sysconfig.get_build_info().get('is_cuda_build'))

# 1. SETUP & LOAD
MAX_LEN = 40  # Must match the value used in training!

# Load the trained model
print("Loading model...")
model = keras.models.load_model("best_poke_model.keras") #, safe_mode=False)

@tf.function(reduce_retracing=True)
def model_step(enc_tensor, dec_input):
    return model([enc_tensor, dec_input], training=False)


# Load the vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Invert vocab (ID -> Char) for decoding
inv_vocab = {v: k for k, v in vocab.items()}

# Special Token IDs
START_TOKEN = vocab["["]
STOP_TOKEN = vocab["]"]
PAD_TOKEN = vocab["[PAD]"]


# Optimized Beam Search with Batching
def decode_sequence_beam_batched(input_text, task_token, beam_width=3):
    full_text = task_token + input_text.lower()
    enc_ids = [vocab.get(c, 0) for c in full_text]
    enc_ids = enc_ids[:MAX_LEN]
    enc_ids += [PAD_TOKEN] * (MAX_LEN - len(enc_ids))

    # Shape: (beam_width, MAX_LEN) -> e.g. (3, 40)
    enc_tensor = tf.convert_to_tensor([enc_ids] * beam_width, dtype=tf.int32)

    # 2. INITIALIZE BEAMS
    # Beams are now kept as parallel lists/arrays
    # If we start all scores at 0, we get 3 identical beams.
    scores = tf.constant([0.0] + [-1e9] * (beam_width - 1))

    # Sequences: (beam_width, 1) -> [[START], [START], [START]]
    sequences = tf.constant([[START_TOKEN]] * beam_width, dtype=tf.int32)

    # Track which beams have finished
    finished_seqs = []
    finished_scores = []

    # 3. AUTOREGRESSIVE LOOP
    for i in range(MAX_LEN - 1):
        # A. PREPARE DECODER INPUT
        # Pad current sequences to (beam_width, MAX_LEN-1)
        # We need manual padding here because TF tensor shapes are strict
        curr_len = sequences.shape[1]
        pad_size = (MAX_LEN - 1) - curr_len

        if pad_size > 0:
            paddings = tf.constant([[0, 0], [0, pad_size]])  # (batch, time)
            dec_input = tf.pad(sequences, paddings, constant_values=PAD_TOKEN)
        else:
            dec_input = sequences

        # B. BATCHED INFERENCE (One call for all beams!)
        # Output Shape: (beam_width, MAX_LEN-1, vocab_size)
        preds = model_step(enc_tensor, dec_input)

        # Get logits for the last token only: (beam_width, vocab_size)
        next_token_logits = preds[:, curr_len - 1, :]

        # Convert to Log Probs: (beam_width, vocab_size)
        log_probs = tf.nn.log_softmax(next_token_logits)

        # C. EXPAND BEAMS
        # Add current beam scores to the new log probs
        # score[b] + log_prob[b, v]
        # Shape: (beam_width, vocab_size)
        candidate_scores = tf.expand_dims(scores, axis=1) + log_probs

        # Flatten to find top K across ALL beams * ALL vocab
        # Shape: (beam_width * vocab_size,)
        flat_scores = tf.reshape(candidate_scores, [-1])

        # Top K scores and indices
        top_k_scores, top_k_indices = tf.math.top_k(flat_scores, k=beam_width)

        # D. RECONSTRUCT BEAMS
        vocab_size = log_probs.shape[-1]
        beam_indices = top_k_indices // vocab_size
        token_indices = top_k_indices % vocab_size

        next_sequences = []
        next_scores = []

        # Update sequences
        for k in range(beam_width):
            beam_idx = beam_indices[k]
            token = token_indices[k]
            score = top_k_scores[k]

            # If token is STOP, move to finished list
            if token == STOP_TOKEN:
                finished_seqs.append(sequences[beam_idx])
                finished_scores.append(score)
                dummy_seq = tf.concat([sequences[beam_idx], [PAD_TOKEN]], axis=0)
                next_sequences.append(dummy_seq)
                next_scores.append(-1e9) # Kill this beam
            else:
                new_seq = tf.concat([sequences[beam_idx], [token]], axis=0)
                next_sequences.append(new_seq)
                next_scores.append(score)
        # Stack back into tensors for next loop
        sequences = tf.stack(next_sequences)  # (beam_width, seq_len + 1)
        scores = tf.stack(next_scores)

        # Early Exit: If all active beams are terrible (very low score), stop
        if tf.reduce_max(scores) < -1e8:
            print("EARLY BREAK!", i)
            break

    # 4. SELECT BEST
    # If we have finished sequences, pick best.
    # If loop finished without STOP token, pick best active beam.
    if len(finished_scores) > 0:
        # Apply penalty: score / (len^alpha)
        # Using a simple length normalization for starters:
        # TODO: alpha as a parameter?
        normalized_scores = [
            s / (len(seq)**0.7) for s, seq in zip(finished_scores, finished_seqs)
        ]
        best_idx = np.argmax(normalized_scores)
        best_seq = finished_seqs[best_idx].numpy()
    else:
        best_idx = np.argmax(scores)
        best_seq = sequences[best_idx].numpy()

    # Decode
    decoded_chars = []
    for idx in best_seq:
        if idx == START_TOKEN:
            continue
        if idx == STOP_TOKEN:
            break
        decoded_chars.append(inv_vocab.get(idx, ""))

    return "".join(decoded_chars)


# Updated wrapper
def generate_ipa(word):
    return decode_sequence_beam_batched(word, "<", beam_width=5)

def generate_word(ipa):
    return decode_sequence_beam_batched(ipa, ">", beam_width=5)


def load_random_pokemon(file_path, n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        # Filter out empty lines
        lines = [line.strip() for line in f if line.strip()]

    # Sample first, then parse (more efficient)
    samples = random.sample(lines, min(n, len(lines)))

    pairs = []
    for line in samples:
        parts = line.split("\t")
        if len(parts) < 2:
            continue  # Skip bad lines

        word, ipa = parts[0], parts[1]

        # --- THE FIX ---
        # Remove slashes '/' and strip whitespace
        ipa = ipa.replace("/", "").strip()

        pairs.append((word, ipa))

    return pairs


samples = load_random_pokemon("pokemon.tsv", n=10)

rows = []

for word, true_ipa in samples:
    try:
        print("Processing", word, "...")
        model_ipa = generate_ipa(word)
        model_word = generate_word(true_ipa)
        round_robin = generate_word(model_ipa)

        rows.append([word, f"{true_ipa}", f"{model_ipa}", model_word, round_robin])
    except Exception as e:
        rows.append([word, f"Error: {e}", "", "", ""])

# Print table nicely
headers = ["POKEMON", "TRUE IPA", "WORD→IPA", "IPA→WORD", "ROUND-TRIP"]
print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

import gc
gc.collect()
tf.keras.backend.clear_session()