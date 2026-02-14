import os

# MUST match the backend used in training
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import json
import numpy as np
import random
from tabulate import tabulate

# 1. SETUP & LOAD
MAX_LEN = 40  # Must match the value used in training!

# Load the trained model
print("Loading model...")
model = keras.models.load_model("best_poke_model.keras") #, safe_mode=False)

# Load the vocabulary
with open("vocab.json", "r") as f:
    vocab = json.load(f)

# Invert vocab (ID -> Char) for decoding
inv_vocab = {v: k for k, v in vocab.items()}

# Special Token IDs
START_TOKEN = vocab["["]
STOP_TOKEN = vocab["]"]
PAD_TOKEN = vocab["[PAD]"]


def encode_input(text, task_token):
    """
    Prepares the input for the encoder.
    task_token:
        "<" = English → IPA
        ">" = IPA → English
    """
    text = task_token + text

    ids = [vocab.get(c, 0) for c in text.lower()]
    ids = ids[:MAX_LEN]

    return ids + [PAD_TOKEN] * (MAX_LEN - len(ids))


# TODO: Beam Search instead.
def generate(text, task_token):
    """
    Autoregressive generation.
    task_token:
        "<" = English → IPA
        ">" = IPA → English
    """
    enc_in = np.array([encode_input(text, task_token)])

    current_ids = [START_TOKEN]

    for i in range(MAX_LEN - 1):
        padded_dec_in = current_ids + \
            [PAD_TOKEN] * (MAX_LEN - 1 - len(current_ids))

        dec_in_tensor = np.array([padded_dec_in])

        preds = model.predict([enc_in, dec_in_tensor], verbose=0)

        next_token_logits = preds[0, i, :]
        next_id = np.argmax(next_token_logits)

        if next_id == STOP_TOKEN:
            break

        current_ids.append(next_id)

    decoded_chars = [inv_vocab.get(idx, "") for idx in current_ids[1:]]

    return "".join(decoded_chars)

def generate_ipa(word):
    return generate(word, "<")


def generate_word(ipa):
    return generate(ipa, ">")


def load_random_pokemon(file_path, n=10):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    samples = random.sample(lines, min(n, len(lines)))

    pairs = []
    for line in samples:
        word, ipa = line.split("\t")

        ipa = ipa.strip()

        pairs.append((word, ipa))

    return pairs

samples = load_random_pokemon("pokemon.tsv", n=30)

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
