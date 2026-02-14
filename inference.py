import os

# MUST match the backend used in training
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import json
import numpy as np

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


# --- MAIN EXECUTION ---
pokemon_names = [
    "Pikachu",
    "Bulbasaur",
    "Charizard",
    "Mewtwo",
    "Gyarados",
    "Rayquaza",
    "Sudowoodo",
    "Gardevoir"
]

print("-" * 60)
print(f"{'POKEMON':<15} | {'IPA':<20} | {'BACK TO WORD'}")
print("-" * 60)

for name in pokemon_names:
    try:
        ipa = generate_ipa(name)
        back = generate_word(ipa)

        print(f"{name:<15} | /{ipa:<18}/ | {back}")
    except Exception as e:
        print(f"Error generating for {name}: {e}")

