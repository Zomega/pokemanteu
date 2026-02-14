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


def encode_input(text):
    """Prepares the input word for the Encoder."""
    # Add the Task Token: "<" means English -> IPA
    text = "<" + text

    # Convert to IDs
    ids = [vocab.get(c, 0) for c in text.lower()]

    # Truncate to MAX_LEN
    ids = ids[:MAX_LEN]

    # Pad to MAX_LEN
    return ids + [PAD_TOKEN] * (MAX_LEN - len(ids))


def generate_ipa(word):
    """
    Runs the Transformer in an autoregressive loop to generate IPA.
    """
    # 1. Prepare Encoder Input (Shape: 1 x 40)
    enc_in = np.array([encode_input(word)])

    # 2. Initialize Decoder Input with [START] token
    # We will append predicted tokens to this list
    current_ids = [START_TOKEN]

    # 3. Autoregressive Loop
    # We loop up to MAX_LEN - 1 times (since input is length 40, output is 39)
    for i in range(MAX_LEN - 1):
        # Pad the current decoder input to the fixed length (39)
        # expected by the model
        padded_dec_in = current_ids + \
            [PAD_TOKEN] * (MAX_LEN - 1 - len(current_ids))
        dec_in_tensor = np.array([padded_dec_in])

        # Run Prediction
        # Returns shape: (1, 39, vocab_size)
        preds = model.predict([enc_in, dec_in_tensor], verbose=0)

        # Get the logits for the LAST valid token step (index i)
        # This effectively asks: "Given the sequence so far, what is next?"
        next_token_logits = preds[0, i, :]

        # Pick the most likely token (Greedy Search)
        # For more variety, you could use random sampling here
        next_id = np.argmax(next_token_logits)

        # If model predicts [STOP], we are done
        if next_id == STOP_TOKEN:
            break

        # Append to our sequence and continue
        current_ids.append(next_id)

    # 4. Decode IDs back to Text
    # Skip the first token (START_TOKEN)
    ipa_chars = [inv_vocab.get(idx, "") for idx in current_ids[1:]]
    return "".join(ipa_chars)


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

print("-" * 30)
print(f"{'POKEMON':<15} | {'IPA PRONUNCIATION'}")
print("-" * 30)

for name in pokemon_names:
    try:
        ipa = generate_ipa(name)
        print(f"{name:<15} | /{ipa}/")
    except Exception as e:
        print(f"Error generating for {name}: {e}")
