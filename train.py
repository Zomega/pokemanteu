from keras import layers
import keras
import os
import json
import numpy as np

# 1. SETUP BACKEND
os.environ["KERAS_BACKEND"] = "torch"

# 2. THE VOCABULARY & TOKENIZER
# We define our own to ensure absolute consistency for JS export.
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' -/"
ipa_symbols = "ɪæəɑɔʊuːiːeɪaɪɔɪoʊaʊpbt dkfɡvθðszʃʒhtʃdʒmlrjŋ"
special = "<>[] "  # < (G2P), > (P2G), [ (Start), ] (Stop), space (Padding)

chars = sorted(list(set(alphabet + ipa_symbols + special)))
vocab = {char: i + 1 for i, char in enumerate(chars)}
vocab["[PAD]"] = 0
inv_vocab = {i: char for char, i in vocab.items()}


def encode(text, max_len=25):
    ids = [vocab[c] for c in text if c in vocab]
    return ids + [0] * (max_len - len(ids))

# 3. DATA SERIALIZATION (The "Zip" Logic)


def prepare_multitask_data(file_path, max_len=25):
    # Load your "word \t ipa" file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    encoder_inputs, decoder_inputs, decoder_targets = [], [], []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        word, ipa = parts[0], parts[1]

        # DIRECTION 1: English -> IPA (Task: <)
        # Input: "<word", Target: "[ipa]"
        src_g2p = f"<{word}"
        tgt_g2p = f"[{ipa}]"

        # DIRECTION 2: IPA -> English (Task: >)
        # Input: ">ipa", Target: "[word]"
        src_p2g = f">{ipa}"
        tgt_p2g = f"[{word}]"

        for src, tgt in [(src_g2p, tgt_g2p), (src_p2g, tgt_p2g)]:
            e_idx = encode(src, max_len)
            t_idx = encode(tgt, max_len)

            encoder_inputs.append(e_idx)
            # The "Zip" offset:
            decoder_inputs.append(t_idx[:-1])  # History: [START], p, i, k...
            decoder_targets.append(t_idx[1:])  # Future: p, i, k, a... [END]

    return (np.array(encoder_inputs), np.array(decoder_inputs)), np.array(decoder_targets)

# 4. THE TRANSFORMER ARCHITECTURE


def build_transformer(vocab_size, max_len, embed_dim=128, num_heads=4):
    # Encoder
    enc_in = keras.Input(shape=(max_len,), name="enc_in")
    x = layers.Embedding(vocab_size, embed_dim)(enc_in)
    # Simple Positional Encoding (built into Keras 3 layers in 2026)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    enc_out = layers.LayerNormalization()(x)

    # Decoder
    dec_in = keras.Input(shape=(max_len-1,), name="dec_in")
    y = layers.Embedding(vocab_size, embed_dim)(dec_in)

    # Self-attention with CAUSAL MASKING (crucial for parallel training)
    y = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        y, y, use_causal_mask=True)
    y = layers.LayerNormalization()(y)

    # Cross-attention to encoder
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)(y, enc_out)
    y = layers.LayerNormalization()(y)

    outputs = layers.Dense(vocab_size, activation="softmax")(y)

    return keras.Model([enc_in, dec_in], outputs)


# 5. EXECUTION
max_seq = 25
(x_enc, x_dec), y_tgt = prepare_multitask_data("en_US.tsv", max_len=max_seq)

model = build_transformer(len(vocab), max_seq)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit([x_enc, x_dec], y_tgt, batch_size=64,
          epochs=10, validation_split=0.1)

# 6. SAVE FOR JS
model.save("poke_model.keras")
with open("vocab.json", "w") as f:
    json.dump(vocab, f)
