import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import layers
import keras
import json
import numpy as np


# 2. THE VOCABULARY & TOKENIZER
# We define our own to ensure absolute consistency for JS export.
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' -/"
ipa_symbols = "ɪæəɑɔʊuːiːeɪaɪɔɪoʊaʊpbt dkfɡvθðszʃʒhtʃdʒmlrjŋ"
special = "<>[] "  # < (G2P), > (P2G), [ (Start), ] (Stop), space (Padding)

chars = sorted(list(set(alphabet + ipa_symbols + special)))
vocab = {char: i + 1 for i, char in enumerate(chars)}
vocab["[PAD]"] = 0
inv_vocab = {i: char for char, i in vocab.items()}


def encode(text, max_len=40):
    ids = [vocab[c] for c in text if c in vocab]
    # 1. Truncate to ensure it never exceeds max_len
    ids = ids[:max_len]
    # 2. Pad (this is now safe)
    return ids + [0] * (max_len - len(ids))

# 3. DATA SERIALIZATION (The "Zip" Logic)


def prepare_multitask_data(file_path, max_len=40):
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


def transformer_block(x, embed_dim, num_heads, ff_dim, rate=0.1):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = layers.Dropout(rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_transformer(vocab_size, max_len, embed_dim=128, num_heads=4, ff_dim=128, rate=0.1):
    # --- ENCODER ---
    enc_in = keras.Input(shape=(max_len,), name="enc_in")

    # Embeddings (Masking enabled)
    token_emb = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(enc_in)
    positions = layers.Lambda(lambda x: keras.ops.arange(
        0, max_len, dtype="int32"))(enc_in)
    pos_emb = layers.Embedding(max_len, embed_dim)(positions)

    x = token_emb + pos_emb
    x = transformer_block(x, embed_dim, num_heads, ff_dim, rate)
    enc_out = x

    # --- DECODER ---
    dec_in = keras.Input(shape=(max_len-1,), name="dec_in")

    token_emb_dec = layers.Embedding(
        vocab_size, embed_dim, mask_zero=True)(dec_in)
    positions_dec = layers.Lambda(lambda x: keras.ops.arange(
        0, max_len-1, dtype="int32"))(dec_in)
    pos_emb_dec = layers.Embedding(max_len, embed_dim)(positions_dec)

    y = token_emb_dec + pos_emb_dec

    # 1. Self-Attention (Causal)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        y, y, use_causal_mask=True)
    attn_output = layers.Dropout(rate)(attn_output)  # Added Dropout
    y = layers.LayerNormalization(epsilon=1e-6)(y + attn_output)

    # 2. Cross-Attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        y, enc_out)
    attn_output = layers.Dropout(rate)(attn_output)  # Added Dropout
    y = layers.LayerNormalization(epsilon=1e-6)(y + attn_output)

    # 3. Feed-Forward
    ffn_output = layers.Dense(ff_dim, activation="relu")(y)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)  # Added Dropout
    y = layers.LayerNormalization(epsilon=1e-6)(y + ffn_output)

    outputs = layers.Dense(vocab_size, activation="softmax")(y)

    return keras.Model([enc_in, dec_in], outputs)


# 5. EXECUTION
max_seq = 40
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


# Reload the model using the TensorFlow backend to export for JS
os.environ["KERAS_BACKEND"] = "tensorflow" 
import keras

# Load the trained .keras file
model = keras.models.load_model("poke_model.keras")

# Export to TF.js format (requires 'pip install tensorflowjs')
# This creates a folder 'tfjs_model' you can upload to your web server
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "tfjs_model")