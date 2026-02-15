import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import json
import numpy as np
#import tensorflowjs as tfjs

# 1. DYNAMIC VOCABULARY BUILDER


def build_vocab_from_files(file_paths):
    # Start with standard special tokens
    unique_chars = set(["<", ">", "[", "]"])
    print("Building Global Vocabulary...")

    for fpath in file_paths:
        if not os.path.exists(fpath):
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue

                word_raw = parts[0]
                ipa_field = parts[1]

                # Handle comma-separated variants
                ipa_variants = ipa_field.split(',')

                word = word_raw.lower().strip()
                unique_chars.update(list(word))

                for ipa_var in ipa_variants:
                    ipa_clean = ipa_var.replace("/", "").replace(" ", "").strip()
                    unique_chars.update(list(ipa_clean))

    # --- THE FIX IS HERE ---
    # Ensure [PAD] is NOT in the set we are about to enumerate.
    # We want to manually assign it to 0 later.
    if "[PAD]" in unique_chars:
        unique_chars.remove("[PAD]")

    sorted_chars = sorted(list(unique_chars))

    # Assign indices 1, 2, 3... N
    vocab = {char: i + 1 for i, char in enumerate(sorted_chars)}

    # Manually assign PAD to 0
    vocab["[PAD]"] = 0

    # Now len(vocab) will match exactly (Max_Index + 1)
    print(f"  Global Vocabulary Size: {len(vocab)}")

    inv_vocab = {i: char for char, i in vocab.items()}

    return vocab, inv_vocab



def encode(text, vocab, max_len):
    ids = [vocab.get(c, vocab["[PAD]"]) for c in text]  # Use .get() to be safe
    ids = ids[:max_len]
    return ids + [0] * (max_len - len(ids))

# 2. DATA LOADING (Consolidated)


def prepare_multitask_data(file_paths, vocab, max_len):
    encoder_inputs, decoder_inputs, decoder_targets = [], [], []

    print("Processing data...")
    for fpath in file_paths:
        if not os.path.exists(fpath):
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            word = parts[0].lower().strip()
            ipa_field = parts[1]

            # 1. SPLIT by comma to handle multiple pronunciations
            # "ˈɛrənsən, ˈɑːrənsən" -> ["ˈɛrənsən", " ˈɑːrənsən"]
            ipa_variants = ipa_field.split(',')

            # 2. Iterate over each variant and create a training pair
            for ipa_raw in ipa_variants:
                ipa = ipa_raw.replace("/", "").replace(" ", "").strip()

                # Skip empty strings (in case of trailing commas)
                if not ipa:
                    continue

                # DIRECTION 1: English -> IPA
                src_g2p = f"<{word}"
                tgt_g2p = f"[{ipa}]"

                # DIRECTION 2: IPA -> English
                # (Both IPA variants map back to the same English word)
                src_p2g = f">{ipa}"
                tgt_p2g = f"[{word}]"

                for src, tgt in [(src_g2p, tgt_g2p), (src_p2g, tgt_p2g)]:
                    e_idx = encode(src, vocab, max_len)
                    t_idx = encode(tgt, vocab, max_len)

                    encoder_inputs.append(e_idx)
                    decoder_inputs.append(t_idx[:-1])
                    decoder_targets.append(t_idx[1:])

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
    pos_emb_enc = layers.Embedding(max_len, embed_dim)
    positions_enc = keras.ops.arange(0, max_len, dtype="int32")
    pos_emb = pos_emb_enc(positions_enc)
    pos_emb = keras.ops.expand_dims(pos_emb, axis=0)

    x = token_emb + pos_emb
    x = transformer_block(x, embed_dim, num_heads, ff_dim, rate)
    enc_out = x

    # --- DECODER ---
    dec_in = keras.Input(shape=(max_len-1,), name="dec_in")

    token_emb_dec = layers.Embedding(
        vocab_size, embed_dim, mask_zero=True)(dec_in)
    pos_emb_dec_layer = layers.Embedding(max_len - 1, embed_dim)
    positions_dec = keras.ops.arange(0, max_len - 1, dtype="int32")
    pos_emb_dec = pos_emb_dec_layer(positions_dec)
    pos_emb_dec = keras.ops.expand_dims(pos_emb_dec, axis=0)

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

FILES = [
    "cmudict-0.7b-ipa.tsv",
    "eng_latn_uk_broad.tsv",
    "eng_latn_uk_broad_filtered.tsv",
    "eng_latn_uk_narrow.tsv",
    "eng_latn_us_broad.tsv",
    "eng_latn_us_broad_filtered.tsv",
    "eng_latn_us_narrow.tsv",
    "en_UK.tsv",
    "en_US.tsv",
    "pokemon.tsv",
]
MAX_SEQ = 40

# Build vocab
vocab, inv_vocab = build_vocab_from_files(FILES)
with open("vocab.json", "w") as f:
    json.dump(vocab, f)

(x_enc, x_dec), y_tgt = prepare_multitask_data(FILES, vocab, max_len=MAX_SEQ)

model = build_transformer(len(vocab), MAX_SEQ)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    # 1. Stop if validation loss doesn't improve for 2 epochs
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    ),
    # 2. Save the best model to disk immediately whenever we hit a new record
    keras.callbacks.ModelCheckpoint(
        filepath="best_poke_model.keras",
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
]

model.fit([x_enc, x_dec], y_tgt,
          batch_size=64,
          epochs=30,
          validation_split=0.1,
          callbacks=callbacks)

# 6. SAVE FOR JS
model.save("poke_model_final.keras")

# Load the trained .keras file
model = keras.models.load_model("best_poke_model.keras")

# Export to TF.js format (requires 'pip install tensorflowjs')
# This creates a folder 'tfjs_model' you can upload to your web server
tfjs.converters.save_keras_model(model, "tfjs_model")
