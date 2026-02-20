import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. EXTRACTION & FILTERING ---


def get_valid_ipa_chars(tsv_path):
    """Parses pokemon.tsv to find all unique IPA characters used in training."""
    valid_chars = set()
    print(f"Parsing {tsv_path} for valid IPA characters...")
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                # Get the IPA part, remove slashes
                ipa = parts[1].replace("/", "").strip()
                for char in ipa:
                    valid_chars.add(char)
    print(f"Found {len(valid_chars)} unique IPA characters.")
    return valid_chars


def load_assets_and_weights(model_path, vocab_path):
    """Loads model, vocab, and extracts raw embedding weights."""
    print("Loading model and vocabulary...")
    model = keras.models.load_model(model_path)

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Find the Embedding Layer
    emb_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Embedding) and layer.input_dim == len(vocab):
            emb_layer = layer
            break

    if not emb_layer:
        raise ValueError(
            "Could not find the correct Embedding layer. Check model.summary().")

    weights = emb_layer.get_weights()[0]
    return model, vocab, weights


def create_ipa_subset(vocab, weights, valid_chars):
    """Creates a sub-vocabulary and sub-weights matrix strictly for IPA characters."""
    ipa_vocab = {}
    ipa_indices = []

    current_idx = 0
    for char in valid_chars:
        if char in vocab:
            ipa_vocab[char] = current_idx
            ipa_indices.append(vocab[char])
            current_idx += 1

    inv_ipa_vocab = {v: k for k, v in ipa_vocab.items()}
    ipa_weights = weights[ipa_indices]

    return ipa_vocab, inv_ipa_vocab, ipa_weights

# --- 2. VISUALIZATION ---


def plot_ipa_similarity(ipa_weights, ipa_vocab, output_img="ipa_clusters.png"):
    """Generates a hierarchical clustermap using strictly IPA characters."""
    ipa_labels = list(ipa_vocab.keys())

    sim_matrix = cosine_similarity(ipa_weights)

    # Normalize to 0.0 - 1.0 range for the heatmap
    sim_matrix = (sim_matrix + 1) / 2

    print(f"Generating heatmap for {len(ipa_labels)} IPA symbols...")
    sns.set_theme(style="white")

    g = sns.clustermap(
        sim_matrix,
        xticklabels=ipa_labels,
        yticklabels=ipa_labels,
        cmap="magma",
        figsize=(14, 14),
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )

    plt.suptitle("Learned IPA Phonetic Similarity Map", fontsize=18)
    plt.savefig(output_img)
    plt.show()
    print(f"Success! Map saved to {output_img}")

# --- 3. CONFUSION MATRIX ---


def get_tunable_confusion_matrix(ipa_weights, top_k=5, sensitivity=15.0):
    """
    Creates a probability matrix for phonetic confusion strictly within the IPA subset.
    """
    print(f"Generating tunable confusion matrix (top_k={top_k}, sensitivity={sensitivity})...")
    sim_matrix = cosine_similarity(ipa_weights)

    # Mask the diagonal (set self-similarity to 0)
    np.fill_diagonal(sim_matrix, 0)

    # Top-K Masking
    masked_sim = np.full_like(sim_matrix, -np.inf)
    for i in range(len(sim_matrix)):
        top_indices = np.argsort(sim_matrix[i])[-top_k:]
        masked_sim[i, top_indices] = sim_matrix[i, top_indices]

    # Sensitivity Scaling
    scaled_sim = masked_sim * sensitivity

    # Softmax
    exp_sim = np.exp(scaled_sim - np.max(scaled_sim, axis=1, keepdims=True))
    prob_matrix = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

    return prob_matrix


def apply_confusion(word, prob_matrix, ipa_vocab, inv_ipa_vocab, mutation_rate=0.2):
    """Randomly mutates characters based strictly on the IPA confusion matrix."""
    tweaked_chars = []

    for char in word:
        # Only attempt mutation if the character is in our strict IPA subset
        if char in ipa_vocab and np.random.rand() < mutation_rate:
            idx = ipa_vocab[char]
            probs = prob_matrix[idx]

            # Sample a new IPA character based on the probabilities
            new_idx = np.random.choice(len(probs), p=probs)
            tweaked_chars.append(inv_ipa_vocab[new_idx])
        else:
            # Keep original (preserves spaces, unmutated IPA, or standard English letters if present)
            tweaked_chars.append(char)

    return "".join(tweaked_chars)


# --- 4. EXECUTION ---

if __name__ == "__main__":
    MODEL_FILE = "best_poke_model.keras"
    VOCAB_FILE = "vocab.json"
    TSV_FILE = "pokemon.tsv"

    if not (os.path.exists(MODEL_FILE) and os.path.exists(VOCAB_FILE) and os.path.exists(TSV_FILE)):
        print(f"Error: Ensure {MODEL_FILE}, {VOCAB_FILE}, and {TSV_FILE} exist in the current directory.")
    else:
        # 1. Parse TSV for valid IPA characters
        valid_ipa_chars = get_valid_ipa_chars(TSV_FILE)

        # 2. Load model assets
        model, vocab, weights = load_assets_and_weights(MODEL_FILE, VOCAB_FILE)

        # 3. Create strict IPA subset
        ipa_vocab, inv_ipa_vocab, ipa_weights = create_ipa_subset(
            vocab, weights, valid_ipa_chars)

        # 4. Generate and save the plot
        plot_ipa_similarity(ipa_weights, ipa_vocab)

        # 5. Create the confusion matrix
        prob_matrix = get_tunable_confusion_matrix(
            ipa_weights, top_k=3, sensitivity=15.0)

        # 6. Demonstrate the "Pronunciation Tweaker"
        original_ipa = "pɪkətʃu"  # Example: Pikachu
        print(f"\nTesting Confusion Matrix on: '{original_ipa}' (Mutation Rate: 100%)")

        variants_left = 20
        variants = set()
        while len(variants) < 20:
            # Set mutation rate high just to see the matrix at work!
            tweaked_ipa = apply_confusion(
                original_ipa, prob_matrix, ipa_vocab, inv_ipa_vocab, mutation_rate=0.05)
            if tweaked_ipa == original_ipa:
                continue
            if tweaked_ipa in variants:
                continue
            variants.add(tweaked_ipa)

        for variant in sorted(variants):
            print(f"Variant: {variant}")
