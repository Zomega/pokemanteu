import json
import os
import random

from collections import Counter
from collections import defaultdict


class MultiMarkovModel:
    def __init__(self, order=2, default_weights=None):
        self.order = order
        self.models = {}
        self.default_weights = default_weights or {}
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"

    def train(self, model_name, data_list):
        if model_name not in self.models:
            self.models[model_name] = defaultdict(Counter)

        print(f"Training '{model_name}' on {len(data_list)} items...")

        for item in data_list:
            padded = [self.START_TOKEN] * self.order + \
                list(item) + [self.END_TOKEN]
            for i in range(len(padded) - self.order):
                state = tuple(padded[i: i + self.order])
                next_char = padded[i + self.order]
                self.models[model_name][state][next_char] += 1

    def _get_fused_probabilities(self, state, weights):
        # ... [Unchanged from previous version] ...
        active_models = {}
        for name, weight in weights.items():
            if name in self.models and weight > 0:
                counter = self.models[name].get(state, Counter())
                total = sum(counter.values())
                if total > 0:
                    active_models[name] = {
                        "counter": counter, "total": total, "weight": weight}

        if not active_models:
            return [], []

        total_active_weight = sum(m["weight"] for m in active_models.values())
        all_chars = set()
        for m in active_models.values():
            all_chars.update(m["counter"].keys())

        population = []
        probabilities = []
        for char in all_chars:
            fused_p = 0.0
            for name, m in active_models.items():
                normalized_w = m["weight"] / total_active_weight
                p_char = m["counter"][char] / m["total"]
                fused_p += normalized_w * p_char
            population.append(char)
            probabilities.append(fused_p)

        return population, probabilities

    def generate(self, weights=None, min_length=4, max_length=12):
        """Generates strings. Falls back to default_weights if none provided."""
        weights = weights or self.default_weights
        if not weights:
            raise ValueError(
                "Must provide weights or set default_weights during init.")

        current_state = tuple([self.START_TOKEN] * self.order)
        generated_chars = []

        while len(generated_chars) < max_length:
            population, probs = self._get_fused_probabilities(
                current_state, weights)
            if not population:
                break

            next_char = random.choices(population, weights=probs, k=1)[0]

            if next_char == self.END_TOKEN:
                if len(generated_chars) >= min_length:
                    break
                else:
                    return self.generate(weights, min_length, max_length)

            generated_chars.append(next_char)
            current_state = tuple(list(current_state[1:]) + [next_char])

        return "".join(generated_chars)

    def export_to_json(self, filepath):
        """Exports models and default weights to JSON."""
        export_data = {
            "order": self.order,
            "default_weights": self.default_weights,
            "models": {}
        }
        for name, transitions in self.models.items():
            export_data["models"][name] = {
                json.dumps(state): dict(counts)
                for state, counts in transitions.items()
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"Exported multi-model to {filepath}")

    @classmethod
    def load_from_json(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(order=data["order"],
                    default_weights=data.get("default_weights", {}))
        for name, saved_dict in data["models"].items():
            model.models[name] = defaultdict(Counter)
            for state_str, counts in saved_dict.items():
                state_tuple = tuple(json.loads(state_str))
                model.models[name][state_tuple] = Counter(counts)
        return model


def load_data_from_tsv(tsv_path):
    """Parses TSV and returns a list of graphemes and a list of IPAs."""
    words = []
    ipas = []

    if not os.path.exists(tsv_path):
        print(f"Warning: {tsv_path} not found.")
        return words, ipas

    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            word = parts[0].lower().strip()
            ipa_field = parts[1]

            # Handle multiple pronunciations separated by commas
            ipa_variants = ipa_field.split(',')

            for ipa_raw in ipa_variants:
                ipa = ipa_raw.replace("/", "").replace(" ", "").strip()
                if ipa:
                    words.append(word)
                    ipas.append(ipa)

    return words, ipas


# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Define Default Weights for the models
    fusion_weights = {"pokemon": 0.85, "english": 0.15}

    # 2. Initialize two separate models
    grapheme_model = MultiMarkovModel(order=2, default_weights=fusion_weights)
    phoneme_model = MultiMarkovModel(order=2, default_weights=fusion_weights)

    # 3. Load Datasets
    poke_words, poke_ipas = load_data_from_tsv("pokemon.tsv")
    eng_words, eng_ipas = load_data_from_tsv("en_US.tsv")

    # 4. Train Grapheme (Word) Model
    print("\n--- Training Grapheme Model ---")
    if poke_words:
        grapheme_model.train("pokemon", poke_words)
    if eng_words:
        grapheme_model.train("english", eng_words)

    # 5. Train Phoneme (IPA) Model
    print("\n--- Training Phoneme Model ---")
    if poke_ipas:
        phoneme_model.train("pokemon", poke_ipas)
    if eng_ipas:
        phoneme_model.train("english", eng_ipas)

    # 6. Export to separate files
    grapheme_model.export_to_json("markov_graphemes.json")
    phoneme_model.export_to_json("markov_phonemes.json")

    # 7. Generate using the embedded default_weights
    print("\n--- Generating IPA (Phonemes) ---")
    for _ in range(3):
        print(f"/{phoneme_model.generate()}/")

    print("\n--- Generating Words (Graphemes) ---")
    for _ in range(3):
        print(grapheme_model.generate())
