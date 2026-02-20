import random
from collections import defaultdict, Counter
import os
import json

class MultiMarkovModel:
    def __init__(self, order=2):
        self.order = order
        # Dictionary mapping model_name -> defaultdict(Counter)
        self.models = {} 
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"

    def train(self, model_name, ipa_words):
        """Trains a named model and adds it to the fusion pool."""
        if model_name not in self.models:
            self.models[model_name] = defaultdict(Counter)
            
        print(f"Training '{model_name}' on {len(ipa_words)} words...")
        
        for word in ipa_words:
            padded_word = [self.START_TOKEN] * self.order + list(word) + [self.END_TOKEN]
            for i in range(len(padded_word) - self.order):
                state = tuple(padded_word[i : i + self.order])
                next_char = padded_word[i + self.order]
                self.models[model_name][state][next_char] += 1

    def _get_fused_probabilities(self, state, weights):
        """Calculates weighted fusion dynamically across N models."""
        active_models = {}
        
        # 1. Find which models actually know this state
        for name, weight in weights.items():
            if name in self.models and weight > 0:
                counter = self.models[name].get(state, Counter())
                total = sum(counter.values())
                if total > 0:
                    active_models[name] = {"counter": counter, "total": total, "weight": weight}
                    
        # Dead end: no models know this state
        if not active_models:
            return [], []
            
        # 2. Re-normalize weights (Dynamic Fallback)
        # If 'poke' doesn't know the state, its weight is removed and 
        # the remaining models share 100% of the probability pool.
        total_active_weight = sum(m["weight"] for m in active_models.values())
        
        all_chars = set()
        for m in active_models.values():
            all_chars.update(m["counter"].keys())
            
        population = []
        probabilities = []
        
        # 3. Calculate fused probabilities
        for char in all_chars:
            fused_p = 0.0
            for name, m in active_models.items():
                normalized_w = m["weight"] / total_active_weight
                p_char = m["counter"][char] / m["total"]
                fused_p += normalized_w * p_char
                
            population.append(char)
            probabilities.append(fused_p)
            
        return population, probabilities

    def generate(self, weights, min_length=4, max_length=12):
        """Generates IPA strings using the provided weight dictionary."""
        current_state = tuple([self.START_TOKEN] * self.order)
        generated_chars = []
        
        while len(generated_chars) < max_length:
            population, probs = self._get_fused_probabilities(current_state, weights)
            
            if not population:
                break # Dead end
                
            next_char = random.choices(population, weights=probs, k=1)[0]
            
            if next_char == self.END_TOKEN:
                if len(generated_chars) >= min_length:
                    break
                else:
                    # Restart if too short
                    return self.generate(weights, min_length, max_length)
                    
            generated_chars.append(next_char)
            current_state = tuple(list(current_state[1:]) + [next_char])
            
        return "".join(generated_chars)

    def export_to_json(self, filepath):
        """Exports all models to a single JSON file."""
        export_data = {
            "order": self.order,
            "models": {}
        }
        
        for name, transitions in self.models.items():
            # Convert tuple keys to stringified JSON arrays
            export_data["models"][name] = {
                json.dumps(state): dict(counts) 
                for state, counts in transitions.items()
            }
            
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"Exported multi-model to {filepath}")

    @classmethod
    def load_from_json(cls, filepath):
        """Loads a multi-model from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        model = cls(order=data["order"])
        
        for name, saved_dict in data["models"].items():
            model.models[name] = defaultdict(Counter)
            for state_str, counts in saved_dict.items():
                state_tuple = tuple(json.loads(state_str))
                model.models[name][state_tuple] = Counter(counts)
                
        print(f"Loaded multi-model with datasets: {list(model.models.keys())}")
        return model


def load_ipa_from_tsv(tsv_path):
    """Helper to load IPA column from TSV."""
    ipa_list = []
    if os.path.exists(tsv_path):
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    ipa = parts[1].replace("/", "").strip()
                    if ipa:
                        ipa_list.append(ipa)
    return ipa_list


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Initialize
    multi_model = MultiMarkovModel(order=2)
    
    # 2. Train on whatever datasets you have
    poke_data = load_ipa_from_tsv("pokemon.tsv")
    eng_data = load_ipa_from_tsv("en_US.tsv")
    latin_data = load_ipa_from_tsv("en_UK.tsv") # Assume you have this!
    
    if poke_data: multi_model.train("pokemon", poke_data)
    if eng_data: multi_model.train("english", eng_data)
    if latin_data: multi_model.train("latin", latin_data)
        
    # 3. Export to JSON
    multi_model.export_to_json("multi_markov.json")
    
    # 4. Generate with Arbitrary Weights!
    print("\n--- Generating with 70% Poke, 20% English, 10% Latin ---")
    weights = {"pokemon": 0.8, "english": 0.15, "latin": 0.05}
    
    for i in range(5):
        print(f"/{multi_model.generate(weights, min_length=4, max_length=12)}/")