import yaml
import os
import glob
import json
import torch
import clip


def load_ipa_dictionaries(tsv_paths):
    """
    Loads multiple TSVs into a single fast lookup dictionary.
    Files are processed in order; the first file to define a word wins.
    """
    ipa_map = {}

    for tsv_path in tsv_paths:
        if not os.path.exists(tsv_path):
            print(f"Warning: Could not find {tsv_path}. Skipping.")
            continue

        print(f"Loading dictionary from {tsv_path}...")
        added_from_file = 0

        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0].lower().strip()

                    # Only add if we haven't seen this word in an earlier TSV
                    if word not in ipa_map:
                        ipa_variants = parts[1].split(',')
                        clean_ipa = ipa_variants[0].replace(
                            "/", "").replace(" ", "").strip()

                        if clean_ipa:
                            ipa_map[word] = clean_ipa
                            added_from_file += 1

        print(f"  -> Added {added_from_file} new words from this file.")

    print(f"\nFinished building dictionary. Total unique words: {len(ipa_map)}\n")
    return ipa_map


def build_pokemon_types_tree(input_dir, ipa_map, missing_words_set):
    """
    Reads .txt files, parses their contents, and builds a hierarchical
    dictionary branch to be attached to the main YAML tree.
    """
    types_node = {
        "word": "POKEMON_TYPES",
        "silent": True,
        "unsearchable": True,
        "children": []
    }

    search_pattern = os.path.join(input_dir, '*.txt')
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f"Warning: No .txt files found in '{input_dir}'!")
        return types_node

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        # e.g., 'water' -> 'WATER'
        type_name = os.path.splitext(filename)[0].upper()

        # Create the specific type node (e.g., __WATER)
        type_node = {
            "word": type_name,
            "silent": True,
            "unsearchable": True,
            "pokemon_type": True,  # Added specific tag
            "children": []
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue

                # Pass each word through the exact same parser used by the YAML
                child_node = parse_label(word, ipa_map, missing_words_set)
                type_node["children"].append(child_node)

        if type_node["children"]:
            types_node["children"].append(type_node)
            print(f"Loaded {len(type_node['children'])} concepts for type: {type_name}")

    return types_node


def parse_label(label, ipa_dict, missing_words_set):
    """Parses a single YAML string into a structured JSON object."""
    parts = label.split(':')
    raw_word = parts[0].strip()
    tags = [t.strip().lower() for t in parts[1:] if t.strip()]

    silent = False
    unsearchable = False
    transparent = False

    # Extract Visibility Flags
    if raw_word.startswith('__'):
        unsearchable = True
        silent = True
        raw_word = raw_word[2:]
    elif raw_word.startswith('_'):
        silent = True
        raw_word = raw_word[1:]

    # Extract Modifiers
    if raw_word.startswith('$'):
        unsearchable = True
        raw_word = raw_word[1:]
    elif raw_word.startswith('^'):
        transparent = True
        raw_word = raw_word[1:]

    # Conditional Casing
    base_word = raw_word.strip()
    if silent and unsearchable:
        clean_word = base_word.upper()
    else:
        clean_word = base_word.lower()

    # Lookup IPA
    ipa = None
    if not silent:
        ipa = ipa_dict.get(clean_word)
        if not ipa:
            # Using .add() because it's a set now
            missing_words_set.add(clean_word)

    # Build Node
    node = {
        "word": clean_word,
    }

    if silent:
        node["silent"] = True
    if unsearchable:
        node["unsearchable"] = True
    if transparent:
        node["transparent"] = True
    if not silent:
        node["ipa"] = ipa
    if tags:
        node["tags"] = tags

    return node


def process_node(node, ipa_dict, missing_words_set):
    """Recursively walks the parsed YAML data."""
    if isinstance(node, str):
        return parse_label(node, ipa_dict, missing_words_set)

    elif isinstance(node, dict):
        results = []
        for key, value in node.items():
            parsed_node = parse_label(key, ipa_dict, missing_words_set)

            children = []
            if isinstance(value, list):
                for child in value:
                    children.append(process_node(
                        child, ipa_dict, missing_words_set))
            elif value is not None:
                children.append(process_node(
                    value, ipa_dict, missing_words_set))

            parsed_node["children"] = children
            results.append(parsed_node)

        return results[0] if len(results) == 1 else results

    elif isinstance(node, list):
        return [process_node(child, ipa_dict, missing_words_set) for child in node]


def add_clip_embeddings(node, model, device):
    """
    Recursively computes and attaches a 512-dimensional CLIP embedding
    to any node that is searchable.
    """
    # Only spend compute on words that will actually be searched!
    if node.get("word") and not node.get("unsearchable", False):
        with torch.no_grad():
            # Tokenize and run the model
            text_tokens = clip.tokenize([node["word"]]).to(device)
            embedding = model.encode_text(text_tokens)

            # Convert PyTorch tensor to a standard Python list of floats for JSON
            # .squeeze() removes the batch dimension [1, 512] -> [512]
            node["vector"] = embedding.squeeze().cpu().tolist()

    # Dig into the children
    for child in node.get("children", []):
        add_clip_embeddings(child, model, device)


# --- EXECUTION ---
if __name__ == "__main__":
    TSV_FILES = [
        "custom.tsv",
        "en_US.tsv",
        "cmudict-0.7b-ipa.tsv",
        "eng_latn_us_broad.tsv",
        "eng_latn_us_broad_filtered.tsv",
        "eng_latn_us_narrow.tsv",
        "eng_latn_uk_broad.tsv",
        "eng_latn_uk_broad_filtered.tsv",
        "eng_latn_uk_narrow.tsv",
        "en_UK.tsv",
        "pokemon.tsv",
    ]

    INPUT_DIRECTORY = "pokemon_type_concepts"
    YAML_FILE = "pokemon_type_concepts/creatures.yaml"
    JSON_OUTPUT = "pokemon_type_concepts/creatures.json"

    # Use a SET so words are globally deduplicated
    global_missing_words = set()

    print("Loading dictionaries...")
    ipa_dict = load_ipa_dictionaries(TSV_FILES)

    # 1. Parse the main YAML tree
    print(f"Parsing {YAML_FILE}...")
    with open(YAML_FILE, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)

    json_tree = process_node(yaml_data, ipa_dict, global_missing_words)

    # 2. Parse the .txt files into a unified branch
    print(f"\nParsing text files from {INPUT_DIRECTORY}...")
    pokemon_types_branch = build_pokemon_types_tree(
        INPUT_DIRECTORY, ipa_dict, global_missing_words)

    # 3. Inject the .txt branch into the ROOT node of the YAML
    if isinstance(json_tree, dict) and json_tree.get("word") == "ROOT":
        json_tree.setdefault("children", []).append(pokemon_types_branch)

    # 4. Compute CLIP Embeddings
    print("\nLoading PyTorch CLIP model for text embeddings...")
    device = "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    print("Computing 512-dimensional semantic vectors for searchable nodes...")
    add_clip_embeddings(json_tree, clip_model, device)

    # 5. Export everything
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(json_tree, f, ensure_ascii=False, indent=2)

    print(f"\nSuccessfully saved unified tree with CLIP vectors to {JSON_OUTPUT}")

    # 6. Print out the unified missing words
    if global_missing_words:
        unique_missing = sorted(list(global_missing_words))
        print(f"\nWARNING: Could not find IPAs for {len(unique_missing)} words across all files:")
        print("\n".join(unique_missing))
