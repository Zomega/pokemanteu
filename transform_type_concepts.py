import yaml
import os
import glob
import json


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


def convert_txt_to_json(input_dir, output_dir, ipa_map):
    """Iterates through .txt files and converts them to .json."""
    os.makedirs(output_dir, exist_ok=True)

    search_pattern = os.path.join(input_dir, '*.txt')
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f"No .txt files found in '{input_dir}'!")
        return

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        out_name = os.path.splitext(filename)[0] + '.json'
        out_path = os.path.join(output_dir, out_name)

        word_data = []
        missing_words = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if not word:
                    continue

                ipa = ipa_map.get(word)
                if not ipa:
                    missing_words.append(word)

                word_data.append({
                    "word": word,
                    "ipa": ipa
                })

        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(word_data, out_f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path} ({len(word_data)} words)")
        if missing_words:
            print(f"  -> Missing IPA for {len(missing_words)} words: {', '.join(missing_words)}")


def parse_label(label, ipa_dict, missing_words):
    """Parses a single YAML string into a structured JSON object."""
    # 1. Extract Tags and lowercase them
    parts = label.split(':')
    raw_word = parts[0].strip()
    tags = [t.strip().lower() for t in parts[1:] if t.strip()]

    silent = False
    unsearchable = False
    transparent = False

    # 2. Extract Visibility Flags (_ and __)
    if raw_word.startswith('__'):
        unsearchable = True
        silent = True
        raw_word = raw_word[2:]
    elif raw_word.startswith('_'):
        silent = True
        raw_word = raw_word[1:]

    # 3. Extract Modifiers ($ and ^)
    if raw_word.startswith('$'):
        unsearchable = True
        raw_word = raw_word[1:]
    elif raw_word.startswith('^'):
        transparent = True
        raw_word = raw_word[1:]

    # 4. CONDITIONAL CASING: UPPERCASE for silent, lowercase for normal
    base_word = raw_word.strip()
    if silent and unsearchable:
        clean_word = base_word.upper()
    else:
        clean_word = base_word.lower()

    # 5. Lookup IPA (Skip if silent!)
    ipa = None
    if not silent:
        ipa = ipa_dict.get(clean_word)
        if not ipa:
            missing_words.append(clean_word)

    # 6. Build the clean JSON node
    node = {
        "word": clean_word,
    }

    # Only append these if they are relevant to keep the JSON incredibly clean
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


def process_node(node, ipa_dict, missing_words):
    """Recursively walks the parsed YAML data."""
    # Leaf node (just a string like "- Egg")
    if isinstance(node, str):
        return parse_label(node, ipa_dict, missing_words)

    # Branch node (a dictionary with a parent key and child list)
    elif isinstance(node, dict):
        results = []
        for key, value in node.items():
            parsed_node = parse_label(key, ipa_dict, missing_words)

            children = []
            if isinstance(value, list):
                for child in value:
                    children.append(process_node(
                        child, ipa_dict, missing_words))
            elif value is not None:
                children.append(process_node(value, ipa_dict, missing_words))

            parsed_node["children"] = children
            results.append(parsed_node)

        # Unpack if it's a single-key dictionary (which YAML list-dicts usually are)
        return results[0] if len(results) == 1 else results

    # List of nodes
    elif isinstance(node, list):
        return [process_node(child, ipa_dict, missing_words) for child in node]


# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Define your paths here
    # Order matters! The first file gets priority for pronunciations.
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
    OUTPUT_DIRECTORY = "pokemon_type_concepts"

    YAML_FILE = "pokemon_type_concepts/creatures.yaml"
    JSON_OUTPUT = "pokemon_type_concepts/creatures.json"

    # 2. Run the pipeline
    print("Loading dictionaries...")
    ipa_dict = load_ipa_dictionaries(TSV_FILES)
    convert_txt_to_json(
        INPUT_DIRECTORY, OUTPUT_DIRECTORY, ipa_dict)
    print("\nAll type files processed successfully!")

    print(f"\nParsing {YAML_FILE}...")
    with open(YAML_FILE, 'r', encoding='utf-8') as f:
        # safe_load converts the YAML string into nested Python lists/dicts
        yaml_data = yaml.safe_load(f)

    missing_words = []

    # Process the tree
    json_tree = process_node(yaml_data, ipa_dict, missing_words)

    # Export
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(json_tree, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved cleanly formatted tree to {JSON_OUTPUT}")

    # Deduplicate and print missing words for you to add to a custom TSV
    if missing_words:
        unique_missing = sorted(list(set(missing_words)))
        print(f"\nWARNING: Could not find IPAs for {len(unique_missing)} words:")
        print("\n".join(unique_missing))
