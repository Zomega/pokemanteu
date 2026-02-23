import os
import glob
import json


def load_ipa_dictionary(tsv_path):
    """Loads the TSV into a fast lookup dictionary."""
    ipa_map = {}
    print(f"Loading dictionary from {tsv_path}...")

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Could not find {tsv_path}")

    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                word = parts[0].lower().strip()

                # We handle multiple pronunciations just like your seq2seq script
                ipa_variants = parts[1].split(',')

                # Clean the first available IPA string
                clean_ipa = ipa_variants[0].replace(
                    "/", "").replace(" ", "").strip()
                if clean_ipa:
                    ipa_map[word] = clean_ipa

    print(f"Loaded {len(ipa_map)} words into the dictionary.")
    return ipa_map


def convert_txt_to_json(input_dir, output_dir, ipa_map):
    """Iterates through .txt files and converts them to .json."""
    # Ensure our output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Grab all .txt files in the input directory
    search_pattern = os.path.join(input_dir, '*.txt')
    txt_files = glob.glob(search_pattern)

    if not txt_files:
        print(f"No .txt files found in '{input_dir}'!")
        return

    for filepath in txt_files:
        filename = os.path.basename(filepath)
        # Swap the .txt extension for .json
        out_name = os.path.splitext(filename)[0] + '.json'
        out_path = os.path.join(output_dir, out_name)

        word_data = []
        missing_words = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if not word:
                    continue  # Skip blank lines

                # Look up the IPA, default to None if missing
                ipa = ipa_map.get(word)

                if not ipa:
                    missing_words.append(word)

                word_data.append({
                    "word": word,
                    "ipa": ipa
                })

        # Save the JSON file
        with open(out_path, 'w', encoding='utf-8') as out_f:
            json.dump(word_data, out_f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path} ({len(word_data)} words)")

        # Helpful logging for missing words
        if missing_words:
            print(f"  -> Missing IPA for {len(missing_words)} words in this file: {', '.join(missing_words)}")


# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Define your paths here
    TSV_FILE = "en_US.tsv"
    INPUT_DIRECTORY = "pokemon_type_concepts"
    OUTPUT_DIRECTORY = "pokemon_type_concepts"

    # 2. Run the pipeline
    try:
        ipa_dict = load_ipa_dictionary(TSV_FILE)
        convert_txt_to_json(INPUT_DIRECTORY, OUTPUT_DIRECTORY, ipa_dict)
        print("\nAll files processed successfully!")
    except Exception as e:
        print(f"Error: {e}")
