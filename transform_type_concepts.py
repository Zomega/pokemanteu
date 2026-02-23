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

    # 2. Run the pipeline
    try:
        # Pass the list of files instead of a single string
        combined_ipa_dict = load_ipa_dictionaries(TSV_FILES)
        convert_txt_to_json(
            INPUT_DIRECTORY, OUTPUT_DIRECTORY, combined_ipa_dict)
        print("\nAll files processed successfully!")
    except Exception as e:
        print(f"Error: {e}")
