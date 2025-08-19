# hero_data_loader.py
# This module is responsible for loading all raw data from disk (CSVs and JSONs).

import csv
import json
import re
from pathlib import Path
import glob
import os
import pandas as pd

# --- Constants ---
try:
    SCRIPT_DIR = Path(__file__).parent
    # This path is fixed as per the project specification.
    DATA_DIR = Path("D:/RED")
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive notebooks)
    SCRIPT_DIR = Path.cwd()
    DATA_DIR = Path("D:/RED")
    print(f"Warning: '__file__' not found. Assuming script dir is {SCRIPT_DIR}")

# --- File Paths ---
HERO_STATS_CSV_PATTERN = "_private_heroes_*.csv"
CSV_EN_PATH = DATA_DIR / "English.csv"
CSV_JA_PATH = DATA_DIR / "Japanese.csv"
JSON_OVERRIDE_PATH = DATA_DIR / "languageOverrides.json"
CHARACTERS_PATH = DATA_DIR / "characters.json"
SPECIALS_PATH = DATA_DIR / "specials.json"
BATTLE_PATH = DATA_DIR / "battle.json"


def load_rules_from_csvs(script_dir: Path) -> dict:
    """
    Loads override rules from two CSV files:
    - exception_lang_rules.csv: For overriding lang_ids.
    - exception_hero_rules.csv: For resolving placeholders.
    """
    print("--- Loading Exception Rules from CSVs ---")
    rules = {
        "lang_overrides": {"specific": {}, "common": {}},
        "hero_rules": {"specific": {}, "common": {}}
    }
    
    # --- Load Language ID Overrides ---
    lang_rules_path = script_dir / "exception_lang_rules.csv"
    if not lang_rules_path.exists():
        print(f"Info: '{lang_rules_path.name}' not found. No language rules loaded.")
    else:
        try:
            with open(lang_rules_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    hero_id = row.get("hero_id", "").strip()
                    skill_id = row.get("skill_id", "").strip()
                    lang_id = row.get("lang_id", "").strip()

                    if not (skill_id and lang_id): continue
                    
                    if hero_id: # Specific rule
                        if hero_id not in rules["lang_overrides"]["specific"]:
                            rules["lang_overrides"]["specific"][hero_id] = {}
                        rules["lang_overrides"]["specific"][hero_id][skill_id] = lang_id
                    else: # Common rule
                        rules["lang_overrides"]["common"][skill_id] = lang_id
                    count += 1
                print(f" -> Loaded {count} language override rules.")
        except Exception as e:
            print(f"Warning: Could not process '{lang_rules_path.name}'. Error: {e}")

    # --- Load Hero Parameter Rules ---
    hero_rules_path = script_dir / "exception_hero_rules.csv"
    if not hero_rules_path.exists():
        print(f"Info: '{hero_rules_path.name}' not found. No hero rules loaded.")
    else:
        try:
            with open(hero_rules_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    hero_id = row.get("hero_id", "").strip()
                    placeholder = row.get("placeholder", "").strip().upper()
                    
                    if not placeholder: continue
                    
                    rule = {k: v.strip() for k, v in row.items()}

                    if hero_id: # Specific rule
                        if hero_id not in rules["hero_rules"]["specific"]:
                            rules["hero_rules"]["specific"][hero_id] = {}
                        rules["hero_rules"]["specific"][hero_id][placeholder] = rule
                    else: # Common rule
                        rules["hero_rules"]["common"][placeholder] = rule
                    count += 1
                print(f" -> Loaded {count} hero parameter rules.")
        except Exception as e:
            print(f"Warning: Could not process '{hero_rules_path.name}'. Error: {e}")
            
    return rules


def read_csv_to_dict(file_path: Path) -> dict:
    """A helper function to read a two-column CSV (KEY, TEXT) into a dictionary."""
    if not file_path.exists(): raise FileNotFoundError(f"CSV not found: {file_path}")
    data_dict = {}
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try: header = [h.upper() for h in next(reader)]
        except StopIteration: return {}
        try:
            key_index, text_index = header.index('KEY'), header.index('TEXT')
        except ValueError: raise ValueError(f"CSV must have 'KEY' and 'TEXT' columns: {file_path.name}")
        for row in reader:
            if len(row) > max(key_index, text_index): data_dict[row[key_index]] = row[text_index]
    return data_dict


def apply_overrides(data_dict: dict, override_list: list) -> int:
    """A helper function to apply language overrides."""
    count = 0
    if not override_list: return 0
    for entry in override_list:
        if "key" in entry and "text" in entry: data_dict[entry["key"]] = entry["text"]
    return count


def load_languages() -> dict:
    """Loads and merges English and Japanese language data."""
    print("--- Loading Language Data ---")
    en_dict = read_csv_to_dict(CSV_EN_PATH)
    ja_dict = read_csv_to_dict(CSV_JA_PATH)
    if JSON_OVERRIDE_PATH.exists():
        with open(JSON_OVERRIDE_PATH, "r", encoding="utf-8") as f: broken_json_string = f.read()
        def fix_newlines(m): return '"text": "' + m.group(1).replace(chr(13), "").replace(chr(10), "\\n") + '"'
        fixed_json_string = re.sub(r'"text":\s*"((?:\\"|[^"])*)"', fix_newlines, broken_json_string, flags=re.DOTALL)
        try: override_data = json.loads(fixed_json_string)
        except json.JSONDecodeError as e: raise e
        overrides_config = override_data.get("languageOverridesConfig", {}).get("overrides", {})
        apply_overrides(en_dict, overrides_config.get("English", {}).get("overrideEntries", []))
        apply_overrides(ja_dict, overrides_config.get("Japanese", {}).get("overrideEntries", []))
    merged_lang_dict = {}
    for key in set(en_dict.keys()) | set(ja_dict.keys()):
        merged_lang_dict[key] = {"en": en_dict.get(key, ""), "ja": ja_dict.get(key, "")}
    print(f" -> Unified language DB created with {len(merged_lang_dict)} keys.")
    return merged_lang_dict
    

def load_game_data() -> dict:
    """Loads all core game data JSONs into a structured dictionary."""
    print("\n--- Loading Core Game Data ---")
    game_data = {}
    def load_json(p):
        if not p.exists(): raise FileNotFoundError(f"Game data not found: {p}")
        with open(p, 'r', encoding='utf-8') as f: return json.load(f)
    
    game_data['heroes'] = load_json(CHARACTERS_PATH).get('charactersConfig', {}).get('heroes', [])
    
    specials_config = load_json(SPECIALS_PATH).get('specialsConfig', {})
    game_data['character_specials'] = {cs['id']: cs for cs in specials_config.get('characterSpecials', [])}
    game_data['special_properties'] = {p['id']: p for p in specials_config.get('specialProperties', [])}
    
    battle_config = load_json(BATTLE_PATH).get('battleConfig', {})
    game_data['status_effects'] = {se['id']: se for se in battle_config.get('statusEffects', [])}
    game_data['familiars'] = {f['id']: f for f in battle_config.get('familiars', [])}
    game_data['familiar_effects'] = {fe['id']: fe for fe in battle_config.get('familiarEffects', [])}
    
    game_data['passive_skills'] = {ps['id']: ps for ps in battle_config.get('passiveSkills', [])}

    # --- NEW: Load and consolidate keys that have extra descriptions (tooltips) ---
    extra_desc_keys = set()
    key_groups = [
        "statusEffectsWithExtraDescription",
        "specialPropertiesWithExtraDescription",
        "familiarEffectsWithExtraDescription",
        "familiarTypesWithExtraDescription"
    ]
    for key_group in key_groups:
        # Lowercase all keys for case-insensitive matching later
        keys = [k.lower() for k in battle_config.get(key_group, [])]
        extra_desc_keys.update(keys)
    
    game_data['extra_description_keys'] = extra_desc_keys
    print(f" -> Found {len(extra_desc_keys)} unique keys with extra descriptions (tooltips).")
    # ---

    game_data['master_db'] = {
        **game_data['character_specials'],
        **game_data['special_properties'],
        **game_data['status_effects'],
        **game_data['familiars'],
        **game_data['familiar_effects'],
        **game_data['passive_skills']
    }

    print(f" -> Loaded {len(game_data['heroes'])} heroes and created a master_db with {len(game_data['master_db'])} items.")
    return game_data


def load_hero_stats_from_csv(base_dir: Path, pattern: str) -> dict:
    """Finds the latest hero stats CSV and loads it into a dictionary."""
    print("\n--- Loading Hero Stats from CSV ---")
    try:
        search_path = str(base_dir / f"*{pattern}")
        list_of_files = glob.glob(search_path)
        if not list_of_files:
            raise FileNotFoundError(f"No hero stats CSV found in {base_dir} matching pattern '{pattern}'")
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Found latest stats file: {Path(latest_file).name}")
        df = pd.read_csv(latest_file)
        if 'ID' not in df.columns: raise ValueError("Stats CSV must contain an 'ID' column.")
        hero_stats_db = df.set_index('ID').to_dict('index')
        print(f" -> Loaded stats for {len(hero_stats_db)} heroes.")
        return hero_stats_db
    except Exception as e:
        print(f"FATAL: Could not load hero stats CSV. Error: {e}")
        raise