import csv
import json
import re
from pathlib import Path
import traceback
from pprint import pprint
import pandas as pd
import math
import glob
import os

# --- Constants ---
try:
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = Path("D:/RED") # Fixed path as per specification
except NameError:
    SCRIPT_DIR = Path.cwd()
    DATA_DIR = Path("D:/RED") # Fixed path as per specification
    print(f"Warning: '__file__' not found. Assuming script dir is {SCRIPT_DIR}")

# --- File Paths ---
HERO_STATS_CSV_PATTERN = "_private_heroes_*.csv"
CSV_EN_PATH = DATA_DIR / "English.csv"
CSV_JA_PATH = DATA_DIR / "Japanese.csv"
JSON_OVERRIDE_PATH = DATA_DIR / "languageOverrides.json"
CHARACTERS_PATH = DATA_DIR / "characters.json"
SPECIALS_PATH = DATA_DIR / "specials.json"
BATTLE_PATH = DATA_DIR / "battle.json"
OUTPUT_CSV_PATH = SCRIPT_DIR / "hero_skill_output.csv"


# --- Data Loading & Helper Functions ---
def read_csv_to_dict(file_path: Path) -> dict:
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
    count = 0
    if not override_list: return 0
    for entry in override_list:
        if "key" in entry and "text" in entry: data_dict[entry["key"]] = entry["text"]
    return count

def load_languages() -> dict:
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
    print(f" -> Loaded {len(game_data['heroes'])} heroes, {len(game_data['character_specials'])} specials, {len(game_data['status_effects'])} status effects.")
    return game_data

def load_hero_stats_from_csv(base_dir: Path, pattern: str) -> dict:
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

# --- Text Generation Helper ---
def generate_description(lang_id: str, lang_params: dict, lang_db: dict) -> dict:
    template = lang_db.get(lang_id, {"en": f"NO_TEMPLATE_FOR_{lang_id}", "ja": f"NO_TEMPLATE_FOR_{lang_id}"})
    desc_en, desc_ja = template.get("en", ""), template.get("ja", "")
    for key, value in lang_params.items():
        desc_en = desc_en.replace(f"{{{key}}}", str(value))
        desc_ja = desc_ja.replace(f"{{{key}}}", str(value))
    return {"en": desc_en, "ja": desc_ja}


# --- Analysis & Parsing Functions ---
def get_hero_final_stats(hero_id: str, hero_stats_db: dict) -> dict:
    hero_data = hero_stats_db.get(hero_id)
    if not hero_data: return {"max_attack": 0, "name": "N/A"}
    attack_col = 'Max level: Attack'
    for i in range(4, 0, -1):
        col_name = f'Max level CB{i}: Attack'
        if col_name in hero_data and pd.notna(hero_data[col_name]):
            attack_col = col_name
            break
    return {"max_attack": int(hero_data.get(attack_col, 0)), "name": hero_data.get('Name', 'N/A')}

def find_best_lang_id(data_block: dict, lang_key_subset: list) -> str:
    """
    Finds the best matching language ID from a subset based on keywords and scoring.
    """
    # 1. Define keyword importance and extract them from the data block
    PRIMARY_KEYWORDS = {'propertyType', 'effectType', 'statusEffect', 'buff'}
    SECONDARY_KEYWORDS = {'targetType', 'sideAffected', 'applicationChance', 'duration'}
    
    keywords = {}
    all_keys = {k.lower() for k in data_block.keys()}

    for k, v in data_block.items():
        if isinstance(v, str):
            keywords[k.lower()] = v.lower()

    # 2. Score each potential language key
    potential_matches = []
    for lang_key in lang_key_subset:
        score = 0
        normalized_lang_key = lang_key.lower()
        
        # --- Scoring Logic ---
        primary_hits = 0
        secondary_hits = 0
        
        for key, value in keywords.items():
            if value in normalized_lang_key:
                # Basic score
                score += 1
                # Bonus for full word match
                if f".{value}" in normalized_lang_key or f"{value}." in normalized_lang_key:
                    score += 1
                # Major bonus for primary keywords
                if key in PRIMARY_KEYWORDS:
                    score += 5
                    primary_hits +=1
                if key in SECONDARY_KEYWORDS:
                    score += 2
                    secondary_hits += 1

        # Bonus for value sign (e.g., negative values suggesting 'decrement')
        for key in all_keys:
            # Create a list of possible key variations to check for the value
            base_key_name = key.split('permil')[0].split('value')[0].split('power')[0]
            possible_keys_to_check = [key, key.capitalize(), key.upper(), base_key_name]
            
            val = None
            for pk in possible_keys_to_check:
                if pk in data_block:
                    val = data_block[pk]
                    break
            
            if val is not None and isinstance(val, (int, float)):
                if val < 0:
                    if 'decrement' in normalized_lang_key or 'negative' in normalized_lang_key:
                        score += 2
                elif val > 0:
                     if 'increment' in normalized_lang_key or 'positive' in normalized_lang_key:
                        score += 1

        if score > 0:
            potential_matches.append({'key': lang_key, 'score': score, 'p_hits': primary_hits, 's_hits': secondary_hits})

    if not potential_matches:
        return f"SEARCH_FAILED_FOR_{data_block.get('id', 'UNKNOWN_ID')}"

    # 3. Determine the best match
    # Sort by score (desc), then by primary keyword hits (desc), then secondary (desc), then by key length (asc, as a tie-breaker)
    potential_matches.sort(key=lambda x: (x['score'], x['p_hits'], x['s_hits'], -len(x['key'])), reverse=True)
    
    # Check for ties in the top score and primary hits
    top_score = potential_matches[0]['score']
    top_p_hits = potential_matches[0]['p_hits']
    ties = [m for m in potential_matches if m['score'] == top_score and m['p_hits'] == top_p_hits]

    if len(ties) > 1:
        # If there's a tie, prefer shorter keys as they are more generic/base
        ties.sort(key=lambda x: len(x['key']))
        # Optional: Add logging for tied results to debug later
        # print(f"\n  - Tie detected for ID {data_block.get('id')}. Top candidates: {[t['key'] for t in ties]}")
        return ties[0]['key']

    return potential_matches[0]['key']

def parse_direct_effect(special_data, hero_stats, lang_db, game_db, parsers):
    effect_data = special_data.get("directEffect")
    if not effect_data or not effect_data.get("effectType"): return None
    try:
        effect_type_str = effect_data.get('effectType', '')
        parts = ["specials.v2.directeffect", effect_type_str.lower()]
        if t := effect_data.get('typeOfTarget'): parts.append(t.lower())
        if s := effect_data.get('sideAffected'): parts.append(s.lower())
        lang_id = ".".join(parts)
        if effect_data.get("hasFixedPower"): lang_id += ".fixedpower"
    except AttributeError: return None
    
    params = {}
    base = effect_data.get('powerMultiplierPerMil', 0)
    inc = effect_data.get('powerMultiplierIncrementPerLevelPerMil', 0)
    lvl = special_data.get('maxLevel', 1)
    
    p_map = {"Damage":"HEALTH","Heal":"HEALTH","HealthBoost":"HEALTHBOOST","AddMana":"MANA"}
    placeholder = p_map.get(effect_type_str, "VALUE")
    
    final_val = 0
    if base > 0 or inc > 0:
        if effect_data.get("hasFixedPower"):
            final_val = round(base + inc * (lvl - 1))
        else:
            total_per_mil = base + inc * (lvl - 1)
            final_val = round(total_per_mil/100) if effect_type_str=="AddMana" else round(total_per_mil/10)
        params[placeholder] = final_val

    desc = generate_description(lang_id, params, lang_db)
    return {"lang_id": lang_id, "params": json.dumps(params), **desc}

def parse_properties(properties_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, parsers: dict) -> list:
    if not properties_list: return []
    
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    prop_lang_subset = parsers['prop_lang_subset']
    
    for prop_id in properties_list:
        prop_details = game_db['special_properties'].get(prop_id)
        if not prop_details:
            print(f"\n  - WARNING: Property ID '{prop_id}' not found.")
            continue

        # Use the new find_best_lang_id function
        lang_id = find_best_lang_id(prop_details, prop_lang_subset)

        lang_params = {}
        main_template_text = lang_db.get(lang_id, {}).get("en", "")
        extra_lang_id = '.'.join(lang_id.split('.')[:4]) + ".extra"
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        all_placeholders = set(re.findall(r'\{(\w+)\}', main_template_text + extra_template_text))
        
        nested_effects = []
        for key, value in prop_details.items():
            # --- RECURSION HANDLING ---
            if isinstance(value, list) and "statuseffect" in key.lower():
                nested_effects.extend(parsers['status_effects'](value, special_data, hero_stats, lang_db, game_db, parsers))
                continue
            
            # --- DYNAMIC VALUE CALCULATION ---
            for p_holder in all_placeholders:
                if p_holder in lang_params: continue
                normalized_pholder = p_holder.lower()
                normalized_key = key.lower()
                
                if normalized_key.startswith(normalized_pholder):
                    base_key = key
                    # Simplified increment key logic, can be improved
                    inc_key_guess = base_key.replace("PerMil", "IncrementPerLevelPerMil") if "PerMil" in base_key else base_key + "IncrementPerLevel"
                    inc_key = inc_key_guess if inc_key_guess in prop_details else None

                    base_val = prop_details.get(base_key)
                    inc_val = prop_details.get(inc_key, 0) if inc_key else 0
                    
                    if isinstance(base_val, (int, float)) and isinstance(inc_val, (int, float)):
                        calculated_val = base_val + inc_val * (max_level - 1)
                        if 'permil' in base_key.lower():
                             # Keep it as a float for now, formatting can be done later
                            lang_params[p_holder] = calculated_val / 10
                        else:
                            lang_params[p_holder] = int(calculated_val)
        
        # Simplified handling for now, can be expanded
        if "FIXEDPOWER" in lang_params:
            if "MAX" in all_placeholders: lang_params["MAX"] = lang_params["FIXEDPOWER"] * 2
            if "MIN" in all_placeholders: lang_params["MIN"] = math.floor(lang_params["FIXEDPOWER"] / 2)

        main_desc = generate_description(lang_id, lang_params, lang_db)
        tooltip_desc = generate_description(extra_lang_id, lang_params, lang_db) if extra_lang_id in lang_db else {"en": "", "ja": ""}

        parsed_items.append({
            "id": prop_id, "lang_id": lang_id,
            "description_en": main_desc["en"], "description_ja": main_desc["ja"],
            "tooltip_en": tooltip_desc["en"], "tooltip_ja": tooltip_desc["ja"],
            "params": json.dumps(lang_params),
            "nested_effects": nested_effects
        })

    return parsed_items

def parse_status_effects(status_effects_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, parsers: dict) -> list:
    if not status_effects_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    se_lang_subset = parsers['se_lang_subset']
    
    for effect_instance in status_effects_list:
        effect_id = effect_instance.get("id")
        effect_details = game_db['status_effects'].get(effect_id)
        if not effect_details: continue

        # Combine instance data (like 'turns') with base effect data for a richer keyword source
        combined_details = {**effect_details, **effect_instance}
        lang_id = find_best_lang_id(combined_details, se_lang_subset)

        lang_params, turns = {}, effect_instance.get("turns", 0)
        if turns > 0: lang_params["TURNS"] = turns
        
        template_text_en = lang_db.get(lang_id, {}).get("en", "")
        template_text_ja = lang_db.get(lang_id, {}).get("ja", "")
        template_text = template_text_en + template_text_ja
        placeholders = set(re.findall(r'\{(\w+)\}', template_text))
        is_total_damage = "over {TURNS} turns" in template_text_en or "{TURNS}ターンに渡って" in template_text_ja

        # Combine instance and details for value lookup
        value_source = {**effect_details, **effect_instance}

        for p_holder in placeholders:
            if p_holder in lang_params: continue
            
            normalized_pholder = p_holder.lower()
            found_key = None
            # Search for a matching key in the value source
            for key in value_source:
                if key.lower().startswith(normalized_pholder):
                    found_key = key
                    break # Take the first match for now

            if found_key:
                base_val = value_source.get(found_key, 0)
                # Simplified increment key logic
                inc_key_guess = found_key.replace("PerMil", "IncrementPerLevelPerMil") if "PerMil" in found_key else found_key + "IncrementPerLevel"
                inc_val = value_source.get(inc_key_guess, 0)

                if isinstance(base_val, (int, float)):
                    val = base_val + inc_val * (max_level - 1)
                    if "permil" in found_key.lower():
                        val /= 1000
                        if p_holder == "DAMAGE":
                            damage = math.floor(val * hero_stats.get("max_attack", 0))
                            lang_params[p_holder] = damage * turns if is_total_damage and turns > 0 else damage
                        else:
                            # Keep precision for now, format later
                            lang_params[p_holder] = val * 100 
                    else: # Fixed value
                        lang_params[p_holder] = val * turns if is_total_damage and turns > 0 else int(val)

        # Re-generate description with potentially rounded/formatted values
        # This part could be enhanced with a dedicated formatting function
        formatted_params = {}
        for k, v in lang_params.items():
            if isinstance(v, float):
                # Example formatting: show one decimal place for floats
                formatted_params[k] = f"{v:+.1f}" if v > 0 and not str(v).startswith('+') else f"{v:.1f}"
            else:
                formatted_params[k] = v

        descriptions = generate_description(lang_id, formatted_params, lang_db)
        parsed_items.append({ "id": effect_id, "lang_id": lang_id, "params": json.dumps(lang_params), **descriptions})
    return parsed_items

# --- CSV Output Function ---
def write_results_to_csv(processed_data: list, output_path: Path):
    print(f"\n--- Writing results to {output_path} ---")
    if not processed_data: return
    flat_data = []
    for hero in processed_data:
        row = {'hero_id': hero.get('id'), 'hero_name': hero.get('name', 'N/A')}
        skills = hero.get('skillDescriptions', {})
        if de := skills.get('directEffect'): row.update({f'de_{k}': v for k, v in de.items()})
        
        props = skills.get('properties', [])
        for i, p in enumerate(props[:3]):
            row.update({f'prop_{i+1}_{k}': v for k, v in p.items() if k != 'nested_effects'})
            if nested := p.get('nested_effects'):
                for j, ne in enumerate(nested[:2]):
                     row.update({f'prop_{i+1}_nested_{j+1}_{k}': v for k, v in ne.items()})

        effects = skills.get('statusEffects', [])
        for i, e in enumerate(effects[:5]): row.update({f'se_{i+1}_{k}': v for k, v in e.items()})
        flat_data.append(row)
    try:
        df = pd.DataFrame(flat_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Successfully saved {len(df)} rows to CSV.")
    except Exception as e: print(f"FATAL: Failed to write CSV: {e}")


# --- Main Processing Function ---
def process_all_heroes(lang_db: dict, game_db: dict, hero_stats_db: dict, parsers: dict) -> list:
    print("\n--- Starting Hero Processing ---")
    all_heroes = game_db.get('heroes', [])
    processed_heroes_data = []
    
    for i, hero in enumerate(all_heroes):
        hero_id = hero.get("id", "UNKNOWN")
        print(f"\r[{i+1}/{len(all_heroes)}] Processing: {hero_id.ljust(40)}", end="")
        
        hero_final_stats = get_hero_final_stats(hero_id, hero_stats_db)
        hero['name'] = hero_final_stats.get('name')
        hero['skillDescriptions'] = {}
        
        if not (special_id := hero.get("specialId")) or not (special_data := game_db['character_specials'].get(special_id)):
            processed_heroes_data.append(hero)
            continue
            
        hero['skillDescriptions']['directEffect'] = parsers['direct_effect'](special_data, hero_final_stats, lang_db, game_db, parsers)
        hero['skillDescriptions']['properties'] = parsers['properties'](special_data.get("properties", []), special_data, hero_final_stats, lang_db, game_db, parsers)
        hero['skillDescriptions']['statusEffects'] = parsers['status_effects'](special_data.get("statusEffects", []), special_data, hero_final_stats, lang_db, game_db, parsers)
        processed_heroes_data.append(hero)
    
    print("\n" + "--- Finished processing all heroes. ---")
    return processed_heroes_data

def main():
    """Main function to run the entire process."""
    try:
        language_db = load_languages()
        game_db = load_game_data()
        hero_stats_db = load_hero_stats_from_csv(DATA_DIR, HERO_STATS_CSV_PATTERN)

        print("\nOptimizing language data for search...")
        parsers = {
            'se_lang_subset': [key for key in language_db if key.startswith("specials.v2.statuseffect.")],
            'prop_lang_subset': [key for key in language_db if key.startswith("specials.v2.property.")]
        }
        # The parser functions themselves are now passed
        parsers['direct_effect'] = parse_direct_effect
        parsers['properties'] = parse_properties
        parsers['status_effects'] = parse_status_effects
        print(f" -> Found {len(parsers['se_lang_subset'])} status effect and {len(parsers['prop_lang_subset'])} property language keys.")

        final_hero_data = process_all_heroes(language_db, game_db, hero_stats_db, parsers)
        write_results_to_csv(final_hero_data, OUTPUT_CSV_PATH)
        print(f"\nProcess complete. Output saved to {OUTPUT_CSV_PATH}")

        # --- NEW: Unresolved Placeholder Summary ---
        print("\n--- Analyzing unresolved placeholders in final output ---")
        from collections import Counter
        unresolved_counter = Counter()

        for hero in final_hero_data:
            if 'skillDescriptions' not in hero: continue
            for skill_type, skill_data in hero['skillDescriptions'].items():
                if not skill_data: continue
                
                items_to_check = []
                if isinstance(skill_data, list):
                    items_to_check.extend(skill_data)
                elif isinstance(skill_data, dict):
                    items_to_check.append(skill_data)

                for item in items_to_check:
                    if not isinstance(item, dict): continue
                    for key, text in item.items():
                        if isinstance(text, str) and ('description' in key or 'tooltip' in key):
                            found = re.findall(r'(\{\w+\})', text)
                            if found:
                                unresolved_counter.update(found)
        
        if not unresolved_counter:
            print("✅ All placeholders resolved successfully!")
        else:
            print("\n--- Unresolved Placeholder Summary ---")
            print(f"{'Placeholder':<30} | {'Count':<10}")
            print("-" * 43)
            for placeholder, count in unresolved_counter.most_common():
                print(f"{placeholder:<30} | {count:<10}")
            print("-" * 43)
            print(f"Total Unique Unresolved Placeholders: {len(unresolved_counter)}")

    except Exception as e:
        print(f"\n[FATAL ERROR]: {type(e).__name__} - {e}")
        traceback.print_exc()

# --- Main Execution Block ---
if __name__ == "__main__":
    main()