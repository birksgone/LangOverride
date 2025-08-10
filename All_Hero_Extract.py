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


def flatten_json(y):
    """ Flattens a nested dictionary and list structure. """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

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

def format_value(value):
    """Formats numbers for display, removing trailing .0"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, float):
        # Format with one decimal place for non-integers
        return f"{value:.1f}"
    return value

def find_and_calculate_value(p_holder: str, data_block: dict, max_level: int, is_modifier: bool = False) -> (any, str):
    """
    Finds and calculates a value for a placeholder using a flattened JSON structure.
    Accepts an is_modifier flag to apply special calculation logic.
    """
    # --- Handle fixed value placeholders first ---
    if p_holder.upper() == 'MAXSTACK':
        return 10, 'Fixed Value'

    if not isinstance(data_block, dict):
        return None, None

    flat_data = flatten_json(data_block)
    normalized_pholder = p_holder.lower()
    
    is_chance_related = 'chance' in normalized_pholder
    
    ph_base_name = normalized_pholder
    ph_index = None
    match = re.match(r'(\w+)(\d+)$', normalized_pholder)
    if match:
        ph_base_name, index_str = match.groups()
        ph_index = int(index_str) - 1

    candidate_keys = []
    for key, value in flat_data.items():
        if not is_chance_related and 'chance' in key.lower(): continue
        if is_chance_related and 'chance' not in key.lower(): continue
        
        if ph_base_name in key.lower():
            if ph_index is not None:
                if f"_{ph_index}_" in key or key.endswith(f"_{ph_index}"):
                    candidate_keys.append(key)
            else:
                candidate_keys.append(key)

    if not candidate_keys:
        return None, None
    
    priority_keywords = ['power', 'value', 'modifier', 'fixed', 'multiplier']
    priority_keys = [k for k in candidate_keys if any(kw in k.lower() for kw in priority_keywords)]
    
    if priority_keys:
        found_key = min(priority_keys, key=len)
    else:
        found_key = min(candidate_keys, key=len)

    # --- Calculation ---
    base_val = flat_data.get(found_key, 0)
    
    inc_key = ""
    if 'fixedpower' in found_key.lower():
        inc_key_pattern = found_key.lower().replace('fixedpower', 'fixedpowerincrementperlevel')
    else:
        inc_key_pattern = found_key.lower().replace('permil', 'incrementperlevelpermil')

    for k in flat_data.keys():
        if k.lower() == inc_key_pattern:
            inc_key = k
            break
            
    inc_val = flat_data.get(inc_key, 0)

    if not isinstance(base_val, (int, float)): return None, None
    
    # --- NEW: Switch calculation based on is_modifier flag ---
    if is_modifier:
        calculated_val = ((base_val - 1000) + (inc_val * (max_level - 1))) / 10
        return calculated_val, found_key
    else:
        calculated_val = base_val + inc_val * (max_level - 1)
        if 'permil' in found_key.lower():
            return calculated_val / 10, found_key
        else:
            return int(calculated_val), found_key

def find_best_lang_id(data_block: dict, lang_key_subset: list) -> str:
    """
    Finds the best matching language ID using a decisive primary keyword and secondary scoring.
    Includes debug prints to trace the logic.
    """
    keywords = {k.lower(): v.lower() for k, v in data_block.items() if isinstance(v, str)}
    
    # --- Step 1: Identify the single most important keyword (the "super key") ---
    primary_keyword = keywords.get('propertytype') or keywords.get('statuseffect')
    
    # --- Optional Debug Print ---
    # print(f"\nProcessing ID: {data_block.get('id', 'N/A')}, Primary Keyword: {primary_keyword}")

    potential_matches = []
    for lang_key in lang_key_subset:
        score = 0
        normalized_lang_key = lang_key.lower()

        # --- Step 2: Core Scoring - give a massive, decisive bonus to the primary keyword ---
        if primary_keyword and primary_keyword in normalized_lang_key:
            score += 100  # This bonus should outweigh all other small scores combined
        # If a primary keyword exists but doesn't match, this lang_key is likely wrong. 
        # We can continue scoring for edge cases, but it will have a hard time winning.
        
        # --- Step 3: Secondary Scoring for tie-breaking and refinement ---
        # Add smaller scores for other descriptive keywords
        other_keywords = {'effecttype', 'targettype', 'sideaffected', 'buff'}
        for key_name in other_keywords:
            if value := keywords.get(key_name):
                if value in normalized_lang_key:
                    score += 5 # Add a small bonus for other matches

        # Bonus for fixedPower
        if 'fixedpower' in normalized_lang_key and ('fixedPower' in data_block or data_block.get('hasFixedPower')):
            score += 3
        
        # Bonus for value sign
        for val in data_block.values():
            if isinstance(val, (int, float)) and val < 0 and 'decrement' in normalized_lang_key:
                score += 2

        if score > 0:
            potential_matches.append({'key': lang_key, 'score': score})
            # --- Optional Debug Print for high-score candidates ---
            # if score > 50:
            #    print(f"  -> Candidate: {lang_key:<100} | Score: {score}")

    if not potential_matches:
        return f"SEARCH_FAILED_FOR_{data_block.get('id', 'UNKNOWN_ID')}"

    # Sort by score (desc), then by key length (asc) to prefer more generic base keys in case of a tie
    potential_matches.sort(key=lambda x: (-x['score'], len(x['key'])))
    
    best_match = potential_matches[0]
    # --- Optional Debug Print for the winner ---
    # print(f"  => Best Match: {best_match['key']} (Score: {best_match['score']})")

    return best_match['key']

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

        lang_id = parsers['find_best_lang_id'](prop_details, prop_lang_subset)
        lang_params = {}
        
        # --- NEW: Check for Modifier context in properties as well ---
        is_modifier_effect = False
        if 'propertyType' in prop_details and 'modifier' in prop_details['propertyType'].lower():
            is_modifier_effect = True
        elif 'effectType' in prop_details and 'modifier' in prop_details['effectType'].lower():
             is_modifier_effect = True

        main_template_text = lang_db.get(lang_id, {}).get("en", "")
        extra_lang_id = '.'.join(lang_id.split('.')[:4]) + ".extra"
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        all_placeholders = set(re.findall(r'\{(\w+)\}', main_template_text + extra_template_text))
        
        processing_order = ['FIXEDPOWER', 'DAMAGE', 'POWER', 'VALUE', 'BASEPOWER']
        ordered_ph = [p for p in processing_order if p in all_placeholders]
        unordered_ph = list(all_placeholders - set(ordered_ph) - {'MIN', 'MAX'})
        
        for p_holder in ordered_ph + unordered_ph:
            if p_holder in lang_params: continue
            # --- NEW: Pass the is_modifier flag ---
            value, _ = find_and_calculate_value(p_holder, prop_details, max_level, is_modifier=is_modifier_effect)
            if value is not None:
                lang_params[p_holder] = value
        
        if 'MAX' in all_placeholders and 'FIXEDPOWER' in lang_params:
            lang_params['MAX'] = lang_params['FIXEDPOWER'] * 2
        elif 'MAX' in all_placeholders and 'BASEPOWER' in lang_params:
             lang_params['MAX'] = lang_params['BASEPOWER'] * 2
        
        if 'MIN' in all_placeholders and 'FIXEDPOWER' in lang_params:
            lang_params['MIN'] = math.floor(lang_params['FIXEDPOWER'] / 2)
        elif 'MIN' in all_placeholders and 'BASEPOWER' in lang_params:
            lang_params['MIN'] = math.floor(lang_params['BASEPOWER'] / 2)

        nested_effects = []
        if 'statusEffectsPerHit' in prop_details:
            nested_effects.extend(parsers['status_effects'](prop_details['statusEffectsPerHit'], special_data, hero_stats, lang_db, game_db, parsers))
        if 'statusEffects' in prop_details:
             nested_effects.extend(parsers['status_effects'](prop_details['statusEffects'], special_data, hero_stats, lang_db, game_db, parsers))

        formatted_params = {}
        template_str_for_check = main_template_text + extra_template_text
        for k, v in lang_params.items():
            formatted_val = format_value(v)
            is_percentage = f"{{{k}}}" in template_str_for_check and "%" in template_str_for_check
            
            if isinstance(v, (int, float)) and v > 0 and k.upper() not in ["TURNS", "DAMAGE", "MAX", "MIN", "FIXEDPOWER", "BASEPOWER"] and is_percentage:
                 formatted_params[k] = f"+{formatted_val}"
            else:
                 formatted_params[k] = formatted_val
        
        for p in all_placeholders:
             if p not in formatted_params:
                 formatted_params[p] = f"{{{p}}}"

        main_desc = generate_description(lang_id, formatted_params, lang_db)
        tooltip_desc = generate_description(extra_lang_id, formatted_params, lang_db) if extra_lang_id in lang_db else {"en": "", "ja": ""}

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

        combined_details = {**effect_details, **effect_instance}
        lang_id = parsers['find_best_lang_id'](combined_details, se_lang_subset)

        lang_params = {}
        if (turns := effect_instance.get("turns", 0)) > 0: 
            lang_params["TURNS"] = turns
        
        # --- NEW: Check for Modifier context ---
        is_modifier_effect = False
        if 'statusEffect' in effect_details and 'modifier' in effect_details['statusEffect'].lower():
            is_modifier_effect = True
        elif 'effectType' in effect_details and 'modifier' in effect_details['effectType'].lower():
             is_modifier_effect = True

        template_text_en = lang_db.get(lang_id, {}).get("en", "")
        template_text_ja = lang_db.get(lang_id, {}).get("ja", "")
        placeholders = set(re.findall(r'\{(\w+)\}', template_text_en + template_text_ja))
        is_total_damage = "over {TURNS} turns" in template_text_en or "{TURNS}ターンに渡って" in template_text_ja
        
        for p_holder in placeholders:
            if p_holder in lang_params: continue
            
            value_source = {**effect_details, **effect_instance}
            # --- NEW: Pass the is_modifier flag to the calculation function ---
            value, found_key = find_and_calculate_value(p_holder, value_source, max_level, is_modifier=is_modifier_effect)

            if value is not None:
                if p_holder.upper() == "DAMAGE" and "permil" in (found_key or "").lower():
                    damage_per_turn = math.floor( (value / 100) * hero_stats.get("max_attack", 0))
                    lang_params[p_holder] = damage_per_turn * turns if is_total_damage and turns > 0 else damage_per_turn
                else:
                    lang_params[p_holder] = value
        
        formatted_params = {}
        template_str_for_check = lang_db.get(lang_id,{}).get("en","")
        
        for k, v in lang_params.items():
            formatted_val = format_value(v)
            is_percentage = f"{{{k}}}" in template_str_for_check and "%" in template_str_for_check
            
            # Refined '+' sign logic
            if isinstance(v, (int, float)) and v > 0 and k.upper() not in ["TURNS", "DAMAGE", "MAX", "MIN", "FIXEDPOWER"] and is_percentage:
                 formatted_params[k] = f"+{formatted_val}"
            else:
                 formatted_params[k] = formatted_val
        
        for p in placeholders:
            if p not in formatted_params:
                formatted_params[p] = f"{{{p}}}"

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
        parsers['find_best_lang_id'] = find_best_lang_id
        parsers['direct_effect'] = parse_direct_effect
        parsers['properties'] = parse_properties
        parsers['status_effects'] = parse_status_effects
        print(f" -> Found {len(parsers['se_lang_subset'])} status effect and {len(parsers['prop_lang_subset'])} property language keys.")

        final_hero_data = process_all_heroes(language_db, game_db, hero_stats_db, parsers)
        write_results_to_csv(final_hero_data, OUTPUT_CSV_PATH)
        print(f"\nProcess complete. Output saved to {OUTPUT_CSV_PATH}")

        # --- Unresolved Placeholder Summary ---
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