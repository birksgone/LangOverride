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

# --- NEW: Load configuration from external JSON file ---
def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"Warning: Config file not found at {path}, using empty config.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Constants ---
try:
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = Path("D:/RED") # Fixed path as per specification
except NameError:
    SCRIPT_DIR = Path.cwd()
    DATA_DIR = Path("D:/RED") # Fixed path as per specification
    print(f"Warning: '__file__' not found. Assuming script dir is {SCRIPT_DIR}")

# --- File Paths ---
CONFIG_PATH = SCRIPT_DIR / "config.json"
HERO_STATS_CSV_PATTERN = "_private_heroes_*.csv"
CSV_EN_PATH = DATA_DIR / "English.csv"
CSV_JA_PATH = DATA_DIR / "Japanese.csv"
JSON_OVERRIDE_PATH = DATA_DIR / "languageOverrides.json"
CHARACTERS_PATH = DATA_DIR / "characters.json"
SPECIALS_PATH = DATA_DIR / "specials.json"
BATTLE_PATH = DATA_DIR / "battle.json"
OUTPUT_CSV_PATH = SCRIPT_DIR / "hero_skill_output.csv"

# --- Configuration Loading ---
def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"Warning: Config file not found at {path}, using empty config.")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"FATAL: Could not parse config.json. Error: {e}")
            raise
            
# Load the rules when the script starts
CONFIG = load_config(CONFIG_PATH)
EXCEPTION_RULES = CONFIG.get("EXCEPTION_RULES", {})
print(f" -> Loaded {len(EXCEPTION_RULES)} exception rules from config.json.")

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
    game_data['familiars'] = {f['id']: f for f in battle_config.get('familiars', [])}
    game_data['familiar_effects'] = {fe['id']: fe for fe in battle_config.get('familiarEffects', [])}
    
    game_data['passive_skills'] = {ps['id']: ps for ps in battle_config.get('passiveSkills', [])}

    # Create a master database for easy ID lookup across all relevant tables
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

def get_full_hero_data(base_data: dict, game_db: dict) -> dict:
    resolved_data = json.loads(json.dumps(base_data))
    processed_ids = set()
    _resolve_recursive(resolved_data, game_db['master_db'], processed_ids)
    return resolved_data

def _resolve_recursive(current_data, master_db, processed_ids):
    if id(current_data) in processed_ids: return
    processed_ids.add(id(current_data))
    
    ID_KEYS_FOR_LISTS = [
        'properties', 'statusEffects', 'statusEffectsPerHit',
        'summonedFamiliars', 'effects', 'passiveSkills', 'costumeBonusPassiveSkillIds',
        'statusEffectsToAdd', 'statusEffectCollections' # Added new keys to resolve
    ]

    if isinstance(current_data, dict):
        for key, value in list(current_data.items()):
            if key.lower().endswith('id') and isinstance(value, str):
                if value in master_db and value not in processed_ids:
                    processed_ids.add(value)
                    new_data = json.loads(json.dumps(master_db[value]))
                    _resolve_recursive(new_data, master_db, processed_ids)
                    current_data[f"{key}_details"] = new_data
            
            elif key in ID_KEYS_FOR_LISTS and isinstance(value, list):
                _resolve_recursive(value, master_db, processed_ids)
            
            elif isinstance(value, (dict, list)):
                _resolve_recursive(value, master_db, processed_ids)

    elif isinstance(current_data, list):
        for i, item in enumerate(current_data):
            item_id_to_resolve = None
            
            if isinstance(item, str):
                item_id_to_resolve = item
            elif isinstance(item, dict) and 'id' in item:
                item_id_to_resolve = item.get('id')

            if item_id_to_resolve and item_id_to_resolve in master_db and item_id_to_resolve not in processed_ids:
                processed_ids.add(item_id_to_resolve)
                new_data = json.loads(json.dumps(master_db[item_id_to_resolve]))
                
                _resolve_recursive(new_data, master_db, processed_ids)
                
                if isinstance(current_data[i], str):
                    current_data[i] = new_data
                else:
                    current_data[i].update(new_data)
            
            elif isinstance(item, (dict, list)):
                 _resolve_recursive(item, master_db, processed_ids)

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
    if isinstance(value, float) and value.is_integer(): return int(value)
    if isinstance(value, float): return f"{value:.1f}"
    return value

# This is based on the STABLE version of the function, NOT the last failed one.
def find_and_calculate_value(p_holder: str, data_block: dict, max_level: int, is_modifier: bool = False) -> (any, str):
    
    # --- Step 1: Manual override ---
    p_holder_upper = p_holder.upper()
    if p_holder_upper in EXCEPTION_RULES:
        rule = EXCEPTION_RULES[p_holder_upper]
        calc_method = rule["calc"]

        if calc_method == "fixed":
            return rule["value"], "Fixed Rule"

        key_to_find = rule["key"]
        flat_data = flatten_json(data_block)
        
        # --- NEW MINIMAL FIX ---
        # Find a key that uniquely ends with the key from the config file.
        matching_keys = [k for k in flat_data if k.endswith(key_to_find)]
        
        if len(matching_keys) == 1:
            found_key = matching_keys[0]
            value = flat_data[found_key]
            # Perform calculation and RETURN right here.
            if isinstance(value, (int, float)):
                # Note: For now, exception rules do not support increment calculations.
                # This can be added later if needed.
                if 'permil' in found_key.lower():
                    return value / 10, f"Exception Rule: {found_key}"
                return int(value), f"Exception Rule: {found_key}"
        # --- END OF FIX ---

        # If the smart search fails, return None.
        return None, f"Exception rule key '{key_to_find}' not found or ambiguous"

    # --- Step 2: Fallback to the STABLE automatic logic ---
    if not isinstance(data_block, dict): return None, None
    # ... (The rest of the function is the known, stable version) ...
    flat_data = flatten_json(data_block)
    
    normalized_pholder = p_holder.lower()
    is_chance_related = 'chance' in normalized_pholder
    
    ph_keywords = [s.lower() for s in re.findall('[A-Z][^A-Z]*', p_holder)]
    if not ph_keywords: ph_keywords = [normalized_pholder]

    ph_base_name, ph_index = normalized_pholder, None
    match = re.match(r'(\w+)(\d+)$', normalized_pholder)
    if match:
        base, index_str = match.groups()
        ph_keywords = [s.lower() for s in re.findall('[A-Z][^A-Z]*', base.capitalize())]
        if not ph_keywords: ph_keywords = [base]
        ph_index = int(index_str) - 1
        
    candidate_keys = []
    for key in flat_data:
        key_lower = key.lower()
        if not is_chance_related and 'chance' in key_lower: continue
        if is_chance_related and 'chance' not in key_lower: continue
        search_key = key_lower.replace('generation', 'regen').replace('value', 'power')
        if any(part in search_key for part in ph_keywords):
            if ph_index is not None:
                if f"_{ph_index}_" in key_lower or key_lower.endswith(f"_{ph_index}"):
                    candidate_keys.append(key)
            else:
                candidate_keys.append(key)
    
    if not candidate_keys: return None, None
    
    priority_keywords = ['power', 'modifier', 'fixed', 'multiplier', 'permil']
    priority_keys = [k for k in candidate_keys if any(kw in k.lower() for kw in priority_keywords)]
    found_key = min(priority_keys, key=len) if priority_keys else min(candidate_keys, key=len)

    base_val = flat_data.get(found_key, 0)
    inc_key_pattern = found_key.lower().replace("permil", "incrementperlevelpermil").replace('fixedpower', 'fixedpowerincrementperlevel')
    inc_key = next((k for k in flat_data if k.lower() == inc_key_pattern), None)
    inc_val = flat_data.get(inc_key, 0)

    if not isinstance(base_val, (int, float)): return None, None
    
    if is_modifier:
        calculated_val = ((base_val - 1000) + (inc_val * (max_level - 1))) / 10
        return calculated_val, found_key
    else:
        calculated_val = base_val + inc_val * (max_level - 1)
        if 'permil' in found_key.lower():
            return calculated_val / 10, found_key
        else:
            return int(calculated_val), found_key

def find_best_lang_id(data_block: dict, lang_key_subset: list, parent_block: dict = None) -> str:
    if 'statusEffect' in data_block:
        buff_map = {"MinorDebuff": "minor", "MajorDebuff": "major", "MinorBuff": "minor", "MajorBuff": "major"}
        intensity = buff_map.get(data_block.get('buff'))
        status_effect_val = data_block.get('statusEffect')
        effect_name = status_effect_val.lower() if isinstance(status_effect_val, str) else None
        target = (parent_block or data_block).get('statusTargetType', '').lower()
        side = (parent_block or data_block).get('sideAffected', '').lower()
        if all([intensity, effect_name, target, side]):
            constructed_id = f"specials.v2.statuseffect.{intensity}.{effect_name}.{target}.{side}"
            if constructed_id in lang_key_subset: return constructed_id

    keywords = {k.lower(): v.lower() for k, v in data_block.items() if isinstance(v, str)}
    if parent_block and isinstance(parent_block, dict):
        context_keys = ['targettype', 'sideaffected', 'statustargettype']
        for key in context_keys:
            if key in parent_block and isinstance(parent_block[key], str):
                if key not in keywords: keywords[key] = parent_block[key].lower()

    prop_type, status_effect = keywords.get('propertytype'), keywords.get('statuseffect')
    primary_keyword_raw = prop_type or status_effect
    primary_keyword = primary_keyword_raw.strip() if isinstance(primary_keyword_raw, str) else None
    
    filtered_candidates = []
    if primary_keyword:
        for lang_key in lang_key_subset:
            if primary_keyword in lang_key.split('.'): filtered_candidates.append(lang_key)
    if not filtered_candidates: filtered_candidates = lang_key_subset

    potential_matches = []
    for lang_key in filtered_candidates:
        score, lang_key_parts = 0, lang_key.lower().split('.')
        if primary_keyword and primary_keyword in lang_key_parts: score += 100
        
        other_keywords = {'effecttype', 'targettype', 'sideaffected', 'buff', 'statustargettype'}
        for key_name in other_keywords:
            if value := keywords.get(key_name):
                if value.lower() in lang_key_parts: score += 5

        if 'fixedpower' in lang_key_parts and ('fixedPower' in data_block or data_block.get('hasFixedPower')): score += 3
        if any(isinstance(v, (int, float)) and v < 0 for v in data_block.values()) and 'decrement' in lang_key_parts: score += 2

        if score > 0: potential_matches.append({'key': lang_key, 'score': score})

    if not potential_matches: return f"SEARCH_FAILED_FOR_{data_block.get('id', 'UNKNOWN_ID')}_TYPE_{primary_keyword}"
    potential_matches.sort(key=lambda x: (-x['score'], len(x['key'])))
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
    base, inc, lvl = effect_data.get('powerMultiplierPerMil', 0), effect_data.get('powerMultiplierIncrementPerLevelPerMil', 0), special_data.get('maxLevel', 1)
    
    p_map = {"Damage":"HEALTH","Heal":"HEALTH","HealthBoost":"HEALTHBOOST","AddMana":"MANA"}
    placeholder = p_map.get(effect_type_str, "VALUE")
    
    if base > 0 or inc > 0:
        total_per_mil = base + inc * (lvl - 1)
        final_val = round(total_per_mil) if effect_data.get("hasFixedPower") else (round(total_per_mil/100) if effect_type_str=="AddMana" else round(total_per_mil/10))
        params[placeholder] = final_val

    desc = generate_description(lang_id, params, lang_db)
    return {"lang_id": lang_id, "params": json.dumps(params), **desc}

def parse_properties(properties_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, parsers: dict) -> list:
    if not properties_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    prop_lang_subset = parsers['prop_lang_subset']
    
    for prop_id_or_dict in properties_list:
        prop_data = {}
        prop_id = None
        
        # --- FIX: Handle both string IDs (str) and pre-resolved dictionaries (dict) ---
        if isinstance(prop_id_or_dict, dict):
            prop_data = prop_id_or_dict
            prop_id = prop_data.get('id')
        elif isinstance(prop_id_or_dict, str):
            prop_id = prop_id_or_dict
            # If it's a string, it's an ID. Look up the full data from the game DB.
            prop_data = game_db['special_properties'].get(prop_id, {})

        # If after the above, we still don't have valid data, skip to the next item.
        if not prop_data or not prop_id:
            continue

        lang_id = find_best_lang_id(prop_data, prop_lang_subset, parent_block=special_data)
        lang_params = {}
        
        is_modifier_effect = 'modifier' in prop_data.get('propertyType', '').lower()
        main_template_text = lang_db.get(lang_id, {}).get("en", "")
        extra_lang_id = '.'.join(lang_id.split('.')[:4]) + ".extra"
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        all_placeholders = set(re.findall(r'\{(\w+)\}', main_template_text + extra_template_text))
        
        search_context = {**special_data, **prop_data}
        for p_holder in all_placeholders:
            if p_holder in lang_params: continue
            value, _ = find_and_calculate_value(p_holder, search_context, max_level, is_modifier_effect)
            if value is not None: lang_params[p_holder] = value
        
        if 'MAX' in all_placeholders and 'FIXEDPOWER' in lang_params: lang_params['MAX'] = lang_params['FIXEDPOWER'] * 2
        if 'MIN' in all_placeholders and 'FIXEDPOWER' in lang_params: lang_params['MIN'] = math.floor(lang_params['FIXEDPOWER'] / 2)

        nested_effects = []
        if 'statusEffectCollections' in prop_data:
            for collection in prop_data['statusEffectCollections']:
                collection_name = collection.get('collectionNameOverride', '')
                parsed_collection = parsers['status_effects'](collection.get('statusEffects', []), special_data, hero_stats, lang_db, game_db, parsers)
                for effect in parsed_collection: effect['collection_name'] = collection_name
                nested_effects.extend(parsed_collection)
        if 'statusEffects' in prop_data:
            nested_effects.extend(parsers['status_effects'](prop_data['statusEffects'], special_data, hero_stats, lang_db, game_db, parsers))

        template_str_for_check = main_template_text + extra_template_text
        formatted_params = {}
        for k, v in lang_params.items():
            formatted_val = format_value(v)
            is_percentage = f"{{{k}}}" in template_str_for_check and "%" in template_str_for_check
            if isinstance(v, (int, float)) and v > 0 and k.upper() not in ["TURNS", "DAMAGE", "MAX", "MIN", "FIXEDPOWER", "BASEPOWER", "MAXSTACK"] and is_percentage:
                 formatted_params[k] = f"+{formatted_val}"
            else: formatted_params[k] = formatted_val
        for p in all_placeholders:
             if p not in formatted_params: formatted_params[p] = f"{{{p}}}"

        main_desc = generate_description(lang_id, formatted_params, lang_db)
        tooltip_desc = generate_description(extra_lang_id, formatted_params, lang_db) if extra_lang_id in lang_db else {"en": "", "ja": ""}
        for d in [main_desc, tooltip_desc]:
            d['en'] = re.sub(r'\n\s*\n', '\n', d['en']).strip()
            d['ja'] = re.sub(r'\n\s*\n', '\n', d['ja']).strip()

        parsed_items.append({
            "id": prop_id, "lang_id": lang_id, "description_en": main_desc["en"], "description_ja": main_desc["ja"],
            "tooltip_en": tooltip_desc["en"], "tooltip_ja": tooltip_desc["ja"],
            "params": json.dumps(lang_params), "nested_effects": nested_effects
        })
    return parsed_items

def parse_status_effects(status_effects_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, parsers: dict) -> list:
    if not status_effects_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    se_lang_subset = parsers['se_lang_subset']
    
    for effect_instance in status_effects_list:
        effect_id = effect_instance.get("id")
        if not effect_id: continue

        # --- FIX: Explicitly fetch the base effect data from game_db to ensure completeness ---
        # This guarantees that base values like 'damageMultiplierPerMil' are always present.
        effect_details = game_db['status_effects'].get(effect_id, {})

        # The instance (from the parent) provides context like 'turns'.
        # The details (from the DB) provide the core skill data.
        combined_details = {**effect_details, **effect_instance}
        
        # Now, use the complete combined_details for all subsequent operations.
        lang_id = find_best_lang_id(combined_details, se_lang_subset, parent_block=special_data)

        lang_params = {}
        # Use combined_details to get the most accurate 'turns' value
        if (turns := combined_details.get("turns", 0)) > 0: 
            lang_params["TURNS"] = turns
        
        is_modifier_effect = 'modifier' in combined_details.get('statusEffect', '').lower()
        
        # The search scope includes the parent special data and the effect's own complete data
        search_context = {**special_data, **combined_details}

        template_text_en = lang_db.get(lang_id, {}).get("en", "")
        placeholders = set(re.findall(r'\{(\w+)\}', template_text_en))
        
        for p_holder in placeholders:
            if p_holder in lang_params: continue
            value, found_key = find_and_calculate_value(p_holder, search_context, max_level, is_modifier_effect)
            
            if value is not None:
                if p_holder.upper() == "DAMAGE" and "permil" in (found_key or "").lower():
                    turns_for_calc = combined_details.get("turns", 0) # Ensure we use the correct turn count
                    is_total = "over {TURNS} turns" in template_text_en
                    damage_per_turn = math.floor((value / 100) * hero_stats.get("max_attack", 0))
                    lang_params[p_holder] = damage_per_turn * (turns_for_calc or 1) if is_total else damage_per_turn
                else: 
                    lang_params[p_holder] = value
        
        nested_effects = []
        # Use combined_details to check for nested effects
        if 'statusEffectsToAdd' in combined_details:
             nested_effects.extend(parsers['status_effects'](combined_details['statusEffectsToAdd'], special_data, hero_stats, lang_db, game_db, parsers))

        formatted_params = {k: format_value(v) for k, v in lang_params.items()}
        descriptions = generate_description(lang_id, formatted_params, lang_db)
        descriptions['en'] = re.sub(r'\n\s*\n', '\n', descriptions['en']).strip()
        descriptions['ja'] = re.sub(r'\n\s*\n', '\n', descriptions['ja']).strip()
        
        parsed_items.append({
            "id": effect_id, "lang_id": lang_id, "params": json.dumps(lang_params),
            "nested_effects": nested_effects, **descriptions
        })
    return parsed_items

def parse_familiars(familiars_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, parsers: dict) -> list:
    if not familiars_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)

    for familiar_instance in familiars_list:
        if not (familiar_id := familiar_instance.get("id")): continue
        fam_type, fam_target = familiar_instance.get("familiarType"), familiar_instance.get("familiarTargetType")
        if not (fam_type and fam_target): continue

        lang_id = f"specials.v2.{fam_type.lower()}.{familiar_id}.{fam_target.lower()}"
        lang_params = {}
        template_text = lang_db.get(lang_id, {}).get("en", "")
        placeholders = set(re.findall(r'\{(\w+)\}', template_text))

        for p_holder in placeholders:
            value, _ = find_and_calculate_value(p_holder, familiar_instance, max_level)
            if value is not None: lang_params[p_holder] = value
        
        formatted_params = {k: format_value(v) for k, v in lang_params.items()}
        main_desc = generate_description(lang_id, formatted_params, lang_db)
        main_desc['en'], main_desc['ja'] = main_desc['en'].replace('[*]', '\n・').strip(), main_desc['ja'].replace('[*]', '\n・').strip()
        
        # --- NEW: Recursive parsing for familiar effects ---
        nested_effects = []
        if 'effects' in familiar_instance:
             # Placeholder for a future parse_familiar_effects parser
             pass

        parsed_items.append({
            "id": familiar_id, "lang_id": lang_id, "description_en": main_desc['en'],
            "description_ja": main_desc['ja'], "params": json.dumps(lang_params),
            "nested_effects": nested_effects
        })
    return parsed_items

# --- CSV Output Function ---
def write_results_to_csv(processed_data: list, output_path: Path):
    print(f"\n--- Writing results to {output_path} ---")
    if not processed_data: return
    
    all_rows = []
    
    for hero in processed_data:
        row = {'hero_id': hero.get('id'), 'hero_name': hero.get('name', 'N/A')}
        skills = hero.get('skillDescriptions', {})

        # --- Step 1: Add all standard columns first, maintaining original order ---
        if de := skills.get('directEffect'):
            row.update({f'de_{k}': v for k, v in de.items()})

        props = skills.get('properties', [])
        for i, p in enumerate(props[:3]):
            row.update({f'prop_{i+1}_{k}': v for k, v in p.items() if k != 'nested_effects'})

        effects = skills.get('statusEffects', [])
        for i, e in enumerate(effects[:5]):
            row.update({f'se_{i+1}_{k}': v for k, v in e.items() if k != 'nested_effects'})

        familiars = skills.get('familiars', [])
        for i, f in enumerate(familiars[:2]):
            row.update({f'fam_{i+1}_{k}': v for k, v in f.items() if k != 'nested_effects'})

        # --- Step 2: Append nested_effects columns at the very end ---
        # This ensures the original structure is not disturbed.
        nested_prop_effects = [p.get('nested_effects', []) for p in props[:3]]
        for i, nested_list in enumerate(nested_prop_effects):
            if not nested_list: continue
            for j, ne in enumerate(nested_list[:2]): # Limit to 2 nested effects per property
                row.update({f'prop_{i+1}_nested_{j+1}_{k}': v for k, v in ne.items() if k != 'nested_effects'})
        
        nested_se_effects = [e.get('nested_effects', []) for e in effects[:5]]
        for i, nested_list in enumerate(nested_se_effects):
            if not nested_list: continue
            for j, ne in enumerate(nested_list[:2]): # Limit to 2 nested effects per status effect
                 row.update({f'se_{i+1}_nested_{j+1}_{k}': v for k, v in ne.items() if k != 'nested_effects'})
        
        all_rows.append(row)

    try:
        # Using pandas to handle potentially different column sets between rows gracefully
        df = pd.DataFrame(all_rows)
        
        # Define a potential column order to keep things tidy, but it's not strictly necessary
        # The most important part is that new columns are added, not inserted.
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, lineterminator='\n')
        print(f"Successfully saved {len(df)} rows to CSV.")
    except Exception as e:
        print(f"FATAL: Failed to write CSV: {e}")


def write_debug_json(debug_data: dict, output_path: Path):
    print(f"\n--- Writing debug data to {output_path} ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved debug data for {len(debug_data)} heroes.")
    except Exception as e: print(f"FATAL: Failed to write debug JSON: {e}")


# --- Main Processing Function ---
def process_all_heroes(lang_db: dict, game_db: dict, hero_stats_db: dict, parsers: dict) -> (list, dict):
    print("\n--- Starting Hero Processing ---")
    all_heroes = game_db.get('heroes', [])
    processed_heroes_data = []
    all_heroes_debug_data = {}

    for i, hero in enumerate(all_heroes):
        hero_id = hero.get("id", "UNKNOWN")
        print(f"\r[{i+1}/{len(all_heroes)}] Processing: {hero_id.ljust(40)}", end="")
        
        full_hero_data = get_full_hero_data(hero, game_db)
        all_heroes_debug_data[hero_id] = full_hero_data
        hero_final_stats = get_hero_final_stats(hero_id, hero_stats_db)
        
        processed_hero = hero.copy()
        processed_hero['name'] = hero_final_stats.get('name')
        
        special_id = hero.get("specialId")
        if not special_id or not (special_data := game_db['character_specials'].get(special_id)):
            processed_hero['skillDescriptions'] = {}
            processed_heroes_data.append(processed_hero)
            continue
            
        # Extract top-level lists to be parsed
        prop_list = special_data.get("properties", [])
        se_list = special_data.get("statusEffects", [])
        familiar_list = special_data.get("summonedFamiliars", [])

        # --- MODIFIED: Pass the 'parsers' dict to each parser ---
        processed_hero['skillDescriptions'] = {
            'directEffect': parsers['direct_effect'](special_data, hero_final_stats, lang_db, game_db, parsers),
            'properties': parsers['properties'](prop_list, special_data, hero_final_stats, lang_db, game_db, parsers),
            'statusEffects': parsers['status_effects'](se_list, special_data, hero_final_stats, lang_db, game_db, parsers),
            'familiars': parsers['familiars'](familiar_list, special_data, hero_final_stats, lang_db, game_db, parsers)
        }
        processed_heroes_data.append(processed_hero)
    
    print("\n" + "--- Finished processing all heroes. ---")
    return processed_heroes_data, all_heroes_debug_data

def main():
    """Main function to run the entire process."""
    try:
        language_db = load_languages()
        game_db = load_game_data()
        hero_stats_db = load_hero_stats_from_csv(DATA_DIR, HERO_STATS_CSV_PATTERN)

        print("\nOptimizing language data for search...")
        # --- MODIFIED: Create a dictionary of all parser functions ---
        parsers = {
            'direct_effect': parse_direct_effect,
            'properties': parse_properties,
            'status_effects': parse_status_effects,
            'familiars': parse_familiars,
            # For quick access to language subsets inside parsers
            'se_lang_subset': [key for key in language_db if key.startswith("specials.v2.statuseffect.")],
            'prop_lang_subset': [key for key in language_db if key.startswith("specials.v2.property.")]
        }
        print(f" -> Found {len(parsers['se_lang_subset'])} status effect and {len(parsers['prop_lang_subset'])} property language keys.")

        final_hero_data, debug_data = process_all_heroes(language_db, game_db, hero_stats_db, parsers)
        
        write_results_to_csv(final_hero_data, OUTPUT_CSV_PATH)
        debug_output_path = SCRIPT_DIR / "debug_hero_data.json"
        write_debug_json(debug_data, debug_output_path)
        
        print(f"\nProcess complete. Output saved to {OUTPUT_CSV_PATH}")
        print(f"Debug data saved to {debug_output_path}")

        print("\n--- Analyzing unresolved placeholders in final output ---")
        from collections import Counter
        unresolved_counter = Counter()

        for hero in final_hero_data:
            if 'skillDescriptions' not in hero: continue
            items_to_check = []
            for skill_data in hero['skillDescriptions'].values():
                if isinstance(skill_data, list): items_to_check.extend(skill_data)
                elif isinstance(skill_data, dict): items_to_check.append(skill_data)

            idx = 0
            while idx < len(items_to_check):
                item = items_to_check[idx]
                idx += 1
                if not isinstance(item, dict): continue
                if 'nested_effects' in item and isinstance(item['nested_effects'], list):
                    items_to_check.extend(item['nested_effects'])
                for key, text in item.items():
                    if isinstance(text, str) and ('description' in key or 'tooltip' in key or key in ['en', 'ja']):
                        found = re.findall(r'(\{\w+\})', text)
                        if found: unresolved_counter.update(found)
        
        if not unresolved_counter: print("✅ All placeholders resolved successfully!")
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

if __name__ == "__main__":
    main()