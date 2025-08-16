# hero_main.py
# This is the main entry point for the Hero Skill Data Processor.
# It orchestrates the loading, parsing, and output writing processes.

import csv
import json
import traceback
from collections import Counter
from pathlib import Path
import pandas as pd
import re
import pandas as pd


# --- Import custom modules ---
# These modules contain the detailed logic for loading and parsing.
from hero_data_loader import (
    load_rules_from_csvs,
    load_languages,
    load_game_data,
    load_hero_stats_from_csv,
    DATA_DIR,
    HERO_STATS_CSV_PATTERN
)
from hero_parser import (
    get_full_hero_data,
    get_hero_final_stats,
    parse_direct_effect,
    parse_properties,
    parse_status_effects,
    parse_familiars,
    parse_passive_skills
)

# --- Constants & Paths ---
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_CSV_PATH = SCRIPT_DIR / "hero_skill_output.csv"
DEBUG_JSON_PATH = SCRIPT_DIR / "debug_hero_data.json"

# --- CSV Output Function ---
def _format_final_description(skill_descriptions: dict, lang: str) -> str:
    """
    A helper function to recursively traverse skill data and format it into a single string.
    """
    output_lines = []
    
    skill_order = ['directEffect', 'properties', 'statusEffects', 'familiars', 'passiveSkills']
    
    def process_level(items: list, is_passive=False):
        if not items:
            return
            
        for item in items:
            if not isinstance(item, dict):
                continue

            if is_passive:
                title_key = f'title_{lang}'
                title = item.get(title_key, "").strip()
                if title:
                    # Add a separator and the title for each passive
                    output_lines.append(f"\n- {title} -")

            desc_key = f'description_{lang}' if f'description_{lang}' in item else lang
            description = item.get(desc_key, "").strip()

            if item.get("id") == "heading":
                output_lines.append(f"\n{description}")
            elif description:
                prefix = "・" if not is_passive else ""
                output_lines.append(f"{prefix}{description}")

            if 'nested_effects' in item and item['nested_effects']:
                process_level(item['nested_effects'], is_passive=False)

    for skill_type in skill_order:
        skill_data = skill_descriptions.get(skill_type)
        if not skill_data:
            continue
        
        items_to_process = skill_data if isinstance(skill_data, list) else [skill_data]
        is_passive_skill = (skill_type == 'passiveSkills')
        
        if is_passive_skill and any(items_to_process) and output_lines:
             output_lines.append("\n--- Passives ---")
            
        process_level(items_to_process, is_passive=is_passive_skill)
            
    return "\n".join(line for line in output_lines if line).strip()


def write_final_csv(processed_data: list, output_path: Path):
    """Writes the main, human-readable CSV with final formatted descriptions."""
    print(f"\n--- Writing final results to {output_path.name} ---")
    if not processed_data:
        print("Warning: No data to write.")
        return
        
    output_rows = []
    for hero in processed_data:
        output_rows.append({
            "hero_id": hero.get('id'),
            "hero_name": hero.get('name', 'N/A'),
            "final_description_en": _format_final_description(hero.get('skillDescriptions', {}), 'en'),
            "final_description_ja": _format_final_description(hero.get('skillDescriptions', {}), 'ja'),
        })
        
    try:
        df = pd.DataFrame(output_rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, lineterminator='\n')
        print(f"Successfully saved {len(df)} rows to {output_path.name}.")
    except Exception as e:
        print(f"FATAL: Failed to write final CSV: {e}")


def write_debug_csv(processed_data: list, output_path: Path):
    """Writes the debug CSV with structural and numerical data only (no long texts)."""
    print(f"\n--- Writing debug data to {output_path.name} ---")
    if not processed_data:
        print("Warning: No data to write.")
        return
    
    all_rows = []
    for hero in processed_data:
        row = {'hero_id': hero.get('id'), 'hero_name': hero.get('name', 'N/A')}
        skills = hero.get('skillDescriptions', {})

        # --- Keep only ID, lang_id, and params ---
        keys_to_keep = ['id', 'lang_id', 'params', 'collection_name']

        if de := skills.get('directEffect'):
            row.update({f'de_{k}': v for k, v in de.items() if k in keys_to_keep})

        props = skills.get('properties', [])
        for i, p in enumerate(props[:3]):
            row.update({f'prop_{i+1}_{k}': v for k, v in p.items() if k in keys_to_keep})

        effects = skills.get('statusEffects', [])
        for i, e in enumerate(effects[:5]):
            row.update({f'se_{i+1}_{k}': v for k, v in e.items() if k in keys_to_keep})

        familiars = skills.get('familiars', [])
        for i, f in enumerate(familiars[:2]):
            row.update({f'fam_{i+1}_{k}': v for k, v in f.items() if k in keys_to_keep})
            
        passives = skills.get('passiveSkills', [])
        for i, ps in enumerate(passives[:3]):
            row.update({f'passive_{i+1}_{k}': v for k, v in ps.items() if k in keys_to_keep})

        nested_prop_effects = [p.get('nested_effects', []) for p in props[:3]]
        for i, nested_list in enumerate(nested_prop_effects):
            if not nested_list: continue
            for j, ne in enumerate(nested_list[:2]):
                row.update({f'prop_{i+1}_nested_{j+1}_{k}': v for k, v in ne.items() if k in keys_to_keep})
        
        nested_se_effects = [e.get('nested_effects', []) for e in effects[:5]]
        for i, nested_list in enumerate(nested_se_effects):
            if not nested_list: continue
            for j, ne in enumerate(nested_list[:2]):
                 row.update({f'se_{i+1}_nested_{j+1}_{k}': v for k, v in ne.items() if k in keys_to_keep})
        
        all_rows.append(row)

    try:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL, lineterminator='\n')
        print(f"Successfully saved {len(df)} rows to {output_path.name}.")
    except Exception as e:
        print(f"FATAL: Failed to write debug CSV: {e}")


def write_debug_json(debug_data: dict, output_path: Path):
    """Writes the fully resolved hero data to a JSON file for debugging."""
    print(f"\n--- Writing debug data to {output_path.name} ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved debug data for {len(debug_data)} heroes.")
    except Exception as e:
        print(f"FATAL: Failed to write debug JSON: {e}")

# --- Main Processing Function ---
def process_all_heroes(lang_db: dict, game_db: dict, hero_stats_db: dict, rules: dict, parsers: dict) -> (list, dict):
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
        
        # --- Active Skill (Special) Parsing ---
        special_id = hero.get("specialId")
        skill_descriptions = {}
        
        if not special_id or not (special_data := game_db['character_specials'].get(special_id)):
            processed_hero['skillDescriptions'] = {}
        else:
            parsers["hero_mana_speed_id"] = hero.get("manaSpeedId")
                
            prop_list = special_data.get("properties", [])
            se_list = special_data.get("statusEffects", [])
            familiar_list = special_data.get("summonedFamiliars", [])

            skill_descriptions = {
                'directEffect': parsers['direct_effect'](special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers),
                'properties': parsers['properties'](prop_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers),
                'statusEffects': parsers['status_effects'](se_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers),
                'familiars': parsers['familiars'](familiar_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            }
        
        # --- NEW: Passive Skill Parsing ---
        passive_list = full_hero_data.get('passiveSkills', [])
        costume_passive_list = full_hero_data.get('costumeBonusPassiveSkillIds_details', []) # Assuming this is the resolved key name
        
        all_passives = passive_list + costume_passive_list
        if all_passives:
            skill_descriptions['passiveSkills'] = parsers['passive_skills'](all_passives, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            
        processed_hero['skillDescriptions'] = skill_descriptions
        processed_heroes_data.append(processed_hero)
    
    print("\n" + "--- Finished processing all heroes. ---")
    return processed_heroes_data, all_heroes_debug_data

def analyze_unresolved_placeholders(final_hero_data: list):
    """Analyzes the final output and prints a summary of unresolved placeholders."""
    print("\n--- Analyzing unresolved placeholders in final output ---")
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
    
    if not unresolved_counter:
        print("✅ All placeholders resolved successfully!")
    else:
        print(f"{'Placeholder':<30} | {'Count':<10}")
        print("-" * 43)
        for placeholder, count in unresolved_counter.most_common():
            print(f"{placeholder:<30} | {count:<10}")
        print("-" * 43)
        print(f"Total Unique Unresolved Placeholders: {len(unresolved_counter)}")

# --- Main Execution Block ---
def main():
    """Main function to run the entire process."""
    try:
        # Step 1: Load all data
        rules = load_rules_from_csvs(SCRIPT_DIR)
        language_db = load_languages()
        game_db = load_game_data()
        hero_stats_db = load_hero_stats_from_csv(DATA_DIR, HERO_STATS_CSV_PATTERN)

        # Step 2: Prepare parser functions
        print("\nOptimizing language data for search...")
        parsers = {
            'direct_effect': parse_direct_effect,
            'properties': parse_properties,
            'status_effects': parse_status_effects,
            'familiars': parse_familiars,
            'passive_skills': parse_passive_skills, # Add the new parser
            'se_lang_subset': [key for key in language_db if key.startswith("specials.v2.statuseffect.")],
            'prop_lang_subset': [key for key in language_db if key.startswith("specials.v2.property.")]
        }
        print(f" -> Found {len(parsers['se_lang_subset'])} status effect and {len(parsers['prop_lang_subset'])} property language keys.")

        # Step 3: Process all heroes
        final_hero_data, debug_data = process_all_heroes(language_db, game_db, hero_stats_db, rules, parsers)
        
        # Step 4: Write the output files
        final_csv_path = SCRIPT_DIR / "hero_skill_output.csv"
        debug_csv_path = SCRIPT_DIR / "hero_skill_output_debug.csv"
        debug_json_path = SCRIPT_DIR / "debug_hero_data.json"

        write_final_csv(final_hero_data, final_csv_path)
        write_debug_csv(final_hero_data, debug_csv_path)
        write_debug_json(debug_data, debug_json_path)
        
        print(f"\nProcess complete. All files saved.")

        # Step 5: Analyze and report any remaining issues
        analyze_unresolved_placeholders(final_hero_data)

    except Exception as e:
        print(f"\n[FATAL ERROR]: {type(e).__name__} - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()