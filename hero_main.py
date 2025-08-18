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
    parse_passive_skills,
    parse_clear_buffs
)

# --- Constants & Paths ---
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUTPUT_CSV_PATH = SCRIPT_DIR / "hero_skill_output.csv"
DEBUG_JSON_PATH = SCRIPT_DIR / "debug_hero_data.json"

# --- CSV Output Function ---
def _format_final_description(skill_descriptions: dict, lang: str, skill_types_to_include: list, special_data: dict) -> str:
    """
    A helper function to format a SPECIFIC LIST of skill types into a single string.
    """
    output_lines = []
    
    # Handle the 'removeBuffsFirst' logic at the very beginning
    if special_data and special_data.get("removeBuffsFirst"):
        if clear_buffs_item := skill_descriptions.get('clear_buffs'):
            desc_key = f'description_{lang}' if f'description_{lang}' in clear_buffs_item else lang
            description = clear_buffs_item.get(desc_key, "").strip()
            if description:
                output_lines.append(f"・{description}")

    def process_level(items: list, is_passive=False):
        if not items:
            return
            
        processed_items = reversed(items) if is_passive else items

        for item in items:
            if not isinstance(item, dict):
                continue

            if is_passive:
                title_key = f'title_{lang}'
                title = item.get(title_key, "").strip()
                if title:
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

    for skill_type in skill_types_to_include:
        skill_data = skill_descriptions.get(skill_type)
        if not skill_data:
            continue
        
        items_to_process = skill_data if isinstance(skill_data, list) else [skill_data]
        is_passive_skill = (skill_type == 'passiveSkills')
        
        if is_passive_skill and any(items_to_process) and not any("--- Passives ---" in line for line in output_lines):
             output_lines.append("\n--- Passives ---")
            
        process_level(items_to_process, is_passive=is_passive_skill)
            
    return "\n".join(line for line in output_lines if line).strip()


def write_final_csv(processed_data: list, output_path: Path):
    """Writes the main, human-readable CSV with separate columns for passives and specials."""
    print(f"\n--- Writing final results to {output_path.name} ---")
    if not processed_data:
        print("Warning: No data to write.")
        return
        
    output_rows = []
    
    ss_skill_types = ['directEffect', 'properties', 'statusEffects', 'familiars', 'clear_buffs']
    
    for hero in processed_data:
        skills = hero.get('skillDescriptions', {})
        # The original special_data is needed to check for 'removeBuffsFirst'
        special_context = hero.get('_special_data_context', {})

        output_rows.append({
            "hero_id": hero.get('id'),
            "hero_name": hero.get('name', 'N/A'),
            "passive_en": _format_final_description(skills, 'en', skill_types_to_include=['passiveSkills'], special_data=special_context),
            "passive_ja": _format_final_description(skills, 'ja', skill_types_to_include=['passiveSkills'], special_data=special_context),
            "ss_en": _format_final_description(skills, 'en', skill_types_to_include=ss_skill_types, special_data=special_context),
            "ss_ja": _format_final_description(skills, 'ja', skill_types_to_include=ss_skill_types, special_data=special_context),
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

    parsers['warnings_list'] = []
    parsers['unique_warnings_set'] = set()

    for i, hero in enumerate(all_heroes):
        hero_id = hero.get("id", "UNKNOWN")
        print(f"\r[{i+1}/{len(all_heroes)}] Processing: {hero_id.ljust(40)}", end="")
        
        full_hero_data = get_full_hero_data(hero, game_db)
        all_heroes_debug_data[hero_id] = full_hero_data
        hero_final_stats = get_hero_final_stats(hero_id, hero_stats_db)
        
        processed_hero = hero.copy()
        processed_hero['name'] = hero_final_stats.get('name')
        
        skill_descriptions = {}
        special_data_for_hero = None
        
        special_id = hero.get("specialId")
        if special_id and (special_data := game_db['character_specials'].get(special_id)):
            special_data_for_hero = special_data # Store for later use
            parsers["hero_mana_speed_id"] = hero.get("manaSpeedId")
            
            # --- MODIFIED: Call all relevant parsers for the special skill ---
            prop_list = special_data.get("properties", [])
            se_list = special_data.get("statusEffects", [])
            familiar_list = special_data.get("summonedFamiliars", [])

            # Each parser returns its result, or None
            skill_descriptions['directEffect'] = parsers['direct_effect'](special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['clear_buffs'] = parsers['clear_buffs'](special_data, lang_db, parsers)
            skill_descriptions['properties'] = parsers['properties'](prop_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['statusEffects'] = parsers['status_effects'](se_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['familiars'] = parsers['familiars'](familiar_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
        
        passive_list = full_hero_data.get('passiveSkills', [])
        costume_passive_list = []
        if costume_bonuses := full_hero_data.get('costumeBonusesId_details'):
            if isinstance(costume_bonuses, dict):
                 costume_passive_list = costume_bonuses.get('passiveSkills', [])

        all_passives = passive_list + costume_passive_list
        if all_passives:
            skill_descriptions['passiveSkills'] = parsers['passive_skills'](all_passives, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
        
        # --- NEW: Store the original special_data for the formatting function ---
        processed_hero['_special_data_context'] = special_data_for_hero
        processed_hero['skillDescriptions'] = {k: v for k, v in skill_descriptions.items() if v} # Clean up None results
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
        rules = load_rules_from_csvs(SCRIPT_DIR)
        language_db = load_languages()
        game_db = load_game_data()
        hero_stats_db = load_hero_stats_from_csv(DATA_DIR, HERO_STATS_CSV_PATTERN)

        print("\nOptimizing language data for search...")
        parsers = {
            'direct_effect': parse_direct_effect,
            'clear_buffs': parse_clear_buffs,
            'properties': parse_properties,
            'status_effects': parse_status_effects,
            'familiars': parse_familiars,
            'passive_skills': parse_passive_skills,
            'se_lang_subset': [key for key in language_db if key.startswith("specials.v2.statuseffect.")],
            'prop_lang_subset': [key for key in language_db if key.startswith("specials.v2.property.")]
        }
        print(f" -> Found {len(parsers['se_lang_subset'])} status effect and {len(parsers['prop_lang_subset'])} property language keys.")

        final_hero_data, debug_data = process_all_heroes(language_db, game_db, hero_stats_db, rules, parsers)
        
        final_csv_path = SCRIPT_DIR / "hero_skill_output.csv"
        debug_csv_path = SCRIPT_DIR / "hero_skill_output_debug.csv"
        debug_json_path = SCRIPT_DIR / "debug_hero_data.json"

        write_final_csv(final_hero_data, final_csv_path)
        write_debug_csv(final_hero_data, debug_csv_path)
        write_debug_json(debug_data, debug_json_path)
        
        print(f"\nProcess complete. All files saved.")

        warnings_list = parsers.get('warnings_list', [])
        if warnings_list:
            unique_warnings = parsers.get('unique_warnings_set', set())
            print(f"\n--- 🚨 Found {len(warnings_list)} lang_id search failures ({len(unique_warnings)} unique types) ---")
        
        analyze_unresolved_placeholders(final_hero_data)

    except Exception as e:
        print(f"\n[FATAL ERROR]: {type(e).__name__} - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()