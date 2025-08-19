# hero_main.py
# This is the main entry point for the Hero Skill Data Processor.
# It orchestrates the loading, parsing, and output writing processes in a two-phase approach.

import csv
import json
import traceback
from collections import Counter
from pathlib import Path
import pandas as pd
import re
from pprint import pformat

# --- Import custom modules ---
from hero_data_loader import (
    load_rules_from_csvs, load_languages, load_game_data, load_hero_stats_from_csv,
    DATA_DIR, SCRIPT_DIR as LOADER_SCRIPT_DIR, HERO_STATS_CSV_PATTERN
)
from hero_parser import (
    get_full_hero_data, get_hero_final_stats,
    parse_direct_effect, parse_properties, parse_status_effects,
    parse_familiars, parse_passive_skills, parse_clear_buffs
)

# --- Constants & Paths ---
SCRIPT_DIR = Path(__file__).parent
FINAL_CSV_PATH = SCRIPT_DIR / "hero_skill_output.csv"
DEBUG_CSV_PATH = SCRIPT_DIR / "hero_skill_output_debug.csv"
DEBUG_JSON_PATH = SCRIPT_DIR / "debug_hero_data.json"
FAMILIAR_LOG_PATH = SCRIPT_DIR / "familiar_debug_log.txt"

# --- Formatting & Output Functions ---

def _format_final_description(skill_descriptions: dict, lang: str, skill_types_to_include: list, special_data: dict) -> (str, list):
    """
    Formats a list of skill types into a main description string and a list of tooltips.
    Returns a tuple: (main_description_string, list_of_tooltip_strings)
    """
    output_lines = []
    tooltip_lines = [] # New list to collect tooltips
    
    local_skill_types_to_include = list(skill_types_to_include)

    if special_data and special_data.get("removeBuffsFirst"):
        if clear_buffs_item := skill_descriptions.get('clear_buffs'):
            description = clear_buffs_item.get(lang, "").strip()
            if description:
                output_lines.append(f"・{description}")
            if 'clear_buffs' in local_skill_types_to_include:
                local_skill_types_to_include.remove('clear_buffs')

    def process_level(items: list, is_passive=False):
        if not items:
            return
            
        # Passive skills are displayed in reverse order of definition.
        processed_items = reversed(items) if is_passive else items

        for item in processed_items:
            if not isinstance(item, dict):
                continue

            if is_passive:
                title = item.get(f'title_{lang}', "").strip()
                if title:
                    output_lines.append(f"\n- {title} -")

            description = item.get(lang, "").strip()
            # Fallback for older passive skill format
            if not description: description = item.get(f'description_{lang}', "").strip()

            if item.get("id") == "heading":
                output_lines.append(f"\n{description}")
            elif description:
                # Add bullet point, but not for passive descriptions that follow a title.
                prefix = "" if is_passive and 'title' in locals() and title else "・"
                output_lines.append(f"{prefix}{description}")

            # --- NEW: Check for and collect tooltips ---
            if 'extra' in item and isinstance(item['extra'], dict):
                tooltip_text = item['extra'].get(lang, "").strip()
                if tooltip_text:
                    tooltip_lines.append(tooltip_text)

            if 'nested_effects' in item and item['nested_effects']:
                process_level(item['nested_effects'], is_passive=False)

    for skill_type in local_skill_types_to_include:
        skill_data = skill_descriptions.get(skill_type)
        if not skill_data:
            continue
        
        items_to_process = skill_data if isinstance(skill_data, list) else [skill_data]
        is_passive_skill = (skill_type == 'passiveSkills')
        
        if is_passive_skill and any(items_to_process) and not any("--- Passives ---" in line for line in output_lines):
             output_lines.append("\n--- Passives ---")
            
        process_level(items_to_process, is_passive=is_passive_skill)
            
    main_description = "\n".join(line for line in output_lines if line).strip()
    return main_description, tooltip_lines


def write_final_csv(processed_data: list, output_path: Path):
    """Writes the main, human-readable CSV with separate columns for passives, specials, and tooltips."""
    print(f"\n--- Writing final results to {output_path.name} ---")
    if not processed_data:
        print("Warning: No data to write.")
        return
        
    output_rows = []
    ss_skill_types = ['directEffect', 'clear_buffs', 'properties', 'statusEffects', 'familiars']
    
    for hero in processed_data:
        skills = hero.get('skillDescriptions', {})
        special_context = hero.get('_special_data_context', {})

        # --- MODIFIED: Unpack tuples from _format_final_description ---
        passive_en_main, passive_en_tooltips = _format_final_description(skills, 'en', ['passiveSkills'], special_context)
        passive_ja_main, passive_ja_tooltips = _format_final_description(skills, 'ja', ['passiveSkills'], special_context)
        ss_en_main, ss_en_tooltips = _format_final_description(skills, 'en', ss_skill_types, special_context)
        ss_ja_main, ss_ja_tooltips = _format_final_description(skills, 'ja', ss_skill_types, special_context)
        
        row = {
            "hero_id": hero.get('id'),
            "hero_name": hero.get('name', 'N/A'),
            "passive_en": passive_en_main,
            "passive_ja": passive_ja_main,
            "ss_en": ss_en_main,
            "ss_ja": ss_ja_main,
        }

        # --- NEW: Add tooltip columns dynamically ---
        # Combine tooltips from both passive and special skills for each language
        all_tooltips_en = passive_en_tooltips + ss_en_tooltips
        all_tooltips_ja = passive_ja_tooltips + ss_ja_tooltips
        
        # Add up to 2 tooltips per language to the row
        for i in range(2):
            row[f'extra_en_{i+1}'] = all_tooltips_en[i] if i < len(all_tooltips_en) else ""
            row[f'extra_ja_{i+1}'] = all_tooltips_ja[i] if i < len(all_tooltips_ja) else ""

        output_rows.append(row)
        
    try:
        df = pd.DataFrame(output_rows)
        # Define column order to ensure extra columns are at the end
        column_order = [
            "hero_id", "hero_name", 
            "passive_en", "passive_ja", "ss_en", "ss_ja",
            "extra_en_1", "extra_ja_1", "extra_en_2", "extra_ja_2"
        ]
        df = df[column_order]
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
        # Define keys to extract from skill/extra dictionaries
        keys_to_keep = ['id', 'lang_id', 'params', 'collection_name']
        extra_keys_to_keep = ['lang_id', 'params']

        # Helper function to flatten and add data to the row
        def update_row_with_item(item, prefix):
            row.update({f'{prefix}_{k}': v for k, v in item.items() if k != 'nested_effects' and k in keys_to_keep})
            # --- NEW: Check for and add 'extra' (tooltip) data ---
            if 'extra' in item and isinstance(item['extra'], dict):
                row.update({f'{prefix}_extra_{k}': v for k, v in item['extra'].items() if k in extra_keys_to_keep})

        if de := skills.get('directEffect'):
            update_row_with_item(de, 'de')
        if cb := skills.get('clear_buffs'):
            update_row_with_item(cb, 'cb')
        
        props = skills.get('properties', [])
        for i, p in enumerate(props[:3]):
            update_row_with_item(p, f'prop_{i+1}')
            # Handle nested effects within properties
            if nested_effects := p.get('nested_effects', []):
                for j, ne in enumerate(nested_effects[:2]):
                    if isinstance(ne, dict):
                         update_row_with_item(ne, f'prop_{i+1}_nested_{j+1}')

        effects = skills.get('statusEffects', [])
        for i, e in enumerate(effects[:5]):
            update_row_with_item(e, f'se_{i+1}')
            # Handle nested effects within status effects
            if nested_effects := e.get('nested_effects', []):
                for j, ne in enumerate(nested_effects[:2]):
                    if isinstance(ne, dict):
                        update_row_with_item(ne, f'se_{i+1}_nested_{j+1}')
        
        familiars = skills.get('familiars', [])
        for i, f in enumerate(familiars[:2]):
            update_row_with_item(f, f'fam_{i+1}')
            # (Note: Familiars will be updated later to also output extra info for their effects)

        passives = skills.get('passiveSkills', [])
        for i, ps in enumerate(passives[:3]):
            # Passives do not have extra info, so no change here
            row.update({f'passive_{i+1}_{k}': v for k, v in ps.items() if k in keys_to_keep})

        all_rows.append(row)
        
    try:
        df = pd.DataFrame(all_rows)
        # Sort columns alphabetically for consistency, hero_id and hero_name first
        cols = sorted([col for col in df.columns if col not in ['hero_id', 'hero_name']])
        df = df[['hero_id', 'hero_name'] + cols]
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

# --- NEW: Two-Phase Processing Functions ---

def phase_one_integrate_data(game_db: dict, output_path: Path):
    """
    Phase 1: Loads all heroes, resolves all data dependencies,
    and writes the complete, unified data to debug_hero_data.json.
    """
    print("\n--- Phase 1: Integrating hero data and creating debug file ---")
    all_heroes = game_db.get('heroes', [])
    all_heroes_debug_data = {}
    total_heroes = len(all_heroes)
    for i, hero in enumerate(all_heroes):
        hero_id = hero.get("id", "UNKNOWN")
        print(f"\r[{i+1}/{total_heroes}] Integrating data for: {hero_id.ljust(40)}", end="")
        full_hero_data = get_full_hero_data(hero, game_db)
        all_heroes_debug_data[hero_id] = full_hero_data
    
    write_debug_json(all_heroes_debug_data, output_path)
    print(f"\n--- Phase 1 Complete. {len(all_heroes_debug_data)} heroes integrated. ---")

def phase_two_parse_skills(debug_data: dict, lang_db: dict, game_db: dict, hero_stats_db: dict, rules: dict, parsers: dict) -> list:
    """
    Phase 2: Loads the unified data from debug_hero_data.json and parses all skills.
    """
    print("\n--- Phase 2: Parsing skills from unified data ---")
    processed_heroes_data = []
    
    # Initialize the final warning collectors
    parsers['warnings_list'] = []
    parsers['unique_warnings_set'] = set()
    parsers['familiar_debug_log'] = []
    parsers['familiar_parameter_log'] = []

    total_heroes = len(debug_data)
    for i, (hero_id, full_hero_data) in enumerate(debug_data.items()):
        print(f"\r[{i+1}/{total_heroes}] Parsing skills for: {hero_id.ljust(40)}", end="")
        
        hero_final_stats = get_hero_final_stats(hero_id, hero_stats_db)
        processed_hero = full_hero_data.copy()
        processed_hero['name'] = hero_final_stats.get('name')
        
        skill_descriptions = {}
        special_data_for_hero = None
        
        # --- MODIFIED: Unpack tuples returned from parsers and collect warnings ---
        def collect_warnings(new_warnings):
            if not new_warnings: return
            for w in new_warnings:
                if w not in parsers['unique_warnings_set']:
                    parsers['unique_warnings_set'].add(w)
                    parsers['warnings_list'].append(w)

        if special_data := full_hero_data.get("specialId_details"):
            special_data_for_hero = special_data
            parsers["hero_mana_speed_id"] = full_hero_data.get("manaSpeedId")
            
            # Note: parse_direct_effect was not modified as it doesn't generate warnings
            skill_descriptions['directEffect'] = parsers['direct_effect'](special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            
            parsed_clear_buffs, new_warnings = parsers['clear_buffs'](special_data, lang_db, parsers)
            skill_descriptions['clear_buffs'] = parsed_clear_buffs
            collect_warnings(new_warnings)

            prop_list = special_data.get("properties", [])
            parsed_properties, new_warnings = parsers['properties'](prop_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['properties'] = parsed_properties
            collect_warnings(new_warnings)
            
            se_list = special_data.get("statusEffects", [])
            parsed_status_effects, new_warnings = parsers['status_effects'](se_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['statusEffects'] = parsed_status_effects
            collect_warnings(new_warnings)

            familiar_list = special_data.get("summonedFamiliars", [])
            parsed_familiars, new_warnings = parsers['familiars'](familiar_list, special_data, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['familiars'] = parsed_familiars
            collect_warnings(new_warnings)

        passive_list = full_hero_data.get('passiveSkills', [])
        costume_passive_list = []
        if costume_bonuses := full_hero_data.get('costumeBonusesId_details'):
            if isinstance(costume_bonuses, dict):
                 costume_passive_list = costume_bonuses.get('passiveSkills', [])

        all_passives = passive_list + costume_passive_list
        if all_passives:
            parsed_passives, new_warnings = parsers['passive_skills'](all_passives, hero_final_stats, lang_db, game_db, hero_id, rules, parsers)
            skill_descriptions['passiveSkills'] = parsed_passives
            collect_warnings(new_warnings)
        
        processed_hero['_special_data_context'] = special_data_for_hero
        processed_hero['skillDescriptions'] = {k: v for k, v in skill_descriptions.items() if v}
        processed_heroes_data.append(processed_hero)
    
    print("\n--- Phase 2 Complete ---")
    return processed_heroes_data

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
        rules = load_rules_from_csvs(LOADER_SCRIPT_DIR)
        language_db = load_languages()
        game_db = load_game_data()
        hero_stats_db = load_hero_stats_from_csv(DATA_DIR, HERO_STATS_CSV_PATTERN)

        phase_one_integrate_data(game_db, DEBUG_JSON_PATH)

        print("\nReloading unified data from file to ensure consistency...")
        with open(DEBUG_JSON_PATH, 'r', encoding='utf-8') as f:
            debug_data_from_file = json.load(f)

        parsers = {
            'direct_effect': parse_direct_effect, 'clear_buffs': parse_clear_buffs,
            'properties': parse_properties, 'status_effects': parse_status_effects,
            'familiars': parse_familiars, 'passive_skills': parse_passive_skills,
            'se_lang_subset': [key for key in language_db if key.startswith("specials.v2.statuseffect.")],
            'prop_lang_subset': [key for key in language_db if key.startswith("specials.v2.property.")]
        }
        
        final_hero_data = phase_two_parse_skills(debug_data_from_file, language_db, game_db, hero_stats_db, rules, parsers)
        
        write_final_csv(final_hero_data, FINAL_CSV_PATH)
        write_debug_csv(final_hero_data, DEBUG_CSV_PATH)
        
        # --- NEW: Write the new parameter log to a CSV file ---
        param_log = parsers.get('familiar_parameter_log', [])
        if param_log:
            param_log_path = SCRIPT_DIR / "familiar_parameter_log.csv"
            print(f"\n--- 📝 Writing familiar parameter log... ---")
            try:
                param_df = pd.DataFrame(param_log)
                param_df.to_csv(param_log_path, index=False, encoding='utf-8-sig')
                print(f"Details saved to {param_log_path.name}")
            except Exception as e:
                print(f"Warning: Could not write familiar parameter log. Error: {e}")
        
        warnings_list = parsers.get('warnings_list', [])
        if warnings_list:
            unique_warnings = parsers.get('unique_warnings_set', set())
            print(f"\n--- 🚨 Found {len(warnings_list)} lang_id search failures ({len(unique_warnings)} unique types) ---")
        
        analyze_unresolved_placeholders(final_hero_data)
        
        print(f"\n✅ Process complete. All files saved.")

    except Exception as e:
        print(f"\n[FATAL ERROR]: {type(e).__name__} - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()