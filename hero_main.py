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

def _format_final_description(skill_descriptions: dict, lang: str, skill_types_to_include: list, special_data: dict) -> str:
    """
    A helper function to format a SPECIFIC LIST of skill types into a single string.
    """
    output_lines = []
    
    local_skill_types_to_include = list(skill_types_to_include)

    if special_data and special_data.get("removeBuffsFirst"):
        # The 'clear_buffs' key should exist if buffToRemove was present
        if clear_buffs_item := skill_descriptions.get('clear_buffs'):
            desc_key = f'description_{lang}' if f'description_{lang}' in clear_buffs_item else lang
            description = clear_buffs_item.get(desc_key, "").strip()
            if description:
                output_lines.append(f"・{description}")
            if 'clear_buffs' in local_skill_types_to_include:
                local_skill_types_to_include.remove('clear_buffs')

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
            if not description and lang in item:
                description = item.get(lang, "").strip()

            if item.get("id") == "heading":
                output_lines.append(f"\n{description}")
            elif description:
                prefix = "" if is_passive and title else "・"
                output_lines.append(f"{prefix}{description}")

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
            
    return "\n".join(line for line in output_lines if line).strip()


def write_final_csv(processed_data: list, output_path: Path):
    """Writes the main, human-readable CSV with separate columns for passives and specials."""
    print(f"\n--- Writing final results to {output_path.name} ---")
    if not processed_data:
        print("Warning: No data to write.")
        return
        
    output_rows = []
    ss_skill_types = ['directEffect', 'clear_buffs', 'properties', 'statusEffects', 'familiars']
    
    for hero in processed_data:
        skills = hero.get('skillDescriptions', {})
        special_context = hero.get('_special_data_context', {})
        output_rows.append({
            "hero_id": hero.get('id'),
            "hero_name": hero.get('name', 'N/A'),
            "passive_en": _format_final_description(skills, 'en', ['passiveSkills'], special_context),
            "passive_ja": _format_final_description(skills, 'ja', ['passiveSkills'], special_context),
            "ss_en": _format_final_description(skills, 'en', ss_skill_types, special_context),
            "ss_ja": _format_final_description(skills, 'ja', ss_skill_types, special_context),
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
        keys_to_keep = ['id', 'lang_id', 'params', 'collection_name']

        if de := skills.get('directEffect'):
            row.update({f'de_{k}': v for k, v in de.items() if k in keys_to_keep})
        if cb := skills.get('clear_buffs'):
             row.update({f'cb_{k}': v for k, v in cb.items() if k in keys_to_keep})
        props = skills.get('properties', [])
        for i, p in enumerate(props[:3]):
            row.update({f'prop_{i+1}_{k}': v for k, v in p.items() if k != 'nested_effects' and k in keys_to_keep})
        effects = skills.get('statusEffects', [])
        for i, e in enumerate(effects[:5]):
            row.update({f'se_{i+1}_{k}': v for k, v in e.items() if k != 'nested_effects' and k in keys_to_keep})
        familiars = skills.get('familiars', [])
        for i, f in enumerate(familiars[:2]):
            row.update({f'fam_{i+1}_{k}': v for k, v in f.items() if k != 'nested_effects' and k in keys_to_keep})
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
    
    parsers['warnings_list'] = []
    parsers['unique_warnings_set'] = set()
    parsers['familiar_debug_log'] = []

    total_heroes = len(debug_data)
    for i, (hero_id, full_hero_data) in enumerate(debug_data.items()):
        print(f"\r[{i+1}/{total_heroes}] Parsing skills for: {hero_id.ljust(40)}", end="")
        
        hero_final_stats = get_hero_final_stats(hero_id, hero_stats_db)
        processed_hero = full_hero_data.copy()
        processed_hero['name'] = hero_final_stats.get('name')
        
        skill_descriptions = {}
        special_data_for_hero = None
        
        special_id = full_hero_data.get("specialId")
        if special_id and (special_data := game_db['character_specials'].get(special_id)):
            special_data_for_hero = special_data
            parsers["hero_mana_speed_id"] = full_hero_data.get("manaSpeedId")
            
            prop_list = special_data.get("properties", [])
            se_list = special_data.get("statusEffects", [])
            familiar_list = special_data.get("summonedFamiliars", [])

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

        # --- Phase 1: Create the unified, trustworthy data source ---
        phase_one_integrate_data(game_db, DEBUG_JSON_PATH)

        # --- Phase 2: Parse skills using only the unified data source ---
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
        
        # --- Step 3: Write all output files ---
        write_final_csv(final_hero_data, FINAL_CSV_PATH)
        write_debug_csv(final_hero_data, DEBUG_CSV_PATH)
        
        # --- Step 4: Final Reporting ---
        familiar_log = parsers.get('familiar_debug_log', [])
        if familiar_log:
            print(f"\n--- 📝 Found {len(familiar_log)} familiar parsing issues. ---")
            with open(FAMILIAR_LOG_PATH, 'w', encoding='utf-8') as f:
                for entry in familiar_log:
                    f.write(pformat(entry, indent=2))
                    f.write("\n" + "="*80 + "\n")
            print(f"Details saved to {FAMILIAR_LOG_PATH.name}")

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