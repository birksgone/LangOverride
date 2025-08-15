# hero_parser.py
# This module contains all the logic for parsing and interpreting hero skill data.

import json
import re
import math
import pandas as pd

# --- Helper Functions (used only by the parsers) ---

def flatten_json(y):
    """ Flattens a nested dictionary and list structure. """
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else: out[name[:-1]] = x
    flatten(y)
    return out

def generate_description(lang_id: str, lang_params: dict, lang_db: dict) -> dict:
    """Generates a description string by filling a template with parameters."""
    template = lang_db.get(lang_id, {"en": f"NO_TEMPLATE_FOR_{lang_id}", "ja": f"NO_TEMPLATE_FOR_{lang_id}"})
    desc_en, desc_ja = template.get("en", ""), template.get("ja", "")
    for key, value in lang_params.items():
        desc_en = desc_en.replace(f"{{{key}}}", str(value))
        desc_ja = desc_ja.replace(f"{{{key}}}", str(value))
    return {"en": desc_en, "ja": desc_ja}

def format_value(value):
    """Formats numbers for display, removing trailing .0"""
    if isinstance(value, float) and value.is_integer(): return int(value)
    if isinstance(value, float): return f"{value:.1f}"
    return value

# --- Core Data Integration Logic ---

def get_full_hero_data(base_data: dict, game_db: dict) -> dict:
    """Recursively resolves all IDs in a hero's data to build a complete data object."""
    resolved_data = json.loads(json.dumps(base_data))
    processed_ids = set()
    _resolve_recursive(resolved_data, game_db['master_db'], processed_ids)
    return resolved_data

def _resolve_recursive(current_data, master_db, processed_ids):
    """Internal recursive function to resolve and merge data based on IDs."""
    if id(current_data) in processed_ids: return
    processed_ids.add(id(current_data))
    
    ID_KEYS_FOR_LISTS = [
        'properties', 'statusEffects', 'statusEffectsPerHit',
        'summonedFamiliars', 'effects', 'passiveSkills', 'costumeBonusPassiveSkillIds',
        'statusEffectsToAdd', 'statusEffectCollections'
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

# --- Analysis & Parsing Functions ---

def get_hero_final_stats(hero_id: str, hero_stats_db: dict) -> dict:
    """Calculates a hero's final attack stat, considering costume bonuses."""
    hero_data = hero_stats_db.get(hero_id)
    if not hero_data: return {"max_attack": 0, "name": "N/A"}
    attack_col = 'Max level: Attack'
    for i in range(4, 0, -1):
        col_name = f'Max level CB{i}: Attack'
        if col_name in hero_data and pd.notna(hero_data[col_name]):
            attack_col = col_name
            break
    return {"max_attack": int(hero_data.get(attack_col, 0)), "name": hero_data.get('Name', 'N/A')}

def find_and_calculate_value(p_holder: str, data_block: dict, max_level: int, hero_id: str, rules: dict, is_modifier: bool = False) -> (any, str):
    p_holder_upper = p_holder.upper()
    
    rule = rules.get("hero_rules", {}).get("specific", {}).get(hero_id, {}).get(p_holder_upper)
    if not rule:
        rule = rules.get("hero_rules", {}).get("common", {}).get(p_holder_upper)

    if rule:
        calc_method = rule.get("calc")
        if calc_method == "fixed":
            return rule.get("value"), "Fixed Rule"

        key_to_find = rule.get("key")
        if key_to_find:
            flat_data = flatten_json(data_block)
            matching_keys = [k for k in flat_data if k.endswith(key_to_find)]
            
            if len(matching_keys) == 1:
                found_key = matching_keys[0]
                value = flat_data[found_key]
                if isinstance(value, (int, float)):
                    if 'permil' in found_key.lower():
                        return value / 10, f"Exception Rule: {found_key}"
                    return int(value), f"Exception Rule: {found_key}"
        return None, f"Exception rule key '{key_to_find}' not found or ambiguous"

    if not isinstance(data_block, dict): return None, None
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
    """Finds the best language ID for a skill block using a scoring system."""
    if 'statusEffect' in data_block:
        buff_map = {
            "MinorDebuff": "minor", "MajorDebuff": "major",
            "MinorBuff": "minor", "MajorBuff": "major",
            "PermanentDebuff": "permanent", "PermanentBuff": "permanent"
        }
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

def parse_direct_effect(special_data, hero_stats, lang_db, game_db, hero_id: str, rules: dict, parsers: dict):
    """Parses the directEffect block of a special skill."""
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

def parse_properties(properties_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> list:
    if not properties_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    prop_lang_subset = parsers['prop_lang_subset']
    
    for prop_id_or_dict in properties_list:
        prop_data, prop_id = {}, None
        
        if isinstance(prop_id_or_dict, dict):
            prop_data, prop_id = prop_id_or_dict, prop_id_or_dict.get('id')
        elif isinstance(prop_id_or_dict, str):
            prop_id, prop_data = prop_id_or_dict, game_db['special_properties'].get(prop_id_or_dict, {})

        if not prop_data or not prop_id: continue

        lang_id = rules.get("lang_overrides", {}).get("specific", {}).get(hero_id, {}).get(prop_id)
        if not lang_id: lang_id = rules.get("lang_overrides", {}).get("common", {}).get(prop_id)
        if not lang_id: lang_id = find_best_lang_id(prop_data, prop_lang_subset, parent_block=special_data)

        lang_params, is_modifier_effect = {}, 'modifier' in prop_data.get('propertyType', '').lower()
        main_template_text = lang_db.get(lang_id, {}).get("en", "")
        extra_lang_id = '.'.join(lang_id.split('.')[:4]) + ".extra"
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        all_placeholders = set(re.findall(r'\{(\w+)\}', main_template_text + extra_template_text))

        search_context = {**special_data, **prop_data}
        for p_holder in all_placeholders:
            if p_holder in lang_params: continue
            value, _ = find_and_calculate_value(
                p_holder, search_context, max_level, hero_id, rules, 
                is_modifier=is_modifier_effect
            )
            if value is not None: lang_params[p_holder] = value
        
        if 'MAX' in all_placeholders and 'FIXEDPOWER' in lang_params: lang_params['MAX'] = lang_params['FIXEDPOWER'] * 2
        if 'MIN' in all_placeholders and 'FIXEDPOWER' in lang_params: lang_params['MIN'] = math.floor(lang_params['FIXEDPOWER'] / 2)

        nested_effects = []
        if 'statusEffectCollections' in prop_data:
            for collection in prop_data['statusEffectCollections']:
                collection_name = collection.get('collectionNameOverride', '')
                parsed_collection = parsers['status_effects'](collection.get('statusEffects', []), special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers)
                for effect in parsed_collection: effect['collection_name'] = collection_name
                nested_effects.extend(parsed_collection)
        if 'statusEffects' in prop_data:
            nested_effects.extend(parsers['status_effects'](prop_data['statusEffects'], special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))

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

def parse_status_effects(status_effects_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> list:
    if not status_effects_list: return []
    parsed_items = []
    max_level = special_data.get("maxLevel", 1)
    se_lang_subset = parsers['se_lang_subset']
    
    for effect_instance in status_effects_list:
        effect_id = effect_instance.get("id")
        if not effect_id: continue

        effect_details = game_db['status_effects'].get(effect_id, {})
        combined_details = {**effect_details, **effect_instance}
        
        lang_id = rules.get("lang_overrides", {}).get("specific", {}).get(hero_id, {}).get(effect_id)
        if not lang_id: lang_id = rules.get("lang_overrides", {}).get("common", {}).get(effect_id)
        if not lang_id: lang_id = find_best_lang_id(combined_details, se_lang_subset, parent_block=special_data)

        lang_params, is_modifier_effect = {}, 'modifier' in combined_details.get('statusEffect', '').lower()
        if (turns := combined_details.get("turns", 0)) > 0: lang_params["TURNS"] = turns
        
        search_context = {**special_data, **combined_details}
        template_text_en = lang_db.get(lang_id, {}).get("en", "")
        placeholders = set(re.findall(r'\{(\w+)\}', template_text_en))
        
        for p_holder in placeholders:
            if p_holder in lang_params: continue
            value, found_key = find_and_calculate_value(
                p_holder, search_context, max_level, hero_id, rules,
                is_modifier=is_modifier_effect
            )
            
            if value is not None:
                if p_holder.upper() == "DAMAGE" and "permil" in (found_key or "").lower():
                    turns_for_calc = combined_details.get("turns", 0)
                    is_total = "over {TURNS} turns" in template_text_en
                    damage_per_turn = math.floor((value / 100) * hero_stats.get("max_attack", 0))
                    lang_params[p_holder] = damage_per_turn * (turns_for_calc or 1) if is_total else damage_per_turn
                else: 
                    lang_params[p_holder] = value
        
        nested_effects = []
        if 'statusEffectsToAdd' in combined_details:
             nested_effects.extend(parsers['status_effects'](combined_details['statusEffectsToAdd'], special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))

        formatted_params = {k: format_value(v) for k, v in lang_params.items()}
        descriptions = generate_description(lang_id, formatted_params, lang_db)
        descriptions['en'], descriptions['ja'] = re.sub(r'\n\s*\n', '\n', descriptions['en']).strip(), re.sub(r'\n\s*\n', '\n', descriptions['ja']).strip()
        
        parsed_items.append({
            "id": effect_id, "lang_id": lang_id, "params": json.dumps(lang_params),
            "nested_effects": nested_effects, **descriptions
        })
    return parsed_items

def parse_familiars(familiars_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> list:
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
            value, _ = find_and_calculate_value(
                p_holder, familiar_instance, max_level, hero_id, rules,
                is_modifier=False
            )
            if value is not None: lang_params[p_holder] = value
        
        formatted_params = {k: format_value(v) for k, v in lang_params.items()}
        main_desc = generate_description(lang_id, formatted_params, lang_db)
        main_desc['en'], main_desc['ja'] = main_desc['en'].replace('[*]', '\n・').strip(), main_desc['ja'].replace('[*]', '\n・').strip()
        
        nested_effects = []
        if 'effects' in familiar_instance: pass

        parsed_items.append({
            "id": familiar_id, "lang_id": lang_id, "description_en": main_desc['en'],
            "description_ja": main_desc['ja'], "params": json.dumps(lang_params),
            "nested_effects": nested_effects
        })
    return parsed_items