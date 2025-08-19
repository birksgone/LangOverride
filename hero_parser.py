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

# --- NEW: Centralized Tooltip Parsing Helper (Robust Version) ---
def _find_and_parse_extra_description(
    categories: list, skill_name: str, search_context: dict, main_params: dict,
    lang_db: dict, hero_id: str, rules: dict, parsers: dict
) -> dict:
    """
    A generic helper to find and parse tooltip (.extra) descriptions.
    Searches for keys containing all required components, regardless of structure.
    """
    if not skill_name or not categories:
        return {}

    skill_name_lower = skill_name.lower()

    # Find all candidate keys that contain the skill name and '.extra'
    candidates = [
        key for key in parsers['extra_lang_ids']
        if skill_name_lower in key and '.extra' in key
    ]

    # From the candidates, find one that also contains one of the category names
    extra_lang_id = None
    for key in candidates:
        if any(cat in key for cat in categories):
            extra_lang_id = key
            break  # Found the first and best match

    if extra_lang_id and extra_lang_id in lang_db:
        extra_params = {}
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        extra_placeholders = set(re.findall(r'\{(\w+)\}', extra_template_text))

        # Inherit params from main description if placeholder exists in both
        for p in extra_placeholders:
            if p in main_params:
                extra_params[p] = main_params[p]

        # Find any remaining params needed only for the tooltip
        remaining_placeholders = extra_placeholders - set(extra_params.keys())
        for p_holder in remaining_placeholders:
            value, _ = find_and_calculate_value(
                p_holder, search_context, search_context.get("maxLevel", 8),
                hero_id, rules, is_modifier=False
            )
            if value is not None:
                extra_params[p_holder] = value

        formatted_extra_params = {k: format_value(v) for k, v in extra_params.items()}
        extra_desc = generate_description(extra_lang_id, formatted_extra_params, lang_db)

        return {
            "lang_id": extra_lang_id,
            "params": json.dumps(extra_params),
            "en": re.sub(r'\[\*\]|\n\s*\n', '\n・', extra_desc.get("en", "")).strip(),
            "ja": re.sub(r'\[\*\]|\n\s*\n', '\n・', extra_desc.get("ja", "")).strip()
        }
    return {}


# --- Core Data Integration Logic (Unchanged) ---
def get_full_hero_data(base_data: dict, game_db: dict) -> dict:
    resolved_data = json.loads(json.dumps(base_data))
    processed_ids = set()
    _resolve_recursive(resolved_data, game_db['master_db'], processed_ids)
    return resolved_data

def _resolve_recursive(current_data, master_db, processed_ids):
    if id(current_data) in processed_ids: return
    processed_ids.add(id(current_data))
    ID_KEYS_FOR_LISTS = ['properties','statusEffects','statusEffectsPerHit','summonedFamiliars','effects','passiveSkills','costumeBonusPassiveSkillIds','statusEffectsToAdd','statusEffectCollections']
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
            item_id_to_resolve = item if isinstance(item, str) else (item.get('id') if isinstance(item, dict) else None)
            if item_id_to_resolve and item_id_to_resolve in master_db and item_id_to_resolve not in processed_ids:
                processed_ids.add(item_id_to_resolve)
                new_data = json.loads(json.dumps(master_db[item_id_to_resolve]))
                _resolve_recursive(new_data, master_db, processed_ids)
                if isinstance(current_data[i], str): current_data[i] = new_data
                else: current_data[i].update(new_data)
            elif isinstance(item, (dict, list)):
                 _resolve_recursive(item, master_db, processed_ids)

# --- Analysis & Parsing Functions (Unchanged) ---
def get_hero_final_stats(hero_id: str, hero_stats_db: dict) -> dict:
    hero_data = hero_stats_db.get(hero_id)
    if not hero_data: return {"max_attack": 0, "name": "N/A"}
    attack_col = 'Max level: Attack'
    for i in range(4, 0, -1):
        col_name = f'Max level CB{i}: Attack'
        if col_name in hero_data and pd.notna(hero_data[col_name]):
            attack_col = col_name; break
    return {"max_attack": int(hero_data.get(attack_col, 0)), "name": hero_data.get('Name', 'N/A')}

def find_and_calculate_value(p_holder: str, data_block: dict, max_level: int, hero_id: str, rules: dict, is_modifier: bool = False, ignore_keywords: list = None) -> (any, str):
    p_holder_upper = p_holder.upper()
    rule = rules.get("hero_rules", {}).get("specific", {}).get(hero_id, {}).get(p_holder_upper)
    if not rule: rule = rules.get("hero_rules", {}).get("common", {}).get(p_holder_upper)
    if rule:
        if rule.get("calc") == "fixed":
            value_str = rule.get("value")
            try: return int(value_str), "Fixed Rule"
            except (ValueError, TypeError):
                try: return float(value_str), "Fixed Rule"
                except (ValueError, TypeError): return value_str, "Fixed Rule"
        if key_to_find := rule.get("key"):
            flat_data = flatten_json(data_block)
            matching_keys = [k for k in flat_data if k.endswith(key_to_find)]
            if len(matching_keys) == 1:
                found_key = matching_keys[0]; value = flat_data[found_key]
                if isinstance(value, (int, float)):
                    if 'permil' in found_key.lower(): return value / 10, f"Exception Rule: {found_key}"
                    return int(value), f"Exception Rule: {found_key}"
        return None, f"Exception rule key '{key_to_find}' not found or ambiguous"
    if not isinstance(data_block, dict): return None, None
    flat_data = flatten_json(data_block)
    if ignore_keywords:
        keys_to_remove = {k for k in flat_data if any(ik in k.lower() for ik in ignore_keywords)}
        for k in keys_to_remove: del flat_data[k]
    ph_keywords = [s.lower() for s in re.findall('[A-Z][^A-Z]*', p_holder)] or [p_holder.lower()]
    candidates = []
    for key, value in flat_data.items():
        if not isinstance(value, (int, float)): continue
        key_lower = key.lower()
        matched_keywords = sum(1 for kw in ph_keywords if kw in key_lower)
        if matched_keywords > 0:
            score = matched_keywords * 10
            if 'power' in key_lower or 'modifier' in key_lower: score += 5
            if 'permil' in key_lower: score += 3
            candidates.append({'key': key, 'score': score})
    if not candidates: return None, None
    best_candidate = sorted(candidates, key=lambda x: (-x['score'], len(x['key'])))[0]
    found_key = best_candidate['key']; base_val = flat_data.get(found_key, 0)
    inc_key = None
    if found_key.endswith("PerMil"):
        potential_inc_key = found_key.replace("PerMil", "PerLevelPerMil")
        if potential_inc_key in flat_data: inc_key = potential_inc_key
    if not inc_key:
        potential_inc_key = found_key.replace("PerMil", "IncrementPerLevelPerMil")
        if potential_inc_key in flat_data: inc_key = potential_inc_key
    if not inc_key:
        if found_key.islower(): potential_inc_key = found_key + "incrementperlevel"
        else: potential_inc_key = re.sub(r'([a-z])([A-Z])', r'\1IncrementPerLevel\2', found_key)
        if potential_inc_key in flat_data: inc_key = potential_inc_key
    inc_val = flat_data.get(inc_key, 0)
    if not isinstance(inc_val, (int, float)): inc_val = 0
    calculated_val = base_val + inc_val * (max_level - 1)
    if is_modifier or 'modifier' in found_key.lower():
        return ((base_val - 1000) + (inc_val * (max_level - 1))) / 10, found_key
    if 'permil' in found_key.lower(): return calculated_val / 10, found_key
    return int(calculated_val), found_key

def _collect_keywords_recursively(data_block, depth=0, max_depth=2) -> list:
    if depth > max_depth: return []
    keywords = []
    if isinstance(data_block, dict):
        for key, value in data_block.items():
            if isinstance(value, str): keywords.append((value.lower(), depth))
        list_keys_to_scan = ['statusEffects','effects','statusEffectsToAdd','statusEffectCollections','properties']
        for key in list_keys_to_scan:
            if key in data_block and isinstance(data_block[key], list):
                for item in data_block[key]: keywords.extend(_collect_keywords_recursively(item, depth + 1, max_depth))
    elif isinstance(data_block, list):
        for item in data_block: keywords.extend(_collect_keywords_recursively(item, depth + 1, max_depth))
    return keywords

def find_best_lang_id(data_block: dict, lang_key_subset: list, parsers: dict, parent_block: dict = None) -> (str, str):
    if 'statusEffect' in data_block:
        buff_map = {"MinorDebuff":"minor","MajorDebuff":"major","MinorBuff":"minor","MajorBuff":"major","PermanentDebuff":"permanent","PermanentBuff":"permanent"}
        intensity = buff_map.get(data_block.get('buff'))
        status_effect_val = data_block.get('statusEffect'); effect_name = status_effect_val.lower() if isinstance(status_effect_val, str) else None
        target_from_data = (parent_block or data_block).get('statusTargetType', ''); target = target_from_data.lower() if isinstance(target_from_data, str) else ''
        side_from_data = (parent_block or data_block).get('sideAffected', ''); side = side_from_data.lower() if isinstance(side_from_data, str) else ''
        if all([intensity, effect_name, target, side]):
            constructed_id = f"specials.v2.statuseffect.{intensity}.{effect_name}.{target}.{side}"
            if constructed_id in lang_key_subset: return constructed_id, None
    contextual_block = {**data_block, "parent": parent_block}
    all_keywords_with_depth = _collect_keywords_recursively(contextual_block, depth=0)
    seen_keywords = {}
    for kw, depth in all_keywords_with_depth:
        if kw not in seen_keywords or depth < seen_keywords[kw]: seen_keywords[kw] = depth
    potential_matches = []
    for lang_key in lang_key_subset:
        score = 0; lang_key_parts = lang_key.lower().split('.')
        for kw, depth in seen_keywords.items():
            if kw in lang_key_parts: score += 100 / (2 ** depth)
        familiar_type = data_block.get("familiarType", "").lower()
        if familiar_type:
            if ("minion" in familiar_type and "allies" in lang_key_parts): score += 20
            if ("parasite" in familiar_type and "enemies" in lang_key_parts): score += 20
        if 'fixedpower' in lang_key_parts and 'hasfixedpower' in seen_keywords: score += 3
        if 'decrement' in lang_key_parts and any(isinstance(v, (int, float)) and v < 0 for v in data_block.values()): score += 2
        if score > 0: potential_matches.append({'key': lang_key, 'score': score, 'parts': lang_key_parts})
    if not potential_matches:
        primary_keyword = (data_block.get('propertyType') or data_block.get('statusEffect') or data_block.get('familiarType') or 'N/A')
        return None, f"Could not find lang_id for skill '{data_block.get('id', 'UNKNOWN')}' (type: {primary_keyword})"
    potential_matches.sort(key=lambda x: (-x['score'], len(x['key'])))
    if "familiar_debug_log" in parsers and data_block.get('familiarType'):
        log_entry = {"familiar_id":data_block.get('id'),"familiar_instance":data_block,"top_candidates":[{'score':f"{m['score']:.2f}",'key':m['key']} for m in potential_matches[:5]]}
        parsers["familiar_debug_log"].append(log_entry)
    return potential_matches[0]['key'], None

def parse_clear_buffs(special_data: dict, lang_db: dict, parsers: dict) -> (dict, list):
    if "buffToRemove" not in special_data: return None, []
    warnings = []
    try:
        buff_to_remove = special_data.get("buffToRemove", "").lower(); target_type = special_data.get("buffToRemoveTargetType", "all").lower()
        side_affected = ""
        if "debuff" in buff_to_remove: side_affected = "allies"
        elif "buff" in buff_to_remove: side_affected = "enemies"
        if not side_affected: side_affected = special_data.get("buffToRemoveSideAffected", "").lower()
        if not side_affected: side_affected = special_data.get("sideAffected", "").lower()
        if not side_affected: side_affected = special_data.get("directEffect", {}).get("sideAffected", "").lower()
        if not side_affected: side_affected = "allies" if "debuff" in buff_to_remove else "enemies"
        lang_id = f"specials.v2.clearbuffs.{buff_to_remove}.{target_type}.{side_affected}"
        if lang_id not in lang_db and lang_id + ".latest" in lang_db: lang_id += ".latest"
        description = generate_description(lang_id, {}, lang_db)
        result = {"id":"clear_buffs_effect","lang_id":lang_id,"params":"{}","nested_effects":[],**description}
        return result, warnings
    except Exception as e:
        warnings.append(f"Error parsing clear_buffs for '{special_data.get('id', 'Unknown Special')}': {e}")
        return None, warnings

def parse_direct_effect(special_data, hero_stats, lang_db, game_db, hero_id: str, rules: dict, parsers: dict):
    effect_data = special_data.get("directEffect") if isinstance(special_data, dict) else None
    if not effect_data or not effect_data.get("effectType"): return {"id": "direct_effect_no_type", "lang_id": "N/A", "params": "{}", "en": "", "ja": ""}
    try:
        effect_type_str = effect_data.get('effectType', '')
        parts = ["specials.v2.directeffect", effect_type_str.lower()]
        if t := effect_data.get('typeOfTarget'): parts.append(t.lower())
        if s := effect_data.get('sideAffected'): parts.append(s.lower())
        lang_id = ".".join(parts)
        if effect_type_str == "AddMana":
            power_value = effect_data.get('powerMultiplierPerMil', 0)
            if power_value > 0: lang_id += ".increment"
            elif power_value < 0: lang_id += ".decrement"
        if effect_data.get("hasFixedPower"): lang_id += ".fixedpower"
    except AttributeError: return {"id": "direct_effect_error", "lang_id": "N/A", "params": "{}", "en": "Error parsing", "ja": "解析エラー"}
    params = {}
    max_level = special_data.get("maxLevel", parsers.get("main_max_level", 8))
    base = effect_data.get('powerMultiplierPerMil', 0); inc = effect_data.get('powerMultiplierIncrementPerLevelPerMil', 0)
    p_map = {"Damage":"HEALTH","Heal":"HEALTH","HealthBoost":"HEALTHBOOST","AddMana":"MANA"}
    placeholder = p_map.get(effect_type_str, "VALUE")
    total_per_mil = base + inc * (max_level - 1)
    if base > 0 or inc > 0:
        final_val = round(total_per_mil) if effect_data.get("hasFixedPower") else (round(total_per_mil/100) if effect_type_str=="AddMana" else round(total_per_mil/10))
        params[placeholder] = final_val
    elif base < 0 or inc < 0:
        params[placeholder] = abs(round(total_per_mil / 100))
    desc = generate_description(lang_id, params, lang_db)
    return {"lang_id": lang_id, "params": json.dumps(params), **desc}

def parse_properties(properties_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> (list, list):
    if not properties_list: return [], []
    parsed_items = []; warnings = []
    main_max_level = special_data.get("maxLevel", 8)
    parsers["main_max_level"] = main_max_level
    prop_lang_subset = parsers['prop_lang_subset']
    for prop_id_or_dict in properties_list:
        prop_data, prop_id = {}, None
        if isinstance(prop_id_or_dict, dict): prop_data, prop_id = prop_id_or_dict, prop_id_or_dict.get('id')
        elif isinstance(prop_id_or_dict, str): prop_id, prop_data = prop_id_or_dict, game_db['special_properties'].get(prop_id_or_dict, {})
        if not prop_data or not prop_id: continue
        property_type = prop_data.get("propertyType", "")
        container_types = {"changing_tides":"RotatingSpecial","charge_ninja":"ChargedSpecial","charge_magic":"ChargedSpecial"}
        if parsers.get("hero_mana_speed_id") in container_types and property_type == container_types[parsers.get("hero_mana_speed_id")]:
            container_lang_ids = {"changing_tides":"specials.v2.property.evolving_special","charge_ninja":"specials.v2.property.chargedspecial.3","charge_magic":"specials.v2.property.chargedspecial.2"}
            container_headings = {"changing_tides":{"en":["1st:","2nd:"],"ja":["第1:","第2:"]},"charge_ninja":{"en":["x1 Mana Charge:","x2 Mana Charge:","x3 Mana Charge:"],"ja":["x1マナチャージ:","x2マナチャージ:","x3マナチャージ:"]},"charge_magic":{"en":["x1 Mana Charge:","x2 Mana Charge:"],"ja":["x1マナチャージ:","x2マナチャージ:"]}}
            container_lang_id = container_lang_ids.get(parsers.get("hero_mana_speed_id"))
            container_desc = generate_description(container_lang_id, {}, lang_db)
            nested_effects = []
            sub_specials_list = prop_data.get("specialIds", [])
            headings = container_headings.get(parsers.get("hero_mana_speed_id"), {})
            for i, sub_special_data in enumerate(sub_specials_list):
                if not isinstance(sub_special_data, dict) or not sub_special_data: continue
                heading_en = headings.get("en", [])[i] if i < len(headings.get("en", [])) else f"Level {i+1}:"
                heading_ja = headings.get("ja", [])[i] if i < len(headings.get("ja", [])) else f"レベル {i+1}:"
                nested_effects.append({"id":"heading","description_en":heading_en,"description_ja":heading_ja})
                if "directEffect" in sub_special_data:
                    nested_effects.append(parsers['direct_effect'](sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))
                if "properties" in sub_special_data:
                    parsed_props, new_warnings = parsers['properties'](sub_special_data.get("properties",[]), sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers)
                    nested_effects.extend(parsed_props); warnings.extend(new_warnings)
                if "statusEffects" in sub_special_data:
                    parsed_ses, new_warnings = parsers['status_effects'](sub_special_data.get("statusEffects",[]), sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers)
                    nested_effects.extend(parsed_ses); warnings.extend(new_warnings)
            parsed_items.append({"id":prop_id,"lang_id":container_lang_id,"description_en":container_desc["en"],"description_ja":container_desc["ja"],"params":"{}","nested_effects":nested_effects})
            continue
        lang_id = rules.get("lang_overrides",{}).get("specific",{}).get(hero_id,{}).get(prop_id) or rules.get("lang_overrides",{}).get("common",{}).get(prop_id)
        if not lang_id:
            lang_id, warning = find_best_lang_id(prop_data, prop_lang_subset, parsers, parent_block=special_data)
            if warning: warnings.append(warning)
        if not lang_id:
            parsed_items.append({"id":prop_id,"lang_id":"SEARCH_FAILED","en":f"Failed for {prop_id}","params":"{}"}); continue
        lang_params = {}; search_context = {**prop_data, "maxLevel": main_max_level}
        placeholders = set(re.findall(r'\{(\w+)\}', lang_db.get(lang_id,{}).get("en","")))
        for p_holder in placeholders:
            value, _ = find_and_calculate_value(p_holder, search_context, main_max_level, hero_id, rules, is_modifier='modifier' in property_type.lower())
            if value is not None: lang_params[p_holder] = value
        main_desc = generate_description(lang_id, {k:format_value(v) for k,v in lang_params.items()}, lang_db)
        nested_effects = []
        if 'statusEffects' in prop_data:
            parsed_ses, new_warnings = parsers['status_effects'](prop_data['statusEffects'], special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers)
            nested_effects.extend(parsed_ses); warnings.extend(new_warnings)
        
        extra_info = _find_and_parse_extra_description(["specialproperty", "property"], property_type, search_context, lang_params, lang_db, hero_id, rules, parsers)
        
        result_item = {"id":prop_id,"lang_id":lang_id,"params":json.dumps(lang_params),"nested_effects":nested_effects,**main_desc}
        if extra_info: result_item["extra"] = extra_info
        parsed_items.append(result_item)
    return parsed_items, warnings

def parse_status_effects(status_effects_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> (list, list):
    if not status_effects_list: return [], []
    parsed_items = []; warnings = []
    main_max_level = special_data.get("maxLevel", 8)
    se_lang_subset = parsers['se_lang_subset']
    for effect_instance in status_effects_list:
        if not isinstance(effect_instance, dict): continue
        effect_id = effect_instance.get("id"); combined_details = effect_instance
        if not effect_id: continue
        lang_id = rules.get("lang_overrides",{}).get("specific",{}).get(hero_id,{}).get(effect_id) or rules.get("lang_overrides",{}).get("common",{}).get(effect_id)
        if not lang_id:
            lang_id, warning = find_best_lang_id(combined_details, se_lang_subset, parsers, parent_block=special_data)
            if warning: warnings.append(warning)
        if not lang_id:
            parsed_items.append({"id":effect_id,"lang_id":"SEARCH_FAILED","en":f"Failed for {effect_id}","params":"{}"}); continue
        lang_params = {}; search_context = {**combined_details, "maxLevel": main_max_level}
        if (turns := combined_details.get("turns", 0)) > 0: lang_params["TURNS"] = turns
        template_text = lang_db.get(lang_id,{}).get("en","")
        placeholders = set(re.findall(r'\{(\w+)\}', template_text))
        for p_holder in placeholders:
            if p_holder in lang_params: continue
            value, found_key = find_and_calculate_value(p_holder, search_context, main_max_level, hero_id, rules, is_modifier='modifier' in combined_details.get('statusEffect','').lower())
            if value is not None:
                if p_holder.upper() == "DAMAGE" and "permil" in (found_key or "").lower():
                    turns_for_calc = combined_details.get("turns",0)
                    damage_per_turn = math.floor((value/100) * hero_stats.get("max_attack",0))
                    lang_params[p_holder] = damage_per_turn * (turns_for_calc or 1) if "over {TURNS} turns" in template_text else damage_per_turn
                else: lang_params[p_holder] = value
        main_desc = generate_description(lang_id, {k:format_value(v) for k,v in lang_params.items()}, lang_db)
        nested_effects = []
        if 'statusEffectsToAdd' in combined_details:
             parsed_nested_ses, new_warnings = parsers['status_effects'](combined_details['statusEffectsToAdd'], special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers)
             nested_effects.extend(parsed_nested_ses); warnings.extend(new_warnings)
        status_effect_type = combined_details.get("statusEffect","")
        
        extra_info = _find_and_parse_extra_description(["statuseffect"], status_effect_type, search_context, lang_params, lang_db, hero_id, rules, parsers)
        
        result_item = {"id":effect_id,"lang_id":lang_id,"params":json.dumps(lang_params),"nested_effects":nested_effects,**main_desc}
        if extra_info: result_item["extra"] = extra_info
        parsed_items.append(result_item)
    return parsed_items, warnings

def parse_familiars(familiars_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> (list, list):
    if not familiars_list: return [], []
    parsed_items = []; warnings = []
    main_max_level = special_data.get("maxLevel", 8)
    all_familiar_lang_ids = [k for k in lang_db if k.startswith("specials.v2.")]
    for familiar_instance in familiars_list:
        familiar_id = familiar_instance.get("id")
        if not familiar_id: continue
        primary_candidates = [k for k in all_familiar_lang_ids if familiar_id in k]
        lang_id, warning = (find_best_lang_id(familiar_instance, primary_candidates, parsers) if primary_candidates else find_best_lang_id(familiar_instance, all_familiar_lang_ids, parsers))
        if warning: warnings.append(warning)
        if not lang_id:
            parsed_items.append({"id":familiar_id,"lang_id":"SEARCH_FAILED","description_en":f"Failed for familiar {familiar_id}","nested_effects":[]}); continue
        lang_params = {}; search_context = {**familiar_instance, "maxLevel": main_max_level}
        placeholders = set(re.findall(r'\{(\w+)\}', lang_db.get(lang_id,{}).get("en","")))
        health_val = familiar_instance.get('healthPerMil',0); inc_val_health = familiar_instance.get('healthPerLevelPerMil',0)
        lang_params['FAMILIARHEALTHPERCENT'] = (health_val + inc_val_health * (main_max_level - 1)) / 10.0
        log_entry = {'hero_id':hero_id,'familiar_id':familiar_id,'raw_healthPerMil':health_val,'calculated_health':lang_params['FAMILIARHEALTHPERCENT']}
        attack_found = False
        if effects := familiar_instance.get('effects'):
            for effect in effects:
                if isinstance(effect,dict) and 'attackPercentPerMil' in effect:
                    attack_val = effect.get('attackPercentPerMil',0)
                    inc_val_attack = effect.get('attackPercentIncrementPerLevelPerMil',0) if "parasite" in familiar_instance.get("familiarType","").lower() else 0
                    lang_params['FAMILIARATTACK'] = (attack_val + inc_val_attack * (main_max_level - 1)) / 10.0
                    log_entry.update({'raw_attackPercentPerMil':attack_val,'raw_attackIncrement':inc_val_attack,'calculated_attack':lang_params['FAMILIARATTACK']})
                    attack_found = True; break
        if not attack_found: log_entry['raw_attackPercentPerMil'] = 'NOT_FOUND'
        parsers["familiar_parameter_log"].append(log_entry)
        for p_holder in placeholders - set(lang_params.keys()):
            value, _ = find_and_calculate_value(p_holder, familiar_instance, main_max_level, hero_id, rules, is_modifier=False, ignore_keywords=['monster'])
            if value is not None: lang_params[p_holder] = value
        main_desc = generate_description(lang_id, {k:format_value(v) for k,v in lang_params.items()}, lang_db)
        main_desc['en'], main_desc['ja'] = main_desc['en'].replace('[*]','\n・').strip(), main_desc['ja'].replace('[*]','\n・').strip()
        nested_effects = []
        if 'effects' in familiar_instance:
            nested_effects, new_warnings = _parse_familiar_effects(familiar_instance, lang_db, hero_stats, game_db, hero_id, rules, parsers)
            warnings.extend(new_warnings)
        familiar_type = familiar_instance.get("familiarType","")
        
        extra_info = _find_and_parse_extra_description(["familiartype"], familiar_type, search_context, lang_params, lang_db, hero_id, rules, parsers)
        
        result_item = {"id":familiar_id,"lang_id":lang_id,"params":json.dumps(lang_params),"nested_effects":nested_effects,"description_en":main_desc['en'],"description_ja":main_desc['ja']}
        if extra_info: result_item["extra"] = extra_info
        parsed_items.append(result_item)
    return parsed_items, warnings

def _parse_familiar_effects(familiar_instance: dict, lang_db: dict, hero_stats: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> (list, list):
    effects_list = familiar_instance.get("effects", [])
    if not effects_list: return [], []
    parsed_effects = []; warnings = []
    main_max_level = parsers.get("main_max_level", 8)
    familiar_id = familiar_instance.get("id", "")
    all_effect_lang_ids = [k for k in lang_db if familiar_id in k and (k.startswith("specials.v2.") or k.startswith("familiar.statuseffect."))]
    for effect_data in effects_list:
        effect_id = effect_data.get("id"); context_block = {**familiar_instance, **effect_data}
        if not effect_id: continue
        effect_type_keyword = effect_data.get('effectType',"")
        primary_candidates = [k for k in all_effect_lang_ids if (effect_type_keyword and effect_type_keyword in k) or (effect_id and effect_id in k)]
        lang_id, warning = (find_best_lang_id(context_block, primary_candidates, parsers) if primary_candidates else find_best_lang_id(context_block, all_effect_lang_ids, parsers))
        if warning: warnings.append(warning)
        if not lang_id:
            parsed_effects.append({"id":effect_id,"lang_id":"SEARCH_FAILED","en":f"Failed for familiar effect {effect_id}","params":"{}"}); continue
        lang_params = {}; search_context = {**effect_data, "maxLevel": main_max_level}
        placeholders = set(re.findall(r'\{(\w+)\}', lang_db.get(lang_id,{}).get("en","")))
        for p_holder in placeholders:
            value, _ = find_and_calculate_value(p_holder, context_block, main_max_level, hero_id, rules, is_modifier=False)
            if value is not None: lang_params[p_holder] = value
        if 'FAMILIAREFFECTFREQUENCY' in placeholders and 'turnsBetweenNonDamageEffects' in familiar_instance:
             lang_params['FAMILIAREFFECTFREQUENCY'] = familiar_instance['turnsBetweenNonDamageEffects'] + 1
        main_desc = generate_description(lang_id, {k:format_value(v) for k,v in lang_params.items()}, lang_db)

        extra_info = _find_and_parse_extra_description(["familiareffect"], effect_type_keyword, search_context, lang_params, lang_db, hero_id, rules, parsers)
        
        result_item = {"id":effect_id,"lang_id":lang_id,"params":json.dumps(lang_params),**main_desc}
        if extra_info: result_item["extra"] = extra_info
        parsed_effects.append(result_item)
    return parsed_effects, warnings

def parse_passive_skills(passive_skills_list: list, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> (list, list):
    # (This function is unchanged as passives do not have tooltips)
    if not passive_skills_list: return [], []
    parsed_items = []; warnings = []
    main_max_level = parsers.get("main_max_level", 8)
    title_lang_subset = [k for k in lang_db if k.startswith("herocard.passive_skill.title.")]
    desc_lang_subset = [k for k in lang_db if k.startswith("herocard.passive_skill.description.")]
    for skill_data in passive_skills_list:
        if not isinstance(skill_data, dict): continue
        skill_id = skill_data.get("id"); skill_type = skill_data.get("passiveSkillType","").lower()
        if not (skill_id and skill_type): continue
        title_lang_id = None
        prefix = f"herocard.passive_skill.title.{skill_type}"
        title_candidates = [k for k in title_lang_subset if k.startswith(prefix)]
        if title_candidates:
            skill_keywords = {kw for kw, depth in _collect_keywords_recursively(skill_data)}
            title_scores = [{'key':c,'score':sum(1 for kw in skill_keywords if kw in c.split('.'))} for c in title_candidates]
            if title_scores: title_lang_id = sorted(title_scores, key=lambda x:(-x['score'],len(x['key'])))[0]['key']
        desc_lang_id = None
        if title_lang_id:
            ideal_desc_id = title_lang_id.replace('.title.','.description.',1)
            if ideal_desc_id in lang_db: desc_lang_id = ideal_desc_id
            else:
                prefix = f"herocard.passive_skill.description.{skill_type}"
                desc_candidates = [k for k in desc_lang_subset if k.startswith(prefix)]
                if desc_candidates:
                    skill_keywords = {kw for kw, depth in _collect_keywords_recursively(skill_data)}
                    refined_candidates = [c for c in desc_candidates if any(kw in c.split('.') for kw in skill_keywords)]
                    if refined_candidates: desc_lang_id = min(refined_candidates, key=len)
                    elif desc_candidates: desc_lang_id = min(desc_candidates, key=len)
        if title_lang_id and desc_lang_id:
            all_placeholders = set(re.findall(r'\{(\w+)\}', lang_db.get(title_lang_id,{}).get("en","") + lang_db.get(desc_lang_id,{}).get("en","")))
            lang_params = {}
            search_context = {**skill_data, "maxLevel": main_max_level}
            for p_holder in all_placeholders:
                value, found_key = find_and_calculate_value(p_holder, search_context, main_max_level, hero_id, rules, is_modifier=False)
                if value is not None:
                    if p_holder.upper() == "DAMAGE" and "permil" in (found_key or "").lower():
                         lang_params[p_holder] = math.floor((value/100) * hero_stats.get("max_attack",0))
                    else: lang_params[p_holder] = value
            formatted_params = {k:format_value(v) for k,v in lang_params.items()}
            title_texts = generate_description(title_lang_id, formatted_params, lang_db)
            desc_texts = generate_description(desc_lang_id, formatted_params, lang_db)
            parsed_items.append({"id":skill_id,"title_en":title_texts.get("en",""),"title_ja":title_texts.get("ja",""),"description_en":desc_texts.get("en",""),"description_ja":desc_texts.get("ja",""),"params":json.dumps(lang_params)})
        else:
            warnings.append(f"Could not resolve passive lang_ids for skill '{skill_id}'")
            parsed_items.append({"id":skill_id,"title_en":f"FAILED: {skill_id}","description_en":"lang_id resolution failed."})
    return parsed_items, warnings