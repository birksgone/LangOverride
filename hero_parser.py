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
            # Ensure fixed values from CSV are treated as numbers if possible
            value_str = rule.get("value")
            try:
                return int(value_str)
            except (ValueError, TypeError):
                try:
                    return float(value_str)
                except (ValueError, TypeError):
                    return value_str, "Fixed Rule" # Fallback to string if not a number

        key_to_find = rule.get("key")
        if key_to_find:
            flat_data = flatten_json(data_block)
            matching_keys = [k for k in flat_data if k.endswith(key_to_find)]
            
            if len(matching_keys) == 1:
                found_key = matching_keys[0]
                value = flat_data[found_key]
                if isinstance(value, (int, float)):
                    # Exception rules currently do not support increment calculations, they return the direct value.
                    if 'permil' in found_key.lower():
                        return value / 10, f"Exception Rule: {found_key}"
                    return int(value), f"Exception Rule: {found_key}"
        return None, f"Exception rule key '{key_to_find}' not found or ambiguous"

    if not isinstance(data_block, dict): return None, None
    flat_data = flatten_json(data_block)
    
    p_holder_lower = p_holder.lower()
    ph_keywords = [s.lower() for s in re.findall('[A-Z][^A-Z]*', p_holder)]
    if not ph_keywords: ph_keywords = [p_holder.lower()]

    candidates = []
    for key, value in flat_data.items():
        if not isinstance(value, (int, float)):
            continue
        key_lower = key.lower()
        score = 0
        matched_keywords = sum(1 for kw in ph_keywords if kw in key_lower)
        if matched_keywords > 0:
            score += matched_keywords * 10
            if 'power' in key_lower or 'modifier' in key_lower:
                score += 5
            if 'permil' in key_lower:
                score += 3
            candidates.append({'key': key, 'score': score})
            
    if not candidates:
        return None, None
        
    best_candidate = sorted(candidates, key=lambda x: (-x['score'], len(x['key'])))[0]
    found_key = best_candidate['key']

    base_val = flat_data.get(found_key, 0)
    
    base_key_name = found_key.split('_')[-1]
    inc_key_name_pattern_1 = base_key_name.replace("permil", "incrementperlevelpermil").replace('power', 'incrementperlevel')
    inc_key_name_pattern_2 = base_key_name + "incrementperlevel"
    
    inc_key = None
    for k in flat_data:
        k_lower = k.lower()
        if k_lower.endswith(inc_key_name_pattern_1) or k_lower.endswith(inc_key_name_pattern_2):
            inc_key = k
            break
            
    inc_val = flat_data.get(inc_key, 0)
    
    # --- FIX: Ensure inc_val is a number before calculation ---
    if not isinstance(inc_val, (int, float)):
        inc_val = 0
    
    if is_modifier:
        calculated_val = ((base_val - 1000) + (inc_val * (max_level - 1))) / 10
        return calculated_val, found_key
    else:
        calculated_val = base_val + inc_val * (max_level - 1)
        if 'permil' in found_key.lower():
            return calculated_val / 10, found_key
        else:
            return int(calculated_val), found_key

def _collect_keywords_recursively(data_block, depth=0, max_depth=2) -> list:
    """
    Recursively collects string values from a data block as keywords,
    along with their nesting depth.
    """
    if depth > max_depth:
        return []

    keywords = []
    
    # Collect keywords from the current level
    if isinstance(data_block, dict):
        for key, value in data_block.items():
            if isinstance(value, str):
                # Add the value itself (e.g., "AddStatusEffects") and the key name (e.g., "propertyType")
                keywords.append((value.lower(), depth))
                # keywords.append((key.lower(), depth + 1)) # Keys are slightly less important
    
    # Recurse into nested lists
    if isinstance(data_block, dict):
        # Define keys that are known to contain lists of other skill blocks
        list_keys_to_scan = ['statusEffects', 'effects', 'statusEffectsToAdd', 'statusEffectCollections', 'properties']
        for key in list_keys_to_scan:
            if key in data_block and isinstance(data_block[key], list):
                for item in data_block[key]:
                    keywords.extend(_collect_keywords_recursively(item, depth + 1, max_depth))

    elif isinstance(data_block, list):
        for item in data_block:
            keywords.extend(_collect_keywords_recursively(item, depth + 1, max_depth))
            
    return keywords

def find_best_lang_id(data_block: dict, lang_key_subset: list, parent_block: dict = None) -> str:
    # --- Step 1: Direct Construction (Unchanged) ---
    if 'statusEffect' in data_block:
        buff_map = {
            "MinorDebuff": "minor", "MajorDebuff": "major",
            "MinorBuff": "minor", "MajorBuff": "major",
            "PermanentDebuff": "permanent", "PermanentBuff": "permanent"
        }
        # ... (The direct construction logic is the same as before, so it's condensed here)
        intensity = buff_map.get(data_block.get('buff'))
        status_effect_val = data_block.get('statusEffect')
        effect_name = status_effect_val.lower() if isinstance(status_effect_val, str) else None
        target_from_data = (parent_block or data_block).get('statusTargetType', '')
        target = target_from_data.lower() if isinstance(target_from_data, str) else ''
        side_from_data = (parent_block or data_block).get('sideAffected', '')
        side = side_from_data.lower() if isinstance(side_from_data, str) else ''
        if all([intensity, effect_name, target, side]):
            constructed_id = f"specials.v2.statuseffect.{intensity}.{effect_name}.{target}.{side}"
            if constructed_id in lang_key_subset: return constructed_id
            
    # --- Step 2: Weighted Scoring using Deep Keyword Collection ---
    
    # Collect all keywords from the block and its children, with depth information.
    # We also include the parent block to catch contextual keywords.
    contextual_block = {**data_block, "parent": parent_block}
    all_keywords_with_depth = _collect_keywords_recursively(contextual_block, depth=0)
    
    # Deduplicate while preserving the keyword with the shallowest depth
    seen_keywords = {}
    for kw, depth in all_keywords_with_depth:
        if kw not in seen_keywords or depth < seen_keywords[kw]:
            seen_keywords[kw] = depth
    
    potential_matches = []
    for lang_key in lang_key_subset:
        score = 0
        lang_key_parts = lang_key.lower().split('.')
        
        # Calculate score based on matched keywords and their depth
        for kw, depth in seen_keywords.items():
            if kw in lang_key_parts:
                # Keywords found at shallower depths get exponentially higher scores
                score += 100 / (2 ** depth)

        # Add a small bonus for certain structural keywords, if they exist
        if 'fixedpower' in lang_key_parts and 'hasfixedpower' in seen_keywords:
            score += 3
        if 'decrement' in lang_key_parts and any(isinstance(v, (int, float)) and v < 0 for v in data_block.values()):
            score += 2

        if score > 0:
            potential_matches.append({'key': lang_key, 'score': score})
    
    # --- Step 3: Final Decision ---
    if not potential_matches:
        primary_keyword = (data_block.get('propertyType') or data_block.get('statusEffect') or 'N/A')
        print(f"\n  - Warning: Could not find lang_id for skill '{data_block.get('id', 'UNKNOWN')}' (type: {primary_keyword})")
        return None

    potential_matches.sort(key=lambda x: (-x['score'], len(x['key'])))
    return potential_matches[0]['key']

def parse_direct_effect(special_data, hero_stats, lang_db, game_db, hero_id: str, rules: dict, parsers: dict):
    # This function can now be called for nested specials, so we check for the directEffect key
    effect_data = special_data.get("directEffect") if isinstance(special_data, dict) else None
    
    if not effect_data or not effect_data.get("effectType"):
        return {"id": "direct_effect_no_type", "lang_id": "N/A", "params": "{}", "en": "", "ja": ""}
    
    try:
        effect_type_str = effect_data.get('effectType', '')
        parts = ["specials.v2.directeffect", effect_type_str.lower()]
        if t := effect_data.get('typeOfTarget'): parts.append(t.lower())
        if s := effect_data.get('sideAffected'): parts.append(s.lower())
        lang_id = ".".join(parts)
        if effect_data.get("hasFixedPower"): lang_id += ".fixedpower"
    except AttributeError:
        return {"id": "direct_effect_error", "lang_id": "N/A", "params": "{}", "en": "Error parsing", "ja": "解析エラー"}

    params = {}
    # Use maxLevel from the sub-special if it exists, otherwise fallback to the main special's maxLevel
    max_level = special_data.get("maxLevel", parsers.get("main_max_level", 8))
    
    base = effect_data.get('powerMultiplierPerMil', 0)
    inc = effect_data.get('powerMultiplierIncrementPerLevelPerMil', 0)
    
    p_map = {"Damage":"HEALTH","Heal":"HEALTH","HealthBoost":"HEALTHBOOST","AddMana":"MANA"}
    placeholder = p_map.get(effect_type_str, "VALUE")
    
    if base > 0 or inc > 0:
        total_per_mil = base + inc * (max_level - 1)
        final_val = round(total_per_mil) if effect_data.get("hasFixedPower") else (round(total_per_mil/100) if effect_type_str=="AddMana" else round(total_per_mil/10))
        params[placeholder] = final_val

    desc = generate_description(lang_id, params, lang_db)
    return {"lang_id": lang_id, "params": json.dumps(params), **desc}

def parse_properties(properties_list: list, special_data: dict, hero_stats: dict, lang_db: dict, game_db: dict, hero_id: str, rules: dict, parsers: dict) -> list:
    if not properties_list: return []
    parsed_items = []
    main_max_level = special_data.get("maxLevel", 8)
    parsers["main_max_level"] = main_max_level
    prop_lang_subset = parsers['prop_lang_subset']
    
    for prop_id_or_dict in properties_list:
        prop_data, prop_id = {}, None
        if isinstance(prop_id_or_dict, dict):
            prop_data, prop_id = prop_id_or_dict, prop_id_or_dict.get('id')
        elif isinstance(prop_id_or_dict, str):
            prop_id, prop_data = prop_id_or_dict, game_db['special_properties'].get(prop_id_or_dict, {})

        if not prop_data or not prop_id: continue

        mana_speed_id = parsers.get("hero_mana_speed_id")
        property_type = prop_data.get("propertyType")
        
        container_types = {
            "changing_tides": "RotatingSpecial",
            "charge_ninja": "ChargedSpecial",
            "charge_magic": "ChargedSpecial"
        }

        if mana_speed_id in container_types and property_type == container_types[mana_speed_id]:
            container_lang_ids = {
                "changing_tides": "specials.v2.property.evolving_special",
                "charge_ninja": "specials.v2.property.chargedspecial.3",
                "charge_magic": "specials.v2.property.chargedspecial.2"
            }
            container_headings = {
                "changing_tides": {"en": ["1st:", "2nd:"], "ja": ["第1:", "第2:"]},
                "charge_ninja": {"en": ["x1 Mana Charge:", "x2 Mana Charge:", "x3 Mana Charge:"], "ja": ["x1マナチャージ:", "x2マナチャージ:", "x3マナチャージ:"]},
                "charge_magic": {"en": ["x1 Mana Charge:", "x2 Mana Charge:"], "ja": ["x1マナチャージ:", "x2マナチャージ:"]}
            }

            container_lang_id = container_lang_ids.get(mana_speed_id)
            container_desc = generate_description(container_lang_id, {}, lang_db)
            
            nested_effects = []
            sub_specials_list = prop_data.get("specialIds", [])
            headings = container_headings.get(mana_speed_id, {})

            for i, sub_special_id_or_dict in enumerate(sub_specials_list):
                sub_special_data = {}
                if isinstance(sub_special_id_or_dict, dict):
                    sub_special_data = sub_special_id_or_dict
                elif isinstance(sub_special_id_or_dict, str):
                    sub_special_data = game_db['character_specials'].get(sub_special_id_or_dict, {})
                
                if not sub_special_data: continue

                heading_en = headings.get("en", [])[i] if i < len(headings.get("en", [])) else f"Level {i+1}:"
                heading_ja = headings.get("ja", [])[i] if i < len(headings.get("ja", [])) else f"レベル {i+1}:"
                
                # --- FIX: Removed redundant "en" and "ja" keys to prevent duplication ---
                nested_effects.append({"id": "heading", "description_en": heading_en, "description_ja": heading_ja})
                
                if "directEffect" in sub_special_data:
                    nested_effects.append(parsers['direct_effect'](sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))
                if "properties" in sub_special_data:
                    nested_effects.extend(parsers['properties'](sub_special_data.get("properties", []), sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))
                if "statusEffects" in sub_special_data:
                    nested_effects.extend(parsers['status_effects'](sub_special_data.get("statusEffects", []), sub_special_data, hero_stats, lang_db, game_db, hero_id, rules, parsers))

            parsed_items.append({
                "id": prop_id, "lang_id": container_lang_id,
                "description_en": container_desc["en"], "description_ja": container_desc["ja"],
                "tooltip_en": "", "tooltip_ja": "",
                "params": "{}", "nested_effects": nested_effects
            })
            continue

        lang_id = rules.get("lang_overrides", {}).get("specific", {}).get(hero_id, {}).get(prop_id)
        if not lang_id: lang_id = rules.get("lang_overrides", {}).get("common", {}).get(prop_id)
        if not lang_id: lang_id = find_best_lang_id(prop_data, prop_lang_subset, parent_block=special_data)

        if not lang_id:
            parsed_items.append({
                "id": prop_id, "lang_id": "SEARCH_FAILED", 
                "description_en": f"Failed to find template for {prop_id}", "description_ja": f"テンプレート検索失敗: {prop_id}",
                "tooltip_en": "", "tooltip_ja": "",
                "params": "{}", "nested_effects": []
            })
            continue

        lang_params, is_modifier_effect = {}, 'modifier' in prop_data.get('propertyType', '').lower()
        main_template_text = lang_db.get(lang_id, {}).get("en", "")
        extra_lang_id = '.'.join(lang_id.split('.')[:4]) + ".extra"
        extra_template_text = lang_db.get(extra_lang_id, {}).get("en", "")
        all_placeholders = set(re.findall(r'\{(\w+)\}', main_template_text + extra_template_text))

        search_context = {**prop_data, "maxLevel": main_max_level}

        for p_holder in all_placeholders:
            if p_holder in lang_params: continue
            value, _ = find_and_calculate_value(
                p_holder, search_context, main_max_level, hero_id, rules, 
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

        if not lang_id:
            parsed_items.append({
                "id": effect_id, "lang_id": "SEARCH_FAILED",
                "en": f"Failed to find template for {effect_id}", "ja": f"テンプレート検索失敗: {effect_id}",
                "params": "{}", "nested_effects": []
            })
            continue

        lang_params, is_modifier_effect = {}, 'modifier' in combined_details.get('statusEffect', '').lower()
        if (turns := combined_details.get("turns", 0)) > 0: lang_params["TURNS"] = turns
        
        search_context = {**combined_details, "maxLevel": max_level}
        
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
    # ... (This function is unchanged)
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