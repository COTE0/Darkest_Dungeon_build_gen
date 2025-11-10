import copy
from typing import List, Dict, Any, Optional
import json
import google.generativeai as genai

class PATHS():
    folder = "data_scripts/"
    
    builds = folder + "builds/"
    yml = folder + "darkest_dungeon_data.yml"
    jsons = folder + "builds/*.json"
    csv = folder + "teams_dataset.csv"
    env = folder + ".env"


def response_to_json(response: genai.types.GenerateContentResponse) -> Optional[List[Dict[str, Any]]]:
    json_string = response.text.strip()
    if json_string.startswith("```json"):
        json_string = json_string[7:]
        if json_string.endswith("```"):
            json_string = json_string[:-3]
    
    try:
        json_string = json_string.strip()
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic JSON string start: {json_string[:200]}...")
        return None
    except Exception as e:
        print(f"Error processing response: {e}")
        return None

def merge_team_data(base_data: List[Dict[str, Any]], trinket_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not base_data or not trinket_data:
        return base_data or []

    final_data = copy.deepcopy(base_data)
    trinket_map = {build['build_id']: build for build in trinket_data}

    for build in final_data:
        build_id = build['build_id']
        trinket_build = trinket_map.get(build_id)
        
        if trinket_build and 'teams' in trinket_build:       
            merged_trinket_variations = []
            if 'build_description' not in build:
                 build['build_description'] = ""
            
            for team_variation in trinket_build['teams']:
                new_variation = {
                    "description": team_variation.get('description', ""),
                    "hero_trinkets": []
                }
                
                for trinket_hero_info in team_variation.get('heroes', []):
                    hero_name = trinket_hero_info.get('name')
                    if not hero_name: continue
                    
                    new_variation["hero_trinkets"].append({
                        "name": hero_name,
                        "trinkets": trinket_hero_info.get('trinkets', [])
                    })
                
                merged_trinket_variations.append(new_variation)

            build.pop('teams', None) 
            build['trinket_variations'] = merged_trinket_variations
        
    return final_data