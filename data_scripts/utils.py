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
            base_heroes = build.pop('heroes') 
            base_heroes_map = {hero['name']: hero for hero in base_heroes}
            merged_teams = []
            if 'build_description' not in build:
                 build['build_description'] = ""
            
            for team_variation in trinket_build['teams']:
                merged_heroes = []
                for trinket_hero_info in team_variation.get('heroes', []):
                    hero_name = trinket_hero_info.get('name')
                    if not hero_name: continue
                    
                    base_hero = base_heroes_map.get(hero_name)

                    if base_hero:
                        merged_hero = {
                            "name": hero_name,
                            "position": base_hero.get('position', 0),
                            "skills": base_hero.get('skills', []),
                            "trinkets": trinket_hero_info.get('trinkets', [])
                        }
                        merged_heroes.append(merged_hero)
                
                team_variation['heroes'] = merged_heroes
                merged_teams.append(team_variation)

            build['teams'] = merged_teams
        
    return final_data