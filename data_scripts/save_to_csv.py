import json
import csv
import glob
import random
import yaml
import os
from typing import List, Dict, Any, Tuple
from utils import PATHS

RARITY_MAP = {
    0: {"pool": ["common", "uncommon"], "range": (10, 20)},
    1: {"pool": ["uncommon", "rare"], "range": (15, 30)},
    2: {"pool": ["uncommon", "rare", "very_rare"], "range": (20, 30)},
    3: {"pool": ["very_rare", "ancestral"], "range": (25, 40)},
    4: {"pool": ["very_rare", "ancestral", "trophy", "crystalline"], "range": (30, 50)},
}

def load_game_data() -> Tuple[List[str], Dict[str, List[str]]]:
    yaml_file_path = PATHS.yml
    
    if not os.path.exists(yaml_file_path):
        print(f"Game data file not found: {yaml_file_path}")
        return [], {}

    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            game_data = yaml.safe_load(file)
            
            all_heroes = [h['name'] for h in game_data.get('heroes', [])]
            
            all_trinkets_by_rarity = {
                rarity: [t['name'] for t in trinkets]
                for rarity, trinkets in game_data.get('trinkets', {}).items()
                if isinstance(trinkets, list)
            }
            
            return all_heroes, all_trinkets_by_rarity
            
    except Exception as e:
        print(f"YAML loading error!: {e}")
        return [], {}


def generate_variant(team: Dict[str, Any], variation_index: int, all_heroes: List[str], all_trinkets_by_rarity: Dict[str, List[str]]) -> Dict[str, str]:    
    team_tokens = []
    original_trinkets = []
    heroes = sorted(team["heroes"], key=lambda hero: hero["position"])

    for hero in heroes:
        hero_str = (
            f"{hero['name']}:" +
            "+".join(hero.get("skills", [])) +
            "+" +
            "+".join(hero.get("trinkets", []))
        )
        team_tokens.append(hero_str)
        original_trinkets.extend(hero.get("trinkets", []))
        
    target_sequence = "|".join(team_tokens)
    
    
    core_heroes = [h['name'] for h in team['heroes']]
    
    roster_size = random.randint(16, 28)
        
    heroes_needed = max(0, roster_size - len(core_heroes))
    other_heroes = random.choices(all_heroes, k=heroes_needed)

    input_roster_names = core_heroes + other_heroes
    
    rarity_info = RARITY_MAP.get(variation_index, RARITY_MAP[0])
    trinket_pool_rarities = rarity_info["pool"]
    total_trinket_range = rarity_info["range"]
    total_trinket_count = random.randint(*total_trinket_range)
    
    available_trinket_names = []
    for rarity in trinket_pool_rarities:
        available_trinket_names.extend(all_trinkets_by_rarity.get(rarity, []))
        
    num_to_sample = max(0, total_trinket_count - len(original_trinkets))
        
    additional_trinkets = random.choices(available_trinket_names, k=num_to_sample)
    
    input_trinkets = additional_trinkets + original_trinkets
    random.shuffle(input_roster_names)

    random.shuffle(input_trinkets)

    input_sequence = "|".join(input_roster_names) + "|"  + "|".join(input_trinkets)
    
    return {"input_sequence": input_sequence, "target_sequence": target_sequence}

def main():
    print("Loading data from darkest_dungeon_data.yml...")
    all_heroes, all_trinkets_by_rarity = load_game_data()
    
    if not all_heroes or not all_trinkets_by_rarity:
        return

    json_files = glob.glob(PATHS.jsons)
    all_rows = []
    VARIANTS_PER_TEAM = 10
    processed_teams_count = 0

    if not json_files:
        print("ERROR: json files not found")
        return

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                builds = json.load(f)
            
            for build in builds:
                base_heroes = build.get("heroes", [])
                base_heroes_map = {hero['name']: hero for hero in base_heroes}
                
                variations = build.get("trinket_variations", [])
                
                for i, variation in enumerate(variations):
                    processed_teams_count += 1
                    
                    reconstructed_team = {"heroes": []}
                    
                    for hero_trinket_info in variation.get("hero_trinkets", []):
                        hero_name = hero_trinket_info.get("name")
                        if not hero_name or hero_name not in base_heroes_map:
                            continue
                        
                        base_hero = base_heroes_map[hero_name]
                        
                        merged_hero = {
                            "name": hero_name,
                            "position": base_hero.get("position", 0),
                            "skills": base_hero.get("skills", []),
                            "trinkets": hero_trinket_info.get("trinkets", [])
                        }
                        reconstructed_team["heroes"].append(merged_hero)
                    
                    for _ in range(VARIANTS_PER_TEAM):
                        row = generate_variant(reconstructed_team, i, all_heroes, all_trinkets_by_rarity)
                        all_rows.append(row)

        except FileNotFoundError:
            print(f"ERROR: file not found - {file_path}")
        except json.JSONDecodeError:
            print(f"ERROR: incorrect json format - {file_path}")
        except Exception as e:
            print(f"ERROR while processing file: {file_path}: {e}")

    csv_file_path = PATHS.csv
    with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input_sequence", "target_sequence"])
        writer.writeheader()
        writer.writerows(all_rows)

    num_variants = len(all_rows)

    print(f"\n Data from {len(json_files)} files connected.")
    print(f"Processed {processed_teams_count} team variations.")
    print(f"Saved {num_variants} rows into {csv_file_path}.")

if __name__ == "__main__":
    main()