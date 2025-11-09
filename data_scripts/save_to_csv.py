import json
import csv
import glob
import random
import yaml
import os
from random import choices
from typing import List, Dict, Any, Tuple
from utils import PATHS

RARITY_MAP = {
    0: {"pool": ["common", "uncommon"], "range": (10, 20)},
    1: {"pool": ["uncommon", "rare"], "range": (15, 25)},
    2: {"pool": ["uncommon", "rare", "very_rare"], "range": (20, 30)},
    3: {"pool": ["very_rare", "ancestral"], "range": (25, 35)},
    4: {"pool": ["very_rare", "ancestral", "trophy", "crystalline"], "range": (30, 40)},
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
    original_trinkets = set()
    heroes = sorted(team["heroes"], key=lambda hero: hero["position"])

    for hero in heroes:
        hero_str = (
            f"{hero['name']}:" +
            "+".join(hero.get("skills", [])) +
            "+" +
            "+".join(hero.get("trinkets", []))
        )
        team_tokens.append(hero_str)
        original_trinkets.update(hero.get("trinkets", []))
        
    target_sequence = "|".join(team_tokens)
    
    
    original_heroes = [h['name'] for h in team['heroes']]
    
    if len(original_heroes) >= 4:
        core_heroes = random.sample(original_heroes, 4)
    else:
        core_heroes = original_heroes 

    roster_size = random.randint(16, 28)
    
    available_for_random = [h for h in all_heroes if h not in core_heroes]
    
    heroes_needed = roster_size - len(core_heroes)
    if heroes_needed > 0:
        if len(available_for_random) >= heroes_needed:
            other_heroes = choices(available_for_random, k=heroes_needed)
        else:
            other_heroes = available_for_random
    else:
        other_heroes = []


    input_roster_names = list(set(core_heroes + other_heroes))
    random.shuffle(input_roster_names)
    
    rarity_info = RARITY_MAP.get(variation_index, RARITY_MAP[0])
    trinket_pool_rarities = rarity_info["pool"]
    total_trinket_range = rarity_info["range"]
    total_trinket_count = random.randint(*total_trinket_range)
    
    available_trinket_names = []
    for rarity in trinket_pool_rarities:
        available_trinket_names.extend(all_trinkets_by_rarity.get(rarity, []))
    
    available_trinket_names = list(set(available_trinket_names))
    
    num_to_sample = max(0, total_trinket_count - len(original_trinkets))
    
    trinkets_to_sample_from = [t for t in available_trinket_names if t not in original_trinkets]
    
    num_to_sample = min(num_to_sample, len(trinkets_to_sample_from))

    additional_trinkets = random.sample(trinkets_to_sample_from, num_to_sample)
    
    input_trinkets = list(original_trinkets | set(additional_trinkets))

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
                for i, team in enumerate(build["teams"]):
                    processed_teams_count += 1
                    
                    for _ in range(VARIANTS_PER_TEAM):
                        row = generate_variant(team, i, all_heroes, all_trinkets_by_rarity)
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

    print(f"\n Data from {len(json_files)} files connected.")
    print(f"Processed {processed_teams_count} teams.")
    print(f"Saved {len(all_rows)} rows (variants x {VARIANTS_PER_TEAM}) into {csv_file_path}.")

main()