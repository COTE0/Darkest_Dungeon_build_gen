import json
import csv
import glob
import random
import yaml
import os
import sys
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import PATHS
from model_core import TOKEN_SPECIALS

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
        print(f"YAML loading error: {e}")
        return [], {}


def generate_variant(team: Dict[str, Any], variation_index: int, all_heroes: List[str], all_trinkets_by_rarity: Dict[str, List[str]]) -> Dict[str, str]:    
    team_tokens = []
    original_trinkets = []
    heroes = sorted(team["heroes"], key=lambda hero: hero["position"])

    for hero in heroes:
        trinkets = hero.get("trinkets", [])
        if not trinkets:
            trinket_str = TOKEN_SPECIALS['EMPTY_TRINKET']
        elif len(trinkets) == 1:
             trinket_str = f"{trinkets[0]}+{TOKEN_SPECIALS['EMPTY_TRINKET']}"
        else:
             trinket_str = "+".join(trinkets[:2])

        hero_str = (
            f"{hero['name']}:" +
            "+".join(hero.get("skills", [])) +
            "+" +
            trinket_str
        )
        team_tokens.append(hero_str)
        original_trinkets.extend(hero.get("trinkets", []))
        
    target_sequence = "|".join(team_tokens)
    
    
    core_heroes = [h['name'] for h in team['heroes']]
    
    roster_size = random.randint(8, 28) #generalizing for smaller pool if user does not wish to include certain characters(eg. when 1 has over 100 stress)
        
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

def add_grammar_tokens(sequence: str) -> str:
    parts = [p.strip() for p in sequence.split('|') if p.strip()]
    new_parts = []
    position = 1

    for part in parts:
        if ':' not in part:
            continue
            
        try:
            hero, rest = part.split(':', 1)
        except ValueError:
            print(f"Parsing error (no ':') in section: {part}")
            continue
            
        tokens = [x.strip() for x in rest.split('+') if x.strip()]
        
        if len(tokens) < 5:
            print(f"Parsing error (too few tokens) in section: {part}")
            continue
            
        skills = tokens[:4]
        trinkets = tokens[4:]

        formatted = (
            f"{TOKEN_SPECIALS['HERO']}{hero.strip()}"
            f"{TOKEN_SPECIALS['POS']}{position}:"
            f"{TOKEN_SPECIALS['SKILL']}{'+'.join(skills)}+"
            f"{TOKEN_SPECIALS['TRINKET']}{'+'.join(trinkets)}"
        )
        new_parts.append(formatted)
        position += 1

    return "|".join(new_parts)

def main():
    all_heroes, all_trinkets_by_rarity = load_game_data()
    
    if not all_heroes or not all_trinkets_by_rarity:
        print("Failed to load game data. Aborting.")
        return

    json_files = glob.glob(PATHS.jsons)
    all_rows = []
    VARIANTS_PER_TEAM = 10
    processed_teams_count = 0

    if not json_files:
        print(f"ERROR: No JSON files found in {PATHS.jsons}")
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
                    
                    if not reconstructed_team["heroes"]:
                        continue

                    for _ in range(VARIANTS_PER_TEAM):
                        row = generate_variant(reconstructed_team, i, all_heroes, all_trinkets_by_rarity)
                        all_rows.append(row)

        except FileNotFoundError:
            print(f"ERROR: file not found - {file_path}")
        except json.JSONDecodeError:
            print(f"ERROR: incorrect json format - {file_path}")
        except Exception as e:
            print(f"ERROR while processing file: {file_path}: {e}")

    if not all_rows:
        print("No rows were generated. Check your JSON files.")
        return

    csv_file_path = PATHS.csv
    try:
        with open(csv_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input_sequence", "target_sequence"])
            writer.writeheader()
            writer.writerows(all_rows)
    except IOError as e:
        print(f"Error writing to {csv_file_path}: {e}")
        return

    special_csv_path = os.path.join(PATHS.folder, "teams_dataset_with_specials.csv")
    all_rows_with_specials = []
    
    for row in all_rows:
        try:
            special_target = add_grammar_tokens(row['target_sequence'])
            if special_target:
                all_rows_with_specials.append({
                    "input_sequence": row['input_sequence'],
                    "target_sequence": special_target
                })
        except Exception as e:
            print(f"Error while adding special tokens for row: {row['target_sequence']}. Error: {e}")

    try:
        with open(special_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["input_sequence", "target_sequence"])
            writer.writeheader()
            writer.writerows(all_rows_with_specials)
    except IOError as e:
        print(f"Error writing to {special_csv_path}: {e}")
        return

    print(f"Merged data from {len(json_files)} JSON files.")
    print(f"Processed {processed_teams_count} team variations.")
    print(f"Saved {len(all_rows)} rows to {csv_file_path}.")
    print(f"Saved {len(all_rows_with_specials)} rows to {special_csv_path}.")

if __name__ == "__main__":
    main()