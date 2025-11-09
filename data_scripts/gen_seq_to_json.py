import csv
import json
import re
from typing import List, Dict, Any, Set
from utils import PATHS
import os 

def parse_sequence_to_heroes(sequence: str) -> List[Dict[str, Any]]:
    """
    Parsuje sekwencję tekstową na listę słowników bohaterów, wyodrębniając umiejętności i bibeloty.
    """
    teams_data: List[Dict[str, Any]] = []
    hero_strings = sequence.split('|')
    position = 1 
    
    for hero_str in hero_strings:
        if not hero_str.strip():
            continue
        
        parts = hero_str.split(':', 1)
        if len(parts) != 2:
            continue 
            
        hero_name = parts[0].strip()
        details_str = parts[1].strip()
        skills = []
        trinkets = []

        if ':' in details_str:
            try:
                skill_str, trinket_str = details_str.rsplit(':', 1)
                skills = [s.strip() for s in skill_str.split('+') if s.strip()]
                trinkets = [
                    (None if t.strip() == "<empty_trinket>" else t.strip())
                    for t in trinket_str.split('+')
                    if t.strip()
                ]

            except ValueError:
                pass
        else:
            all_items = [s.strip() for s in details_str.split('+') if s.strip()]
            if len(all_items) >= 2:
                skills = all_items[:-2]
                trinkets = all_items[-2:]
            else:
                skills = all_items
                trinkets = []


        teams_data.append({
            "name": hero_name,
            "position": position,
            "skills": skills,
            "trinkets": trinkets
        })
        position += 1

    return teams_data


def load_from_csv_generated(csv_file_path: str) -> List[Dict[str, Any]]:
    
    final_builds: List[Dict[str, Any]] = []
    current_build_id = 1 
    processed_sequences = set() 
    
    print("Walidacja będzie przeprowadzana W KONTEKŚCIE tokenów dostępnych tylko w DANYM WIERZSU 'input_sequence'.")

    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Iteracja po wierszach
            for row_num, row in enumerate(reader, 1):
                generated_sequence = row.get("generated_sequence")
                input_sequence = row.get("input_sequence")

                if not generated_sequence or not generated_sequence.strip():
                    continue

                valid_tokens: Set[str] = set()
                if input_sequence and input_sequence.strip():
                    tokens = re.split(r'[|:+]', input_sequence.strip())
                    valid_tokens.update({t.strip() for t in tokens if t.strip()})
                
                if not valid_tokens:
                    print(f"Ostrzeżenie (Wiersz {row_num}): Puste 'input_sequence'. Pomijam walidację.")
                    continue
                
                heroes_data = parse_sequence_to_heroes(generated_sequence)
                
                if generated_sequence in processed_sequences:
                    continue
                processed_sequences.add(generated_sequence)

                if heroes_data:
                    all_generated_trinkets: Set[str] = set()
                    
                    for hero in heroes_data:
                        all_generated_trinkets.update(hero.get('trinkets', []))

                    missing_trinkets = [
                        t for t in all_generated_trinkets
                        if t not in valid_tokens and t not in (None, "<empty_trinket>")
                    ]


                    if missing_trinkets:
                        print(f"ANOMALIA (Wiersz {row_num})!")
                        print(f" Input (Roster + Trinkets): {input_sequence[:100]}...")
                        print(f" Generated: {generated_sequence}")
                        print(f" BŁĄD TRINKETÓW: Brakujące w 'input' (NIEDOZWOLONE): {', '.join(missing_trinkets)}\n")

                if generated_sequence.count(':') < 4 or generated_sequence.count('+') < 20:
                    print(f"Wiersz {row_num}: podejrzanie mało separatorów ':' lub '+' (może brakować skilli/trinketów).")


                heroes_data_sorted = sorted(heroes_data, key=lambda h: h['position'])

                if not heroes_data_sorted:
                    continue

                build = {
                    "build_id": current_build_id,
                    "build_description": "Auto-generated build from generated_sequence column.", 
                    "heroes": heroes_data_sorted
                }
                final_builds.append(build)
                current_build_id += 1


    except FileNotFoundError:
        print(f"Błąd: Plik CSV nie znaleziony pod ścieżką: {csv_file_path}")
        return []
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd podczas wczytywania CSV: {e}")
        return []

    return final_builds


def save_to_json(data: List[Dict[str, Any]], output_file_path: str):
    """
    Zapisuje dane do pliku JSON.
    """
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Pomyślnie zapisano {len(data)} unikalnych buildów do {output_file_path}")
    except Exception as e:
        print(f"Błąd podczas zapisywania pliku JSON: {e}")


if __name__ == "__main__":
    csv_file = "data_scripts/generated_output.csv" 
    json_output_file = PATHS.folder + "test_builds/restored_teams_generated_fixed.json" 
    
    print(f"Rozpoczynanie konwersji z {csv_file} na JSON, używając 'generated_sequence' i heurystyki...")
    
    restored_builds = load_from_csv_generated(csv_file)

    if restored_builds:
        save_to_json(restored_builds, json_output_file)
    else:
        print("Nie wygenerowano żadnych buildów do zapisania.")

