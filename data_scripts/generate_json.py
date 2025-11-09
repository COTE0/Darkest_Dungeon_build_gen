import json
from typing import Any, Dict, List
from data_preparer import DataPreparer
from team_generator import TeamGenerator
from utils import merge_team_data
from datetime import datetime
from utils import PATHS
def generate_and_to_file() -> List[Dict[str, Any]]:
    try:
        preparer = DataPreparer()
        
        generator = TeamGenerator(preparer)
        
        heroes_list = ["Highwayman", "Leper", "Vestal",
                          "Musketeer", "Occultist", "Jester", "Houndmaster"
                          "Hellion", "Bounty Hunter", "Man-at-Arms"]
        
        base_teams = generator.generate_base_teams(heroes_list)
        if not base_teams:
            print("Failed to generate base teams. Exiting.")
            return

        trinket_variations = generator.generate_trinket_variations(base_teams)
        if not trinket_variations:
            print("Failed to generate trinket variations. Exiting.")
            return
        
        print("Starting step 3: Merging data...")
        final_teams_data = merge_team_data(base_teams, trinket_variations)

        if final_teams_data:
            print('\n' + '='*40)
            print('Fully generated and Merged Teams')
            print('='*40)
            date = datetime.now().strftime("%Y%m%d%H%M%S") 
            filePath = f'{PATHS.builds}aoutput{date}.json'
            with open(filePath, 'w') as file:
                file.write(json.dumps(final_teams_data, indent=2, sort_keys=False))
                return final_teams_data
        else:
            print("Final data merge failed.")

    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"\nConfiguration or Data Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


for _ in range(5):
    generate_and_to_file()
    