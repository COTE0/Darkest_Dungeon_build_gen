from google.api_core import retry
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional
from data_preparer import DataPreparer
from utils import response_to_json 

class TeamGenerator:
    def __init__(self, preparer: DataPreparer, model_name: str = 'gemini-2.5-flash', max_retries: int = 5):
      self.model = genai.GenerativeModel(model_name)
      self.valid_heroes_data = preparer.valid_heroes_data
      self.valid_trinkets = preparer.valid_trinkets
      self.max_retries = max_retries
      self.valid_trinkets_optimized = preparer.valid_trinkets_optimized

    def _generate_content_with_retry(self, prompt: str) -> genai.types.GenerateContentResponse:
      @retry.Retry(timeout=300.0, initial=5.0, deadline=self.max_retries * 5.0)
      def generate_content(model, prompt):
        return model.generate_content(prompt)
        
      return generate_content(self.model, prompt)

    def generate_base_teams(self, heroes: List[str]) -> Optional[List[Dict[str, Any]]]:
      print("Starting step 1: Generating base team compositions...")
      prompt = f""" You are a Darkest Dungeon 1 expert strategist. Generate **5 unique, well-synergized team compositions**.
The goal is to generate a diverse set of teams for a large-scale dataset. Therefore, the **absolute priority** is to **maximize the variety of heroes used across the 5 teams** and explore diverse build concepts. You don't have to create optimal hero combinations and I recomend making it chaotic as if a player had built it.
Only requirement is that the team must make sense.

Available heroes for this generation run:
{', '.join(heroes)}.

**Requirements for the 5 teams (Priorities 1 & 2 are most important):**

1.  **MAXIMIZE HERO DIVERSITY:** The 5 teams combined must use **at least 8 distinct heroes** from the available roster. No single hero can appear in more than two of the four teams.
2.  **STRATEGIC DIVERSITY:** Each of the 5 teams must have a **fundamentally different primary focus** or win condition. (e.g., one on high CRIT/Burst, one on backline BLIGHT/STUN, one on Prot/Guard/Marking, one on Shuffle/Lunge Mechanics).
3.  **Synergy and Coherence:** Each team must maintain:
    - skill synergy (Mark/Finishers, Stun/Blight/Bleed combos, Stress Control, Riposte setups, etc.).
    - correct hero positioning for the chosen skills.
    - tactical coherence (a clear, sensible plan for combat).
4. Make sure that skills match their character's position(eg. using Hands from the Abyss on Occultist when he's in position 3 or 4 is a mistake unless it's appropriate dance team which moves occultist to position 1 or 2)

Here is the full dataset of valid heroes and their skills:
{json.dumps(self.valid_heroes_data, indent=2)}

Each team should have a short tactical description ("build_description") describing its playstyle and primary synergy.

You can use up to 2 heroes with the same class, but only if the heroes enhance the team's overall synergy and effectiveness.

Return output ONLY in pure JSON list format:
[
  {{
    "build_id": 1,
    "build_description": "...",
    "heroes": [
      {{"name": "Highwayman", "position": 2, "skills": ["Wicked Slice", "Pistol Shot", "Duelist's Advance", "Open Vein"]}},
      ...
    ]
  }},
  ...
]
Do not include any explanation or text outside JSON.
"""        
      try:
          response = self._generate_content_with_retry(prompt)
          return response_to_json(response)
      except Exception as e:
          print(f"Error generating base team: {e}")
          return None

    def generate_trinket_variations(self, base_teams_data: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
      if not base_teams_data:
          return None

      print("Starting step 2: Generating trinket variations...")
      simplified_data = [{
          "build_id": build["build_id"],
          "build_description": build.get("build_description", ""),
          "heroes": [{"name": hero["name"]} for hero in build["heroes"]]
      } for build in base_teams_data]
      prompt = f"""
You are balancing Darkest Dungeon 1 team compositions for different progression stages.
For each team composition below (identified by 'build_id' and 'heroes' names),
create **5 team variations** that differ ONLY in trinket selection.

Each variation represents a different power level:

1. Early-game viable (common/uncommon trinkets)
2. Early-mid transition (uncommon/rare)
3. Mid-game (rare/very rare/uncommon)
4. Late-game (very rare/ancestral)
5. Optimized high-power build (very rare/ancestral/trophy or exceptional crystalline if synergistic)

You can assume that player is playing on rather high light level
Crystalline trinkets must appear **only in the 5th variation** and only if they provide direct synergy with the team concept.
Avoid using crystalline trinkets otherwise — they should be special and rare.

Base teams:
{json.dumps(simplified_data, indent=2)}

Available trinkets:
{json.dumps(self.valid_trinkets_optimized, separators=(',', ':'))}

When selecting trinkets:
- Match them to hero roles and skills.
- Ensure logical synergy within the team.
- Keep the team balanced across power levels.

Output only valid JSON list with structure:
[
  {{
    "build_id": 1,
    "teams": [
      {{
        "description": "Early-game sustain build",
        "heroes": [{{"name": "Crusader", "trinkets": ["Knight's Crest", "Damage Stone"]}}, ...]
      }},
      ...
    ]
  }},
  ...
]
Do not include any text outside the JSON.
"""  
      try:
          response = self._generate_content_with_retry(prompt)
          return response_to_json(response)
      except Exception as e:
          print(f"Error generating trinkets: {e}")
          return None