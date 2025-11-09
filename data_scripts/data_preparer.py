from dotenv import load_dotenv
import yaml
import os
import google.generativeai as genai
from typing import Dict, List, Any
from utils import PATHS

class DataPreparer:
    def __init__(self):
        self.valid_trinkets: Dict[str, Any] = {}
        self.valid_heroes_data: List[Dict[str, Any]] = []
        self._load_config()
        self.valid_trinkets_optimized = self.compress_trinkets_for_prompt()

    def _load_config(self):
        dotenv_path = PATHS.env
        load_dotenv(dotenv_path=dotenv_path)
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        yaml_data_path = PATHS.yml
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set. Check your .env file.")
        if not yaml_data_path:
            raise ValueError("YAML_DATA_PATH is not set. Check your .env file.")

        genai.configure(api_key=gemini_api_key)
        self.yaml_data_path = yaml_data_path
        
        self._load_game_data()

    def _load_game_data(self):
        try:
            with open(self.yaml_data_path, 'r', encoding='utf-8') as file:
                game_data = yaml.safe_load(file)
                self.valid_trinkets = game_data.get('trinkets', {})
                self.valid_heroes_data = game_data.get('heroes', [])
                print("Loaded heroes and trinkets from yaml.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Game data file not found at: {self.yaml_data_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading YAML data: {e}")
    def compress_trinkets_for_prompt(self) -> dict:
        result = {}
        data = self.valid_trinkets
        for rarity, trinkets in data.items():
            result[rarity] = [
                {"n": t["name"], "c": t.get("class", None)}
                for t in trinkets
            ]
        return result