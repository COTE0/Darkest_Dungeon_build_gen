import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_core import TOKEN_SPECIALS

def add_grammar_tokens(sequence: str) -> str:
    parts = [p.strip() for p in sequence.split('|') if p.strip()]
    new_parts = []
    position = 1

    for part in parts:
        if ':' not in part:
            continue
        hero, rest = part.split(':', 1)
        tokens = [x.strip() for x in rest.split('+') if x.strip()]
        skills, trinkets = tokens[:4], tokens[4:]

        formatted = (
            f"{TOKEN_SPECIALS['HERO']}{hero.strip()}"
            f"{TOKEN_SPECIALS['POS']}{position}:"
            f"{TOKEN_SPECIALS['SKILL']}{'+'.join(skills)}+"
            f"{TOKEN_SPECIALS['TRINKET']}{'+'.join(trinkets)}"
        )
        new_parts.append(formatted)
        position += 1

    return "|".join(new_parts)
