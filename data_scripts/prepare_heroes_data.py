import yaml

def preprocess_dd_data(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    hero_data = {hero['name']: {'skills': hero['skills'], 'trinkets': []} 
                  for hero in data.get('heroes', [])}

    trinkets_all = [] 
    
    for category, trinket_list in data.get('trinkets', {}).items():
        for trinket in trinket_list:
            trinket_name = trinket['name']
            hero_class = trinket.get('class')
            
            if not hero_class:
                trinkets_all.append(trinket_name)
            
            elif hero_class in hero_data:
                hero_data[hero_class]['trinkets'].append(trinket_name)

    for hero_name in hero_data:
        hero_data[hero_name]['trinkets'] = sorted(hero_data[hero_name]['trinkets'])

    trinkets_all = sorted(list(set(trinkets_all)))

    output_data = {
        'heroes': hero_data,
        'trinkets_all': trinkets_all
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

from utils import PATHS
input_yaml_path = PATHS.folder+'darkest_dungeon_data.yml'
output_yaml_path = PATHS.folder+'dd_hero_data.yml'
preprocess_dd_data(input_yaml_path, output_yaml_path)