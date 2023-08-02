import json
import os
import pandas as pd
import pickle


def load_config(filepath='config.json'):
    with open(filepath, 'r') as f:
        config = json.load(f)

    data_mode = config['data_mode']

    prompts_dir = 'prompts'
    prompts_file_path = os.path.join(prompts_dir, f'{data_mode}.json')
    with open(prompts_file_path, 'r') as f:
        prompts = json.load(f)

    data_dict_dir = 'data/data_dict'
    data_dict_file_path = os.path.join(data_dict_dir, f'{data_mode}.pkl')
    with open(data_dict_file_path, 'rb') as f:
        data_dict = pickle.load(f)

    data_frame_dir = 'data/data_frame'
    data_frame_file_path = os.path.join(data_frame_dir, f'{data_mode}.csv')
    with open(data_frame_file_path, 'r') as f:
        data_frame = pd.read_csv(f)

    return {'data_mode': data_mode, 'prompts': prompts,
            'data_dict': data_dict, 'data_frame': data_frame}
