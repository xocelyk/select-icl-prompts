import numpy as np
import pandas as pd
import numpy as np
import openai
from config import load_config
from dotenv import load_dotenv
import os

load_dotenv()

config = load_config()
prompts = config['prompts']

api_key = os.getenv('API_KEY')
api_base = os.getenv('API_BASE')
api_type = os.getenv('API_TYPE')
api_version = os.getenv('API_VERSION')
deployment_name = os.getenv('DEPLOYMENT_NAME')

openai.api_key = api_key
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version
deployment_name = deployment_name


def get_train_ts_label(icl_data):
    ts_list = []
    label_list = []
    train_keys = list(icl_data.keys())
    np.random.shuffle(train_keys)
    for key in train_keys:
        # take all keys except Label
        sample_ts = {k: v for k, v in icl_data[key].items() if k != 'Label'}
        sample_label = icl_data[key]['Label']
        ts_list.append(sample_ts)
        label_list.append(sample_label)
    return ts_list, label_list


def create_prompt(icl_data=None, test_data=None, messages=[], train_mode=False, test_mode=False):
    # train and test not mutually exclusive
    # if only train is on, we do not include the test prompt
    # if only test is on, we do not include the train prompt (zero-shot)

    if train_mode:
        train_ts_list, train_label_list = get_train_ts_label(icl_data)
        for i in range(len(train_ts_list)):
            messages.append({"role": "user", "content": ts_to_string(train_ts_list[i])})
            messages.append({"role": "assistant", "content": label_to_string(train_label_list[i])})

    if test_mode:
        test_ts, _ = get_test_ts_label(test_data)
        messages.append({"role": "user", "content": ts_to_string(test_ts)})

    return messages


def get_test_ts_label(test_data):
    return {k: v for k, v in test_data.items() if k != 'Label'}, test_data['Label']


def ts_to_string(ts_dict: dict) -> str:
    res = 'Data:\n'
    res += '\n'.join(f'{key}: {value}' for key, value in ts_dict.items())
    return res


def label_to_string(label):
    if label == 1:
        return config['prompts']['LABEL_1']
    else:
        return config['prompts']['LABEL_0']


def parse_response(response_string):
    # return 1 if correct, 0 if incorrect, -1 if invalid response
    if ('(A)' in response_string and '(B)' in response_string) or ('(A)' not in response_string and '(B)' not in response_string): # invalid response
        return -1
    else:
        return int('(A)' in response_string)


def get_response(prompt, temperature=0, num_responses=1, timeout=10):
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=prompt,
        temperature=temperature,
        n=num_responses,
        timeout=timeout
    )
    return [response.choices[i]['message']['content'] for i in range(min(num_responses, len(response.choices)))]
