import pickle
import numpy as np
import pandas as pd


def split_data(data_dict, train_size, test_icl_size, test_validation_size):
    data_dict_keys = list(data_dict.keys())
    np.random.shuffle(data_dict_keys)

    # train icl_data is the set of data used to generate the hypothesis
    # test icl data is the set of data available for few-shot prompting
    # note that examples will likely be reused for few-shot prompting during testing
    # test validation data is the set of data used to evaluate the model

    train_keys = data_dict_keys[:train_size]
    test_icl_keys = data_dict_keys[train_size:train_size+test_icl_size]
    test_validation_keys = data_dict_keys[train_size+test_icl_size:train_size+test_icl_size+test_validation_size]

    train_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in train_keys}
    test_icl_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in test_icl_keys}
    test_validation_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in test_validation_keys}

    return train_data, test_icl_data, test_validation_data
