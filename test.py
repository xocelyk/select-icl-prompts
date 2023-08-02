import pickle
from multiprocessing import Process, Manager
from select_icl_sample import cluster_learn_sample, entropy_learn_sample, active_learn_sample
from few_shot import few_shot_one_example


def save_state(state_dict_proxy, filename):
    # Convert proxy dict to regular dict before pickling
    state_dict = dict(state_dict_proxy)
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f)


def load_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def reset_state(filename):
    test_validation_data_keys = pickle.load(open('data/experiment2/test_validation_data_keys.pkl', 'rb'))
    test_validation_data_keys = list(test_validation_data_keys)
    state_dict = {'correct': 0, 'incorrect': 0, 'total': 0, 'invalid': 0, 'test_keys_remaining': test_validation_data_keys, 'test_keys_used': []}
    save_state(state_dict, filename)


def runner(main_func, state_dict_proxy, *args, **kwargs):
    while len(state_dict_proxy['test_keys_remaining']) > 0:
        proc = Process(target=main_func, args=[state_dict_proxy, *args], kwargs=kwargs)  # Pass state_dict_proxy to the main function
        proc.start()
        proc.join(timeout=10)
        if proc.is_alive():
            print('TIMEOUT')
            proc.terminate()
            proc.join()
            continue


def few_shot(state_dict_proxy, icl_data, filename):
    test_validation_data_filename = 'data/experiment2/test_validation_data.pkl'

    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    test_keys_remaining = state_dict_proxy['test_keys_remaining']
    test_key = test_keys_remaining[0]

    result_dict = few_shot_one_example(icl_data=icl_data, test_validation_data=test_validation_data[test_key])
    result = result_dict['correct']
    if result == 1:
        state_dict_proxy['correct'] += 1
    elif result == 0:
        state_dict_proxy['incorrect'] += 1
    elif result == -1:
        state_dict_proxy['invalid'] += 1

    state_dict_proxy['total'] += 1
    state_dict_proxy['test_keys_remaining'] = state_dict_proxy['test_keys_remaining'][1:]
    state_dict_proxy['test_keys_used'] = state_dict_proxy['test_keys_used'] + [test_key]
    print('correct:', state_dict_proxy['correct'], 'incorrect:', state_dict_proxy['incorrect'], 'invalid:', state_dict_proxy['invalid'], 'total:', state_dict_proxy['total'], 'test keys remaining:', len(state_dict_proxy['test_keys_remaining']), 'test keys used:', len(state_dict_proxy['test_keys_used']))
    save_state(state_dict_proxy, filename)


def test_few_shot(icl_data, filename):
    reset_state(filename)
    with Manager() as manager:
        state_dict_proxy = manager.dict(load_state(filename))
        runner(few_shot, state_dict_proxy, icl_data, filename)
    
    results = load_state(filename)
    return results

if __name__ == '__main__':
    import datetime

    # active learning, cluster learn, entropy learn
    num_shots = 8
    state_dict_filename = 'data/experiment2/state_dict_few_shot_{}.pkl'.format(num_shots)
    icl_data = pickle.load(open('data/experiment2/test_icl_data.pkl', 'rb'))
    sample_method = entropy_learn_sample
    icl_data = sample_method(icl_data, num_shots)
    print(icl_data)
    results = test_few_shot(icl_data=icl_data, filename=state_dict_filename)
    print(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_filename = 'data/experiment2/results_few_shot_{}_{}.pkl'.format(num_shots, timestamp)
    save_state(results, save_filename)
