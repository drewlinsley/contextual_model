import numpy as np
from copy import deepcopy

def grab_sampler(parameters):
    if parameters.hp_optim_type == 'uniform':
        ofun = uniform_sampling
    elif parameters.hp_optim_type == 'random':
        ofun = random_sampling
    elif parameters.hp_optim_type == 'random_linear':
        ofun = random_linear_sampling
    elif parameters.hp_optim_type == 'random_exp':
        ofun = random_exp_sampling
    elif parameters.hp_optim_type == 'none':
        ofun = pass_params
    return ofun

def uniform_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] - ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] + ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        lin_range = np.linspace(low_lim,high_lim,parameters.tune_max_scale[idx])
        new_parameters._DEFAULT_PARAMETERS[key] = np.random.choice(lin_range)
    return new_parameters

def random_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] - ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] + ((parameters.tune_max_scale[idx] * parameters.tune_step[idx]) / 2)
        new_parameters._DEFAULT_PARAMETERS[key] = (high_lim - low_lim) * np.random.random() + low_lim
    return new_parameters

def random_linear_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        low_lim = new_parameters._DEFAULT_PARAMETERS[key] / 10
        high_lim = new_parameters._DEFAULT_PARAMETERS[key] * 10
        new_parameters._DEFAULT_PARAMETERS[key] = (high_lim - low_lim) * np.random.random() + low_lim
    return new_parameters

def random_exp_sampling(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        # new_parameters._DEFAULT_PARAMETERS[key] = new_parameters._DEFAULT_PARAMETERS[key] ** np.random.uniform(low=-2.,high=2.)  # previously did [0, 2]
        hp_value = new_parameters._DEFAULT_PARAMETERS[key]
        new_parameters._DEFAULT_PARAMETERS[key] = abs(np.random.uniform(low=hp_value - 1, high=hp_value + 1) + hp_value) ** np.random.uniform(low=-2.,high=2.)  # previously did [0, 2]
    return new_parameters

def pass_params(parameters):
    new_parameters = deepcopy(parameters)
    for idx,key in enumerate(parameters.tunable_params):
        new_parameters._DEFAULT_PARAMETERS[key] = np.asarray(new_parameters._DEFAULT_PARAMETERS[key])
    return new_parameters
