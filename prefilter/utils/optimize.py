import numpy as np
from argparse import ArgumentParser

def _parse_tunable_args(tuning_list):
    float_values = ('b', 'begin', 'e', 'end', 's', 'step')
    true_false_values = ('log', 'random')
    if len(tuning_list) == 1:
        # we've been given a single value for this param, tunable = False
        return float(tuning_list[0])

    var_to_arg = {'log': False,
                  'random': False} # initialize with default

    for arg in tuning_list:
        if arg in true_false_values:
            var_to_arg[arg] = True

    for present_arg in var_to_arg:
        try:
            # try to remove true/false values from list since they were
            # overwritten in the above for loop.
            tuning_list.remove(present_arg)
        except ValueError:
            # if they aren't present in the tuning list, we'll fall back to
            # defaults.
            continue

    parsed = 0
    for float_value in float_values:
        try:
            idx = tuning_list.index(float_value)
            var_to_arg[tuning_list[idx]] = tuning_list[idx+1]
            parsed += 2
        except ValueError:
            continue

    if parsed != len(tuning_list):
        for present_arg, value in var_to_arg.items():
            tuning_list.remove(present_arg)
            tuning_list.remove(value)
        raise ValueError(f'unrecognized arguments: {tuning_list}')

    if var_to_arg['random'] and ('step', 's') in var_to_arg:
        raise ValueError('cannot specify both random and a step value')

    if len(var_to_arg) < 5 and not (('s', 'step') not in var_to_arg and var_to_arg['random']):
        raise ValueError('must specify begin, end, and step or just enter 1 value')

    return var_to_arg


def add_tunable_arg(argument_parser, name, type_default):
    """
    tunable arguments:
    --params_to_tune begin end step log random
    """
    argument_parser.add_argument(f'--{name}', nargs="+", default=type_default,
                                 help=f'usage: --{name} N_FFT or b/begin BEGIN e/end END s/step STEP log '
                                      'random')

    return argument_parser


def _standardize_names(dct):
    o = {}
    for k, v in dct.items():
        if k == 'b':
            o['begin'] = v
        elif k == 'e':
            o['end'] = v
        elif k == 's':
            o['step'] = v
        else:
            o[k] = v
    return o


class SlurmOptimizer:
    """
    Class for doing hyperparameter tuning with slurm.
    a) assume the training routine is atomized and runs through
    a train() or main() function.
    b) ingest an argument parser and parse it to generate trials
       b.1) the argument parser will have two sub-groups: tunable and non-tunable.
       each tunable argument will have nargs="+" as its action and
    c) use subprocess.call() or subprocess.run()
    """

    def __init__(self, log_dir, n_max_trials, args):
        self.args = args
        self.n_max_trials = n_max_trials
        self.log_dir = log_dir

    # def _gen_random_trial(self, val, params):
    #    argument_list = []
    #    params = _standardize_names(params)
    #    if params["log"] and params["random"]:
    #    elif params["log"] and not

    # def generate_trials(self):
    #     unique_configs = []
    #     for param, value in self.args:
    #         if isinstance(value, dict):







if __name__ == '__main__':

    ap = ArgumentParser()
    add_tunable_arg(ap, 'n_fft', int)
    add_tunable_arg(ap, 'lr', float)
    args = ap.parse_args()
    ap2 = ArgumentParser()
    ap2.parse_known_args(args=['--frog', 'dog'])
    # argument_dict = vars(args)
    # argument_dict = {k: _parse_tunable_args(v) for k, v in argument_dict.items()}
    # print(argument_dict)


