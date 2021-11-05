import numpy as np
from argparse import ArgumentParser

def _parse_tunable_args(tuning_list):

    subparser = ArgumentParser()
    subparser.add_argument('b', 'begin')
    subparser.add_argument('e', 'end')
    subparser.add_argument('l', 'log', action='store_true')
    mutex = subparser.add_mutually_exclusive_group()
    mutex.add_argument('s', 'step')
    mutex.add_argument('r', 'random', action='store_true')

    command_specific_args = subparser.parse_known_args(tuning_list)
    print(command_specific_args)
    return command_specific_args


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
    argument_dict = vars(args)
    argument_dict = {k: _parse_tunable_args(v) for k, v in argument_dict.items()}
    # print(argument_dict)


