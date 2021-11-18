import itertools
import pdb
import time
import subprocess
from glob import glob
import os
import psutil
import signal
import yaml
from random import seed

from random import shuffle
import numpy as np

seed(0)
np.random.seed(0)


def _parse_version(s):
    bs = os.path.splitext(os.path.basename(s))[0]
    return int(bs[bs.rfind('v') + 1:])

class SlurmExperiment:

    def __init__(self,
                 experiment_dir,
                 max_iter,
                 experiment_id,
                 **hparams):

        self.experiment_dir = experiment_dir
        self.hparams = hparams
        self.experiment_id = experiment_id
        self.max_iter = max_iter
        self.hparams['max_iter'] = max_iter
        self.resubmit_cmd = None
        self.slurm_jobid = None
        os.makedirs(self.experiment_dir, exist_ok=True)

    def __str__(self):
        # TODO: this seems hacky for resubmitting
        args = []
        for k, v in self.hparams.items():
            args.append(f"--{k} {v}")
        args.append(f"--experiment_dir {self.experiment_dir}")
        if self.resubmit_cmd is not None:
            args.append(f"--{self.resubmit_cmd}")
        return ' '.join(args)

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, hparams):
        script_path = self._create_slurm_script(hparams)
        self.slurm_jobid = subprocess.check_output(
            f"sbatch --parsable {script_path}", shell=True
        )
        self.slurm_jobid = int(self.slurm_jobid)
        return self.slurm_jobid

    @property
    def completed(self):
        job_status = subprocess.check_output(
            f"sacct --format State -u {os.environ['USER']} -j {self.slurm_jobid}".split()
        )
        job_status = job_status.decode('utf-8')
        # TODO: will this always work?
        return "COMPLETED" in job_status

    def _create_slurm_script(self, hparams):
        sub_commands = []

        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)

        # make sure the entire logs are kept for resubmitting experiments
        command = [
            "#SBATCH --open-mode=append"
        ]
        sub_commands.extend(command)

        self.job_name_with_version = f'{hparams.project_name}v{self.experiment_id}'
        command = [
            f"#SBATCH --job-name={self.job_name_with_version}\n"
        ]
        sub_commands.extend(command)

        # set an outfile.
        slurm_out_path = os.path.join(self.experiment_dir, 'slurm_out.out')
        command = [
            f"#SBATCH --output={slurm_out_path}\n"
        ]
        sub_commands.extend(command)

        # add any slurm directives that the user specifies. No defaults are given.
        for cmd in hparams.slurm_directives:
            command = [
                f"#SBATCH {cmd}\n",
            ]
            sub_commands.extend(command)

        # add any commands necessary for running the training script.
        for cmd in hparams.environment_commands:
            command = [
                f"{cmd}\n",
            ]
            sub_commands.extend(command)

        # add commands to the experiment object that describe
        # a) the supervisor directory
        # b) the process PID
        self['exp_id'] = "$SLURM_JOB_ID"

        run_cmd = f"{hparams.run_command} {self}"

        slurm_script = '\n'.join(sub_commands)
        slurm_script += '\n' + run_cmd + '\n'

        slurm_file = os.path.join(self.experiment_dir, 'slurm_script.sh')

        with open(slurm_file, 'w') as dst:
            dst.write(slurm_script)

        return slurm_file


class BashExperiment:

    def __init__(self,
                 experiment_dir,
                 max_iter,
                 **hparams):

        self.experiment_dir = experiment_dir
        self.hparams = hparams
        self.max_iter = max_iter
        self.hparams['max_iter'] = max_iter
        self.resubmit_cmd = None

        os.makedirs(self.experiment_dir, exist_ok=True)
        self.process = None

    def __str__(self):
        # TODO: this seems hacky?
        args = []
        for k, v in self.hparams.items():
            args.append(f"--{k} {v}")
        args.append(f"--experiment_dir {self.experiment_dir}")
        if self.resubmit_cmd is not None:
            args.append(f"--{self.resubmit_cmd}")
        return ' '.join(args)

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        getattr(self, key)

    def submit(self, hparams):
        script_path = self._create_bash_script(hparams)
        stdout_path = os.path.join(self.experiment_dir, 'log_file.stdout')
        self.process = subprocess.Popen(
            f"bash {script_path} >> {stdout_path} 2>&1", shell=True
        )
        return self.process.pid

    def _create_bash_script(self, hparams):

        sub_commands = []
        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)
        # set an outfile.
        for cmd in hparams.environment_commands:
            command = [
                f"{cmd}\n",
            ]
            sub_commands.extend(command)

        run_cmd = f"{hparams.run_command} {self}"

        bash_script = '\n'.join(sub_commands)
        bash_script += '\n' + run_cmd + '\n'

        bash_file = os.path.join(self.experiment_dir, 'submit_script.sh')
        with open(bash_file, 'w') as dst:
            dst.write(bash_script)

        return bash_file


class HyperRange:
    """
    An object for sampling from ranges of hyperparameters.
    Initialized with
    low, high, step, <random>, <log>
    """

    def __init__(self, name, begin=None, end=None, step=None,
                 random=False, log=False, num=None,
                 value=None):

        self.value = value
        self.name = name

        self.random = random
        self.begin = begin
        self.end = end

        self.step = step
        self.num = num

        self.log = log
        self.arr = None

        if self.value is None:

            self.begin = float(begin)
            self.end = float(end)

            if self.step is not None:
                self.step = float(step)

            if self.num is not None:
                self.num = int(num)

            self._generate_params()

    def _generate_params(self):
        if self.random:
            self.arr = None
            return

        if self.log:
            if self.step:
                self.arr = np.logspace(self.begin, self.end,
                                       num=(end - self.begin) // self.step)
            elif self.num:
                self.arr = np.logspace(self.begin, self.end,
                                       num=self.num)
            else:
                raise ValueError(f"specify either step or num for argument {self.name}")
        else:
            if self.step:
                self.arr = np.arange(self.begin, self.end, self.step)
            elif self.num:
                self.arr = np.linspace(self.begin, self.end, self.num)
            else:
                raise ValueError(f"specify either step or num for argument {self.name}")

    def _sample_random(self):
        if self.log:
            return np.exp(self.begin + (self.end - self.begin) * np.random.rand())
        else:
            return self.begin + (self.end - self.begin) * np.random.rand()

    def __getitem__(self, idx):
        if self.value is not None:
            return self.value
        elif self.random:
            return self._sample_random()
        else:
            return self.arr[idx]

    def sample(self):
        if self.random:
            return self._sample_random()
        elif self.value is not None:
            return self.value
        else:
            return self.arr[int(np.random.rand() * len(self.arr))]

    def __iter__(self):
        return iter(self.arr)

    def __len__(self):
        return len(self.arr) if self.arr is not None else 1

    def __str__(self):
        if self.value is not None:
            return f'{self.name}: {self.value}'
        elif self.random:
            return f'stochastic hparam {self.name}: {self._sample_random()}'
        else:
            return f'{self.name}: ' + ' '.join(map(str, self.arr))

    def __repr__(self):
        return str(self)


def dct_from_yaml(yaml_file):
    with open(yaml_file, "r") as src:
        dct = yaml.safe_load(src)
    return dct


class HPConf:

    def __init__(self, config_file):
        self.params = dct_from_yaml(config_file)
        self.hparams = self.params['hparams']
        self.slurm_directives = self.params['slurm_directives']
        self.run_command = self.params['run_command']
        for k, v in self.params.items():
            setattr(self, k, v)


class ExperimentGenerator:

    def __init__(self, hparams: HPConf,
                 experiment_type: str) -> None:

        self.statics = []
        self.stochastics = []
        self.uniform = []
        self.hparams = hparams
        self.experiment_type = experiment_type

        for hparam, setting in hparams.hparams.items():
            hrange = HyperRange(hparam, **setting)

            if hrange.random:
                self.stochastics.append(hrange)
            elif len(hrange) > 1:
                self.uniform.append(hrange)
            else:
                self.statics.append(hrange)

        uniform_cartesian_product = self.generate_cartesian_prod_of_uniform_hparams()
        self.experiments = []

        self.base_parameter_set = {}
        # shove all of the static arguments to the training function into a dict containing
        # non-mutable hparams # TODO: change this in the config.yaml file.
        for static in self.statics:
            self.base_parameter_set[static.name] = static[0]

        if uniform_cartesian_product is not None:

            names = list(map(lambda x: x.name, self.uniform))

            for combination in uniform_cartesian_product:

                param_dict = self.base_parameter_set.copy()

                for name, val in zip(names, combination):
                    param_dict[name] = val

                self.experiments.append(param_dict)

    def submit_new_experiment(self, experiment_dir, max_iter,
                              experiment_id=None):

        if len(self.experiments) != 0:
            # grab a random set of uniform hparams if they are available
            base_params = self.experiments[int(np.random.rand() * len(self.experiments))]
        else:
            # if the user didn't specify any uniform hparams, just grab the statics
            base_params = self.base_parameter_set

        # now add in the randomly generated hparam
        for stochastic in self.stochastics:
            base_params[stochastic.name] = stochastic.sample()

        if self.experiment_type == 'slurm':
            exp = SlurmExperiment(experiment_dir, max_iter, experiment_id, **base_params)
        elif self.experiment_type == 'bash':
            exp = BashExperiment(experiment_dir, max_iter, **base_params)
        else:
            raise ValueError(f"experiment type should be one of [bash,slurm], got {self.experiment_type}")

        exp.submit(self.hparams)

        return exp

    def __iter__(self):
        return self

    def generate_cartesian_prod_of_uniform_hparams(self):
        if len(self.uniform):
            prod = list(itertools.product(*self.uniform))
            shuffle(prod)
            return prod
        else:
            return None


def load_and_parse_logfile(file, num_iter):
    loss = None

    with open(file, 'r') as src:
        lines = src.readlines()

    for line in lines:
        iteration, loss_candidate = line.split(":")
        if int(iteration) == int(num_iter):
            return float(loss_candidate.replace("\n", ""))

    return loss


class CPUSupervisor:

    def __init__(self, experiment_generator,
                 project_directory,
                 poll_interval=0.1):

        self.poll_interval = poll_interval
        self.file_to_process = {}
        self.project_directory = project_directory
        self.experiment_generator = experiment_generator
        self.running_experiments = []
        self.experiment_id = 0

    def submit_new_experiment(self, experiment_directory, max_iter):

        experiment_dir = os.path.join(self.project_directory, experiment_directory,
                                      f"exp_{self.experiment_id}")

        exp = self.experiment_generator.submit_new_experiment(experiment_dir,
                                                              max_iter=max_iter)
        self.experiment_id += 1
        self.running_experiments.append(exp)

    def watch_experiments(self, n_best_to_keep):
        while True:
            finished = 0
            for experiment in self.running_experiments:
                poll = experiment.process.poll()
                if poll is not None:
                    finished += 1
            time.sleep(self.poll_interval)
            if finished == len(self.running_experiments):
                print("Hyperband loop finished. Culling poorly-performing experiments.")
                break

        losses = []
        for experiment in self.running_experiments:
            log_path = os.path.join(experiment.experiment_dir, 'checkpoint.txt')
            with open(log_path, 'r') as src:
                step, loss = src.read().split(":")
            losses.append(float(loss))

        indices = np.argsort(losses)  # smallest metric first
        self.running_experiments = [self.running_experiments[i] for i in indices[0:n_best_to_keep]]

    def resubmit_experiments(self, max_iter):
        for experiment in self.running_experiments:
            experiment.max_iter = max_iter
            experiment.resubmit_cmd = 'load_from_ckpt'
            # TODO.. fix this.
            experiment.submit(self.experiment_generator.hparams)

def _many_or_one(x):
    return "iteration" if x == 1 else "iterations"


def hyperband(supervisor):
    max_iter = 100  # maximum iterations / epochs per configuration
    eta = 3  # defines downsampling rate (default=3)
    logeta = lambda x: np.log(x) / np.log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(B / max_iter / (s + 1)) * eta ** s)  # initial number of configurations
        r = int(max_iter * eta ** (-s))  # initial number of iterations to run configurations for
        print(f"iteration {s_max - s} of HyperBand outside loop, submitting {n} experiments,"
              f" each running for {r} " + _many_or_one(r))

        loop_directory = f"{s}_{n}_{r}"
        # the supervisor has a project directory and each outer loop
        # of hyperband has a subdirectory. Each experiment in each outer
        # loop has its own directory
        first = True
        for i in range(s + 1):
            n_i = int(n * eta ** (-i))  # submit N experiments
            r_i = int(r * eta ** i)  # and run for R iterations.
            if first:
                for _ in range(n_i):
                    supervisor.submit_new_experiment(experiment_directory=loop_directory,
                                                     max_iter=r_i)
                first = False
            else:
                supervisor.resubmit_experiments(max_iter=r_i)
            # this is a blocking method. It'll monitor whether or not the experiments are done.
            # When they are done, it'll resubmit the best-performing ones.
            supervisor.watch_experiments(n_best_to_keep=int(n_i / eta))
            print(f"Hyperband inner loop {i} finished. Keeping {int(n_i / eta)} experiments.")


class SlurmSupervisor:

    def __init__(self, experiment_generator,
                 project_directory,
                 poll_interval=0.1):

        self.poll_interval = poll_interval
        self.file_to_process = {}
        self.project_directory = project_directory
        self.experiment_generator = experiment_generator
        self.running_experiments = []
        self.experiment_id = 0

    def submit_new_experiment(self, experiment_directory, max_iter):

        experiment_dir = os.path.join(self.project_directory, experiment_directory,
                                      f"exp_{self.experiment_id}")

        exp = self.experiment_generator.submit_new_experiment(experiment_dir,
                                                              max_iter=max_iter,
                                                              experiment_id=self.experiment_id)
        self.experiment_id += 1
        self.running_experiments.append(exp)

    def watch_experiments(self, n_best_to_keep):

        while True:
            finished = 0
            for experiment in self.running_experiments:
                if experiment.completed:
                    finished += 1
            time.sleep(self.poll_interval)
            if finished == len(self.running_experiments):
                break

        losses = []
        for experiment in self.running_experiments:
            log_path = os.path.join(experiment.experiment_dir, 'checkpoint.txt')
            with open(log_path, 'r') as src:
                _, loss = src.read().split(":")
            losses.append(float(loss))

        indices = np.argsort(losses)  # smallest metric first
        self.running_experiments = [self.running_experiments[i] for i in indices[0:n_best_to_keep]]

    def resubmit_experiments(self, max_iter):
        for experiment in self.running_experiments:
            experiment.max_iter = max_iter
            experiment.resubmit_cmd = 'load_from_ckpt'
            # TODO.. fix this.
            experiment.submit(self.experiment_generator.hparams)


if __name__ == "__main__":
    # where should parameter list generation go?
    # probably in the slurm supervisor constructor
    f = 'hparams.yaml'
    hp = HPConf(f)
    exp_gen = ExperimentGenerator(hp, experiment_type='slurm')
    x = SlurmSupervisor(exp_gen, 'slurm', poll_interval=10)
    hyperband(x)
