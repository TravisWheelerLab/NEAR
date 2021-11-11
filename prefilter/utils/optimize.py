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


class Experiment:

    def __init__(self, **kwargs):
        self.params = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
            # TODO: change this
            self.params[k] = v
        self.args = []

    def __str__(self):
        for k, v in self.params.items():
            self.args.append(f"--{k} {v}")
        return ' '.join(self.args)

    def __repr__(self):
        return str(self)

    def __setitem__(self, key, value):
        self.params[key] = value

    def __getitem__(self, key):
        return self.params[key]


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

    def __init__(self, hparams):

        self.statics = []
        self.stochastics = []
        self.uniform = []

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

    def get_new_experiment(self):
        if len(self.experiments) != 0:
            # grab a random set of uniform hparams if they are available
            base_params = self.experiments[int(np.random.rand() * len(self.experiments))]
        else:
            # if the user didn't specify any uniform hparams, just grab the statics
            base_params = self.base_parameter_set

        # now add in the randomly generated hparams
        for stochastic in self.stochastics:
            base_params[stochastic.name] = stochastic.sample()

        return Experiment(**base_params)

    def __next__(self):
        return self.get_new_set_of_hparams()

    def __iter__(self):
        return self

    def generate_cartesian_prod_of_uniform_hparams(self):
        # generate all combinations of hyperparameters
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

    def __init__(self, config_file):
        self.hparams = HPConf(config_file)
        self.experiments = ExperimentGenerator(self.hparams)
        self.poll_interval = 1
        self.file_to_process = {}

        t = str(int(time.time()))

        self.supervisor_dir = os.path.join(self.hparams.logs_dir, "run_" + t, 'supervisor')
        self.script_dir = os.path.join(self.hparams.logs_dir, "run_" + t, 'scripts')
        self.stdout_dir = os.path.join(self.hparams.logs_dir, "run_" + t, 'stdout')

        os.makedirs(self.supervisor_dir, exist_ok=True)
        os.makedirs(self.script_dir, exist_ok=True)
        os.makedirs(self.stdout_dir, exist_ok=True)

    def submit_experiment(self, exp, exp_num):
        script_path = self._create_bash_script(self.hparams, exp, exp_num)
        stdout_path = os.path.join(self.stdout_dir, exp['exp_id'] + '.stdout')
        result = subprocess.Popen(
            f"bash {script_path} >> {stdout_path} 2>&1", shell=True
        )
        self.file_to_process[os.path.splitext(os.path.basename(script_path))[0]] = result
        return result.pid

    def _create_bash_script(self, hparams, exp, exp_num):
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

        job_with_version = f"{hparams.project_name}v{exp_num}"

        exp['supervisor_dir'] = self.supervisor_dir
        exp['exp_id'] = job_with_version

        run_cmd = f"{hparams.run_command} {exp}"

        bash_script = '\n'.join(sub_commands)
        bash_script += '\n' + run_cmd + '\n'
        bash_file = self.write_bash_script_to_file(bash_script, job_with_version)

        return bash_file

    def get_new_experiment(self):
        return self.experiments.get_new_experiment()

    def write_bash_script_to_file(self, script, job_with_version):
        fout = os.path.join(self.script_dir, f'{job_with_version}.sh')
        with open(fout, 'w') as dst:
            dst.write(script)
        return fout

    def supervise(self, num_iters, n_lowest_to_terminate):

        trials = self.get_valid_experiments(self.supervisor_dir)
        losses = list(map(lambda x: x is None, [load_and_parse_logfile(t, num_iters) for t in trials]))

        # there's got to be a better way to handle this
        if len(losses) != 0 and np.any(losses):
            trials = []

        while len(trials) < len(self.file_to_process):

            trials = self.get_valid_experiments(self.supervisor_dir)
            # if the trial is valid (i.e. it's one that hasn't been canceled)
            # we need to check if it's still running. If it's terminated on its own,
            # we need to remove it from self.file_to_process since it doesn't

            losses = list(map(lambda x: x is None, [load_and_parse_logfile(t, num_iters) for t in trials]))
            bad = [i for i, x in enumerate(losses) if x]
            bad_trials = [trials[i] for i in bad]

            for trial in bad_trials:
                bs = os.path.splitext(os.path.basename(trial))[0]
                pid = self.file_to_process[bs].pid
                proc = psutil.Process(pid)
                if proc.status() == 'zombie':
                    del self.file_to_process[bs]

            if np.any(losses):
                trials = []

            time.sleep(self.poll_interval)

        candidates_for_termination = []
        for trial in trials:
            loss = load_and_parse_logfile(trial, num_iters)
            trial_to_loss = {'trial_file': trial, 'loss': loss}
            candidates_for_termination.append(trial_to_loss)

        candidates_for_termination = sorted(candidates_for_termination, key=lambda x: x['loss'])

        for exp in candidates_for_termination[-n_lowest_to_terminate:]:
            basename = os.path.splitext(os.path.basename(exp['trial_file']))[0]
            if basename in self.file_to_process:
                pid = self.file_to_process[basename].pid
                os.kill(pid, signal.SIGTERM)
                os.remove(exp['trial_file'])
                del self.file_to_process[basename]
                print(f"terminated experiment {basename} (pid {pid}) with loss {exp['loss']}")
            else:
                print(f"couldnt find {exp['version']} // did you kill it?")

    def get_valid_experiments(self, supervisor_dir):
        trials = glob(os.path.join(supervisor_dir, "*"))
        valid_trials = []
        for trial in trials:
            if os.path.splitext(os.path.basename(trial))[0] in self.file_to_process:
                valid_trials.append(trial)
        return valid_trials


def hyperband(supervisor):

    max_iter = 100  # maximum iterations / epochs per configuration
    eta = 3  # defines downsampling rate (default=3)
    logeta = lambda x: np.log(x) / np.log(eta)
    s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
    B = (s_max + 1) * max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)
    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(int(B / max_iter / (s + 1)) * eta ** s))  # initial number of configurations
        r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for
        print(f"iteration {s_max - s} of HyperBand outside loop, submitting {n} experiments")
        j = 0
        while j < n:
            exp = supervisor.get_new_experiment()
            supervisor.submit_experiment(exp, exp_num=j)
            j += 1

        for i in range(s + 1):
            n_i = n * eta ** (-i)
            r_i = r * eta ** i
            print(f"iteration {i} of HyperBand inside loop, throwing out {int(n_i / eta)}, running for {int(r_i)} steps")
            supervisor.supervise(num_iters=int(r_i),
                                 n_lowest_to_terminate=int(n_i / eta))


class SlurmSupervisor:

    def __init__(self, config_file):
        self.hparams = HPConf(config_file)
        self.experiments = _generate_experiments(self.hparams.hparams, self.hparams.n_trials)

    def submit_experiment(self, exp, exp_num):
        script_path = self._create_slurm_script(self.hparams, exp, exp_num)
        result = subprocess.Popen(
            f"sbatch {script_path}".split(), shell=True
        )
        return result

    def get_max_experiment_version(self):
        script_files = glob(os.path.join(self.hparams.out_dir, "scripts", '*sh'))
        versions = []
        for s in script_files:
            bs = os.path.splitext(os.path.basename(s))[0]
            try:
                version_num = int(bs[bs.rfind('v') + 1:])
                versions.append(version_num)
            except ValueError:
                print(f"found file that did not conform to naming spec: {s}")
                continue
        return max(versions) + 1 if len(versions) != 0 else 0

    def _create_slurm_script(self, hparams, exp, exp_num):
        sub_commands = []

        header = [
            "#!/bin/bash\n",
        ]
        sub_commands.extend(header)
        job_with_version = f"{hparams.project_name}v{exp_num}"
        command = [
            f"#SBATCH --job-name={job_with_version}\n",
        ]
        sub_commands.extend(command)

        # set an outfile.
        slurm_out_path = os.path.join(hparams.out_dir, 'out', f"{job_with_version}_slurm_output.out")
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
        exp['supervisor_dir'] = hparams.supervisor_dir
        # b) the process PID
        exp['exp_id'] = "$SLURM_JOB_ID"

        run_cmd = f"{hparams.run_command} {exp}"

        slurm_script = '\n'.join(sub_commands)
        slurm_script += '\n' + run_cmd + '\n'
        slurm_file = self.write_slurm_script_to_file(slurm_script, job_with_version)

        return slurm_file

    def write_slurm_script_to_file(self, script, job_with_version):

        os.makedirs(os.path.join(self.hparams.out_dir, 'scripts'), exist_ok=True)
        fout = os.path.join(self.hparams.out_dir, 'scripts', f'{job_with_version}.sh')
        with open(fout, 'w') as dst:
            dst.write(script)
        return fout

    def run(self):
        i = self.get_max_experiment_version()
        for experiment in self.experiments:
            self.submit_experiment(experiment, i)
            i += 1


if __name__ == "__main__":
    # where should parameter list generation go?
    # probably in the slurm supervisor constructor
    f = 'hparams.yaml'

    x = CPUSupervisor(f)
    hyperband(x)
