# How to use the Griz Cluster
Prerequisites: A Unix/Linux machine.
# Logging in.
Use the "ssh" command below. The cluster will prompt you for your password. Enter it and don't be surprised if no
characters show up in the terminal - it's a Linux security thing.
```
ssh <your_netid>@login.gscc.umt.edu
```
## Logging in _without a password_
Entering your password every time you want to use the cluster can become annoying. Use ssh key based authetication to
enable passwordless authentication. This is usually a painless process.
Steps below are from [this digitalocean link](https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server).
### Step 1 - Generating a key pair.
On your local computer, generate a SSH key pair by typing:
```
$ ssh-keygen
Output:
Generating public/private rsa key pair.
Enter file in which to save the key (/home/username/.ssh/id_rsa):
```
The utility will prompt you to select a location for the keys that will be generated. By default, the keys will be stored in the ~/.ssh directory within your user’s home directory. The private key will be called id_rsa and the associated public key will be called id_rsa.pub.
Usually, it is best to stick with the default location at this stage. Doing so will allow your SSH client to automatically find your SSH keys when attempting to authenticate. 
Next, the program will ask if you want to create a password:
```
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
```
This is optional. I personally don't have one set up.
### Step 2 - Copying your public key to the server.
Use `ssh-copy-id`.
```
$ ssh-copy-id <your_netid>@login.gscc.umt.edu.
```
If you've done everything right, this will be the last time you'll have to enter your password to get onto 
the cluster.
Now, ssh into the clusterwith `$ ssh <your_netid>@login.gscc.umt.edu`.
You'll see something like this:
```
Output
The authenticity of host '203.0.113.1 (203.0.113.1)' can't be established.
ECDSA key fingerprint is fd:fd:d4:f9:77:fe:73:84:e1:55:00:ad:d6:6d:22:fe.
Are you sure you want to continue connecting (yes/no)?
```
Type yes and you're good to go.

# Orienting yourself on the cluster.

You will automatically be placed in your home directory at `/home/<your_netid>`. Once you're connected, you can interact
with the cluster with typical linux commands: `ls`, `cat`, `top`, etc. You can create, delete, and edit files anywhere
in your home directory. This is where you will save scripts that you want to run as well as data. I suggest breaking up
your home directory into logical subfolders, like `data`, `code`, `scratch`, and so forth.

The different compute nodes on the cluster are connected via the internet to the same filesystem. This means you can upload
code to the login node and it will automatically propagate out to the compute nodes. No more moving your files to compute
nodes with `scp`! However, you still need to get your code onto the cluster somehow. 
Many IDEs can do this for you automatically. If you want to be old-school, use `scp` (stands for "secure copy", and its
syntax is identical to `cp`s syntax). `rsync` is another nice utility for transferring files.

To test scp, copy this script into a file named hello.py on your local macine:
```python
print("hello world. I'm on the cluster")
```
Now, move it to the cluster with
```shell
scp hello.py <your_netid>@login.gscc.umt.edu:/home/<your_netid>/
```
The scp command will place it in your home directory.

# Now that my code is on the cluster... how do I run it?
There are two options for running code on the cluster: `sbatch` and `srun`. They are both commands provided by the
[slurm](https://slurm.schedmd.com/documentation.html) cluster manager. `srun` is used for interactive jobs and `sbatch`
for headless jobs. 
`sbatch` requires writing a so called "slurm script" to specify the environment in which you want your code to run.
Slurm has a ton of options, but an example script could look like this:
```shell
#!/bin/bash

# disclaimer: this is not an exhaustive list of all available
# options

#SBATCH --partition=wheeler_lab_large_cpu,wheeler_lab_small_cpu,wheeler_lab_gpu
^ specify which partition you want to run your code on
#SBATCH --job-name=train
^ this name will show up when you run squeue
#SBATCH --output=test-%A-%a.out
^ this is where stdout from your job is saved
#SBATCH --error=test-%A-%a.error
^ & this is where stderr from your job is saved (if --error isn't specified,
stderr will be dumped to the --output file)
#SBATCH --gres=gpu:1
^ what "generalized resource" do you want? In this case I want 1 gpu (i've only
needed --gres when training NNs on the gpu)
#SBATCH --exclude="compute-1-[2,8]"
^ don't consider these nodes
#SBATCH --include="compute-1-[4,9]"
^ include these nodes
#SBATCH --nodes=1
^ only ask for 1 node (usually not necessary to specify this)
#SBATCH --array=[1-1000]%100
^ array job commands - i want 100 jobs numbered from 1-1000, and I want 100 of
them to be running on the cluster at any one time.

# The commands you want to run lie below the various slurm directives.
# It's usually helpful to put absolute paths to the scripts you want to run (if
# they're not on your $PATH). This helps me when debuggin errors.

# Pretty much all I do is write the same commands in this script that I would use
# to run the code interactively from the command line.

conda activate base
export MY_ENV='test'
./my_script_here.sh
python my_script.py
```
I use this script as a template when creating new slurm scripts. Lines that start with `#SBATCH` will be interpreted by
slurm. `#` without `SBATCH` indicate comments and carets indicate where I've annotated the file. After the `#SBATCH`
lines, you write the commands to run whatever code you want. This can include changing directories, listing files,
loading environments, and anything else you do on a Linux machine from the command line. Let's write a slurm script for
our hello.py file. Open your favorite text editor and save the below as slurm_script.sh.
```shell
#!/bin/bash
# first, include the "shebang" line above.
# This tells the computer which shell to use to interpret the subsequent commands.
# Now, specify a partition. The available ones are wheeler_lab_gpu, wheeler_lab_small_cpu, and wheeler_lab_large_cpu.
# You can include multiple by using a comma-separated list. Since our script is small, we only need the small partition.
#SBATCH --partition=wheeler_lab_small_cpu
# Specify a job name:
#SBATCH --job-name=my_first_slurmer
# And an output file:
#SBATCH --output=slurmski.out

# Make sure we're in the home directory:
cd
# python hello.py
```

That's it! Submit it to the slurm queue with `sbatch slurm_script.sh`. Wait. Slurm queue? What's that? Slurm schedules jobs
to operate on compute nodes using a FIFO queue. You can see the queue with `squeue`. By default, `squeue` is configured to
show you jobs submitted by people in your group (in this case, the Wheeler Lab). See only your jobs with `squeue -u $USER`
and everyone's jobs with `squeue -a`. `squeue` is very helpful for seeing what the status of your job is. Is it
completed (in which case it _won't_ show up in the queue)? Is it pending? Running?

The job should have completed while you were finishing that paragraph. Check the output with `cat slurmski.out`.
You should see `hello world. I'm on the cluster`.

Congratulations! You've run your first job. Now, copy your own scripts to the cluster, edit the slurm submission file, 
and experiment away!

