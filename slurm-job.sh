#!/bin/bash
#SBATCH --job-name=an-cockrell         # Job name
#SBATCH --mail-type=ALL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adam.knapp@ufl.edu # Where to send mail  
#SBATCH --nodes=1                      # Use one node (non-MPI)
#SBATCH --ntasks=1                     # Run a single task (non-MPI)
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb              # Memory per job
#SBATCH --time=72:00:00                # Time limit hrs:min:sec
#SBATCH --output=an-cockrell.out  # Standard output and error log
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users. 
pwd; hostname; date

module load python
cd /home/adam.knapp/blue_rlaubenbacher/adam.knapp/learning-and-control-of-abms/

python3 an-cockrell-runner.py

date


