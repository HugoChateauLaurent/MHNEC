import sys
import pickle
import numpy as np
import os
import random

class Task(object):
    """docstring for Task"""
    def __init__(self, code_path, simpath):
        super(Task, self).__init__()
        self.codepath = code_path
        self.simpath = simpath

class JobDispatcher(object):
    """docstring for JobDispatcher"""
    def __init__(self, xp_name):
        super(JobDispatcher, self).__init__()
        self.tasks = []
        self.nodes = ["sirocco"+n for n in ["22","23","24","25"]] #["07","08","09","10","11","12","13","14","15","16","17"]]
        self.xp_name = xp_name

    def launch_job(self, tasks, node):

        sh_script = """#!/usr/bin/env bash
#SBATCH -J MHNEC # name of job
#SBATCH --nodelist={node} # Ask for sirocco nodes (if less tasks than nodes then slurm adjusts list automatically)
#SBATCH -t2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -o {log_path}/{node}.out # standard output message
#SBATCH -e {log_path}/{node}.err # output error message

# Load modules
module purge
module load language/python/3.9.6
module load compiler/cuda/11.7
module load dnn/cudnn/11.2-v8.1.1.33
module load compiler/gcc/11.2.0
module load mpi/openmpi/4.1.1

source env/bin/activate

echo “=====my job informations ====”
echo “Node List: ” $SLURM_NODELIST
echo “my jobID: ” $SLURM_JOB_ID
echo “Partition: ” $SLURM_JOB_PARTITION
echo “submit directory:” $SLURM_SUBMIT_DIR
echo “submit host:” $SLURM_SUBMIT_HOST
echo “In the directory: `pwd`”
echo “As the user: `whoami`”""".format(node=node, log_path="logs/"+self.xp_name+"/slurm")

        for i,t in enumerate(tasks):
            sh_script += "\n"
            sh_script += "srun -N1 -n1 -c1 --exclusive python3 plafrim_launcher.py --codepath " + t.codepath + " --simpath " + t.simpath

        sh_path = "logs/"+self.xp_name+"/scripts/launch_{node}.sh".format(node=node)

        os.makedirs("logs/"+self.xp_name+"/scripts", exist_ok=True)
        os.makedirs("logs/"+self.xp_name+"/slurm", exist_ok=True)
        with open(sh_path, "w") as file:
            file.write(sh_script)

        os.system("sbatch "+sh_path)


    def launch_jobs(self):
        random.shuffle(self.tasks)
        splitted_tasks = np.array_split(self.tasks, len(self.nodes))
        for i,n in enumerate(self.nodes):
            self.launch_job(splitted_tasks[i], n)
