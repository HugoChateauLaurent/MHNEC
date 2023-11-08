import os
import numpy as np
import pickle
from opportunistic_pfc.plafrim_utils import Task,JobDispatcher

def main(traj_name, rerun, print_not_run):
    path = './logs/'+traj_name+'/simulations'
    dirs = sorted(os.listdir(path))
    not_run = []

    for d in dirs[:]:
        to_remove = []
        if os.path.isfile(path+"/"+d+"/results.pickle"):
            pass
            # print(os.path.getsize(path+"/"+d+"/results.pickle"))
        else:
            not_run.append(d)
            if print_not_run:
                print(d)



    print("Not run:", len(not_run), "/", len(dirs))
    # print("params left", len(params))

    if rerun:
        tasks = [Task('logs/'+traj_name+'/code', 'logs/'+traj_name+'/simulations/'+d) for d in not_run]
        jd = JobDispatcher(xp_name=traj_name)
        jd.tasks = tasks
        jd.launch_jobs()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--traj_name', required=True)
    parser.add_argument('-p', '--print_not_run', action='store_true')
    parser.add_argument('-r', '--rerun', action='store_true')
    
    args = parser.parse_args()

    main(args.traj_name, args.rerun, args.print_not_run)
