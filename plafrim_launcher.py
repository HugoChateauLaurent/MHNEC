import pickle
import argparse
import MHNEC
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--codepath', required=True, help='path to code')
    parser.add_argument('--simpath', required=True, help='path to parameter file')
    args = parser.parse_args()

    with open(args.simpath+"/params.pickle",'rb') as f:
        params = pickle.load(f)
    simpath = os.path.abspath(args.simpath)
    os.chdir(args.codepath)

    params["xp_cls"].simulate(simpath, params)
