import numpy as np
import pandas as pd

import MHNEC
from MHNEC.experiments import *
from MHNEC.analyses import *

from pathlib import Path
import os
import sys
import shutil
import argparse

import time
import pickle

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plafrim', action='store_true', help='Use PlaFRIM: https://www.plafrim.fr/')
    parser.add_argument('-c', '--classtorun', help="Name of the experiment or analysis class to run")
    parser.add_argument('-xpid', '--experiment_id', help='Name of the experiment to analyse if running an analysis, or name to give to the experiment to run.')

    return parser

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    class_to_run = getattr(sys.modules[__name__], args.classtorun)
    class_to_run(args)()
