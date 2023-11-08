import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('tableau-colorblind10')
import time
import datetime
import pickle
import random

from scipy.stats import sem

def formatted_time(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    format_string = '%Y_%m_%d_%Hh%Mm%Ss'
    formatted_time = datetime.datetime.fromtimestamp(timestamp).strftime(format_string)
    return formatted_time

def save_results(path, results):
    pickle.dump(results, open(path+"/results.pickle", "wb"))

def flatten_dict(d):
    new_d = {}
    for k,v in d.items():
        if type(v) is dict:
            for k_v, v_v in v.items():
                new_d["_".join([k_v,k])] = v_v
        else:
            new_d[k] = v

    return new_d

def make_fig(fig, ax, dir_name="./", name=None, show=False, save_png=True, save_pdf=True, close=True):
    if name is None:
        name = "_".join([str(ax.xaxis.get_label().get_text()), str(ax.yaxis.get_label().get_text())])
    dir_name = dir_name if dir_name[-1] == "/" else dir_name+"/"
    if save_pdf:
        print("Saving pdf figure to", dir_name+name+".pdf")
        fig.savefig(dir_name+name+".pdf", transparent=True, bbox_inches='tight', pad_inches=.01)
    if save_png:
        print("Saving png figure to", dir_name+name+".png")
        fig.savefig(dir_name+name+".png", transparent=False, bbox_inches='tight', pad_inches=.01)
    if show:
        plt.show()
    if close:
        plt.close()

def unique(a,k): # like np.unique with special sorting of mechanisms and target labels

        if k in ["mechanisms", "target_labels"]:
            unique_elements = []
            for element in a:
                if element not in unique_elements:
                    unique_elements.append(element)
            if k=="mechanisms":
                unique_elements = sorted(unique_elements, key=lambda x: (x is not None, x))
            else:
                unique_elements = sorted(unique_elements, key=lambda x: (x is not None, list(READABLE_TARGET_LABEL.keys()).index(tuple(x)) if x is not None else 0))
            return unique_elements
        else:
            return list(np.unique(a))