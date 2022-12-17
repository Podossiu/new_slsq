import logging
from pathlib import Path

import torch as t
import yaml

import new_process as process
import quan
import util
from model import create_model
import os
import random
import numpy as np
import wandb
import matplotlib.pyplot as plt
import argparse 
import scipy
def make_sparsity(model, args):    
    directory = os.path.join('./sparsity_figure', args.name)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    from tqdm import tqdm;
    with t.no_grad():
        with tqdm() as pbar:
            for n, m in model.named_modules():
                pbar.set_description(n)
                pbar.update()
                if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
                    m.quan_w_fn.hard_pruning = True
                    weight = m.weight.reshape(-1).abs()
                    weight_q = m.quan_w_fn(m.weight).reshape(-1).abs()
                    nbins = 100
                    #bins_w = np.linspace(0, weight.max(), nbins + 1)
                    #bins_wq = np.linspace(0, weight_q.max(), nbins + 1)
                    #bins_w = np.linspace(0, weight.max(), nbins + 1)
                    #bins_wq = np.linspace(0, weight_q.max(), nbins + 1)
                    range_w = (0, weight.max().numpy())
                    bins_w = np.histogram_bin_edges(weight, bins = nbins, range = range_w)
                    
                    bins_wq = np.histogram_bin_edges(weight_q, bins = nbins, range = (0, weight_q.max().numpy()))
                    ind_w = np.digitize(weight, bins_w)
                    ind_wq = np.digitize(weight_q, bins_wq)
                    

                    bins_num = [np.sum(ind_w == j).astype(float) for j in range(1,nbins)]
                    bins_sparse = [np.sum((ind_w == j).astype(float) * (ind_wq == 1)) for j in range(1, nbins)]
                    bins_sparsity = t.tensor(bins_sparse) / (t.tensor(bins_num) + 1e-12)

                    plt.plot(bins_w[1:-1], bins_sparsity)
                    plt.axvline(x = m.quan_w_fn.p.clone().detach().cpu(), color = "red", linestyle = "dashed")
                    plt.title(n + "sparsity")
                    plt.savefig(directory + '/'+n+ '.png')
                    plt.close()
                        

                
def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file = script_dir / 'config.yaml')
    model = create_model(args)
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    
    model = quan.replace_module_by_names(model, modules_to_replace)
    
    model = t.nn.DataParallel(model)
    resume_path = os.path.join(script_dir, args.resume.path)
    model,_,_ = util.load_checkpoint(
            model, resume_path, args.device.type, lean = args.resume.lean)
    model.cpu()
    make_sparsity(model, args)
if __name__ == "__main__":
    main()
