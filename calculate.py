import logging
from pathlib import Path

import torch as t
import yaml
import gc
import new_process as process
import quan
import util
from model import create_model
import os
import random
import numpy as np
import wandb
import torchvision as tv

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.device_count() > 0:
        t.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    t.backends.cudnn.deterministic = True

def pruning_log(model):
    pruning_log = {}
    for n, m in model.named_modules():
        if hasattr(m,"quan_w_fn") and hasattr(m.quan_w_fn, "p"):
            pruning_log[n + "pruning_point"] = m.quan_w_fn.p.detach()
            pruning_log[n + "clipping_point"] = m.quan_w_fn.c.detach()
            pruning_log[n + "gamma"] = m.quan_w_fn.gamma.detach()
            pruning_log[n + "distance"] = (m.quan_w_fn.c- m.quan_w_fn.p).detach()
            pruning_log[n + "z"] = m.quan_w_fn.z.detach()

def model_parameter_log(model, logger, args):
    with t.no_grad():
        hard_sparsity = 0.
        total_zero = 0.
        total_numel = 0.
        total_bit = 0.
        for n,m in model.named_modules():
            if hasattr(m, "weight"):
                if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
                    m.weight.data = m.quan_w_fn(m.weight.detach())[0]
                    weight_zero = (m.weight == 0).sum()
                    
                    weight_numel = m.weight.detach().numel()
                    #print("pruning")
                    #print(weight_numel)
                    weight_bit = (weight_numel - weight_zero) * args.quan.weight.bit

                    total_zero += weight_zero
                    total_numel += weight_numel
                    total_bit += weight_bit
                elif hasattr(m, "quan_w_fn"):
                    m.weight.data = m.quan_w_fn(m.weight.detach())
                    weight_zero = (m.weight == 0).sum()
                    weight_numel = m.weight.detach().numel()
                    #print("no_pruning")
                    #print(weight_numel)
                    weight_bit = weight_numel * 32
                    
                    total_zero += weight_zero
                    total_numel += weight_numel
                    total_bit += weight_bit
        hard_sparsity = total_zero / total_numel
    
    logger.info("Hard Sparsity")
    logger.info(hard_sparsity)
    logger.info("Model Parameter Size")
    logger.info(str(total_bit.item()) + "Bit")
    logger.info(str(total_bit.item() / 8 * 1e-6) + "MB")

def model_bflops_log(model, logger, args):
    model = model.module.cpu()
    #model = create_model(args)
    with t.no_grad():
        macs, params = util.get_model_complexity_info(model, (3, 224, 224),
                                                as_strings = False,
                                                print_per_layer_stat = True,
                                                flops_units = "GMac")
        print(macs, params)
        bitops = 0.
        for n, m in model.named_modules():
            if hasattr(m, "__bops__"):
                bitops += m.__bops__
        for n, m in model.named_modules():
            if hasattr(m, "__bops__"):
                print(n, str(m.__bops__/bitops * 100) + "%")
        print(bitops)
def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    model = create_model(args)
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    
    resume_path = os.path.join(script_dir, args.resume.path)
    model = t.nn.DataParallel(model)
    model, _, _ = util.load_checkpoint(
            model, resume_path, args.device.type, lean = args.resume.lean)


    # 1. sparsity 
    logger.info("Pruning Sparsity")
    model_parameter_log(model, logger, args)
    
    # 2. Model Size
    model_bflops_log(model, logger, args)
if __name__ == "__main__":
    main()
