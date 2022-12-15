import logging
from pathlib import Path

import torch as t
import yaml

import process
import quan
import util
from model import create_model
import os
import random
import numpy as np
import wandb

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.device_count() > 0:
        t.cuda.manual_seed_all(seed)

def calculate_compression_ratio(model, args):
    weight_numel = 0.
    block_sum = 0.
    block_numel = 0.
    block_size = 4
    total_compression_byte = 0.
    for n, m in model.named_modules():
        if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
            sum, numel = m.quan_w_fn.calculate_block_sparsity(m.weight)
            num_nnz = sum * block_size
            byte_nnz = num_nnz * args.quan.weight.bit / 8
            # Co // block_size * 4 ( byte ) 
            indices = m.weight.shape[0] * 4 / block_size
            # col_indicies = num_nnz * 4 (byte) / block_size
            col_indicies = num_nnz * 4 / block_size

            compression_byte = byte_nnz + indices + col_indicies
            weight_numel += m.weight.numel()
            total_compression_byte += compression_byte
    weight_byte = weight_numel * 4
    return weight_byte / total_compression_byte
    

def main():
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(args, yaml_file)

    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
    t.backends.cudnn.benchmark = True
    t.backends.cudnn.deterministic = False

    model = create_model(args)
    
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    logger.info('Inserted quantizers into the original model')
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)
    model.to(args.device.type)
    start_epoch = 0
    if args.hard_pruning:
        resume_path = os.path.join(script_dir, args.resume.path)
        model, _, _ = util.load_checkpoint(
                model, resume_path, args.device.type, lean = args.resume.lean)
        for n, m in model.named_modules():
            if hasattr(m, "p"):
                m.p.requires_grad = False
    elif args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)
    
    print(calculate_compression_ratio(model, args))


if __name__ == "__main__":
    main()
