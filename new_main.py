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
    import wandb;
    wandb.log(pruning_log)

def main():
    set_seed(42)
    script_dir = Path.cwd()
    args = util.get_config(default_file=script_dir / 'config.yaml')
    wandb.init(reinit=True, name = args.name + str(args.quan.weight.ste), project="SLSQ")
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
    
    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    model = create_model(args)
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    process.initialize_parameter(train_loader, model, args)

    #tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)
    model.to(args.device.type)
    start_epoch = 0
    if args.eval:
        criterion = t.nn.CrossEntropyLoss().to(args.device.type)
        process.validate(test_loader, model, criterion, -1, monitors, args)
    else: # training
        # soft pruning
        for n, m in model.named_modules():
            if hasattr(m, "c") and hasattr(m, "p"):
                m.p.requires_grad = True
                m.hard_pruning = False
        criterion = t.nn.CrossEntropyLoss().to(args.device.type)

        #optimizer = t.optim.AdamW(model.parameters(), lr=args.optimizer.learning_rate, weight_decay = args.optimizer.weight_decay)
            
        #t.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.)
        optimizer = t.optim.SGD(model.parameters(),
                                lr=args.optimizer.learning_rate,
                                momentum=args.optimizer.momentum,
                                weight_decay=args.optimizer.weight_decay)
        lr_scheduler = util.lr_scheduler(optimizer,
                                         batch_size=train_loader.batch_size,
                                         num_samples=len(train_loader.sampler),
                                         **args.lr_scheduler)
        logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
        logger.info('LR scheduler: %s\n' % lr_scheduler)
        
        perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss, masking_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args)
            v_top1, v_top5, v_loss,sparsity = process.validate(val_loader, model, criterion, epoch, monitors, args)
            log_data = {"t_top1" : t_top1, "t_top5" : t_top5, "t_loss" : t_loss, "v_top1" : v_top1, \
                    "v_top5" : v_top5, "v_loss" : v_loss, "sparsity" : sparsity, "masking_loss" : masking_loss}
            wandb.log(log_data)
            pruning_log(model)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch, sparsity)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
            with t.no_grad():
                hard_sparsity = 0.
                total_zero = 0.
                total_numel = 0.
                for n,m in model.named_modules():
                    if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
                        if hasattr(m.quan_w_fn, "hard_pruning"):
                            m.quan_w_fn.hard_pruning = True
                        weight_zero = (m.quan_w_fn(m.weight.detach())[0] == 0).sum()
                        weight_numel = m.weight.detach().numel()
                        total_zero += weight_zero
                        total_numel += weight_numel
                        if hasattr(m.quan_w_fn, "hard_pruning"):
                            m.quan_w_fn.hard_pruning = False
                hard_sparsity = total_zero / total_numel
                wandb.log({"hard_pruning_sparsity": hard_sparsity})

        logger.info('>>>>>>>> Hard Pruning Mode')
        
        model = model.module.to("cpu")
        print(model)
        if args.device.gpu and not args.dataloader.serialized:
            model = t.nn.DataParallel(model, device_ids=args.device.gpu)
        print(model)
        model.to(args.device.type)
        # hard_pruning
        criterion = t.nn.CrossEntropyLoss().to(args.device.type)

        #optimizer = t.optim.AdamW(model.parameters(), lr=args.optimizer.learning_rate, weight_decay = args.optimizer.weight_decay )
            
        #t.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.)
        optimizer = t.optim.SGD(model.parameters(),
                                lr=args.optimizer.learning_rate,
                                momentum=args.optimizer.momentum,
                                weight_decay=args.optimizer.weight_decay)
        lr_scheduler = util.lr_scheduler(optimizer,
                                         batch_size=train_loader.batch_size,
                                         num_samples=len(train_loader.sampler),
                                         **args.lr_scheduler)
        logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
        logger.info('LR scheduler: %s\n' % lr_scheduler)
        perf_scoreboard = process.PerformanceScoreboard(args.log.num_best_scores)
        for epoch in range(0, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss, masking_loss = process.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, args, hard_pruning = True)
            v_top1, v_top5, v_loss,sparsity = process.validate(val_loader, model, criterion, epoch, monitors, args, hard_pruning = True)
            log_data = {"t_top1" : t_top1, "t_top5" : t_top5, "t_loss" : t_loss, "v_top1" : v_top1, \
                    "v_top5" : v_top5, "v_loss" : v_loss, "sparsity" : sparsity, "masking_loss" : masking_loss}
            wandb.log(log_data)
            pruning_log(model)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch, sparsity)
            is_best = perf_scoreboard.is_best(epoch)
            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
            if is_best:
                output_dir = script_dir / 'hard_pruned_model' 
                output_dir.mkdir(exist_ok=True)
                save_dir = os.path.join(output_dir)
                util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, True, args.name,  save_dir)
            
            with t.no_grad():
                hard_sparsity = 0.
                total_zero = 0.
                total_numel = 0.
                for n,m in model.named_modules():
                    if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
                        if hasattr(m.quan_w_fn, "hard_pruning"):
                            m.quan_w_fn.hard_pruning = True
                        weight_zero = (m.quan_w_fn(m.weight.detach())[0] == 0).float().sum()
                        weight_numel = m.weight.detach().numel()
                        total_zero += weight_zero
                        total_numel += weight_numel
                        if hasattr(m.quan_w_fn, "hard_pruning"):
                            m.quan_w_fn.hard_pruning = True
                hard_sparsity = total_zero / total_numel
                wandb.log({"hard_pruning_sparsity": hard_sparsity})

        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        process.validate(test_loader, model, criterion, -1, monitors, args, hard_pruning = True)

if __name__ == "__main__":
    main()
