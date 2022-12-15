import logging
import math
import operator
import time

import torch as t

from util import AverageMeter

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    masking = AverageMeter()

    model.train()
    
    regularizer = not args.hard_pruning
    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.init_mode:
            for n, m in model.named_modules():
                if hasattr(m, "init_mode"):
                    m.init_mode = True
                    print(m)
            args.init_mode = False
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))
        if lr_scheduler is not None:
            lr_scheduler.step(epoch, batch_idx)

        optimizer.zero_grad()

        masking_loss_list = []
        if regularizer:
            for n, m in model.named_modules():
                if hasattr(m, "soft_mask") and m.soft_mask is not None:
                    masking_loss_list.append(m.soft_mask.mean())
            masking_loss = t.stack(masking_loss_list)
            masking_loss = (masking_loss * masking_loss).mean()
            #masking_loss = t.linalg.norm(t.stack(masking_loss_list), dim =0, ord = 2)
            masking_loss = masking_loss * args.lamb
            loss += masking_loss
            masking.update(masking_loss.item(), inputs.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr'],
                })   
    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg, masking.avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    total_zero = 0.
    total_numel = 0.
    sparsity_log = {}
    block_numel = 0.
    block_sum = 0.
    for n, m in model.named_modules():
        if hasattr(m, "quan_w_fn") and hasattr(m.quan_w_fn, "p"):
            m.quan_w_fn.hard_pruning = True
            weight_zero = (m.quan_w_fn(m.weight.detach())==0).sum()
            weight_numel = m.weight.detach().numel()
            sparsity = weight_zero / weight_numel
            total_zero += weight_zero
            total_numel += weight_numel
            sparsity_log[n + "sparsity"] = sparsity
            m.quan_w_fn.hard_pruning = args.hard_pruning
            sum, numel = m.quan_w_fn.calculate_block_sparsity(m.weight)
            block_sum += sum
            block_numel += numel
        
    sparsity_log["block_sparsity"] = 1. - block_sum / block_numel
    print(block_sum, block_numel)
    import wandb; wandb.log(sparsity_log)
    sparsity = total_zero / total_numel
    return top1.avg, top5.avg, losses.avg, sparsity


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch, sparsity):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch, 'sparsity' : sparsity})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f] Sparsity : %.3f',
                        idx + 1, score['epoch'], score['top1'], score['top5'], score['sparsity'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
