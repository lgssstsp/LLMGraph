from collections import defaultdict
import torch
import time
import logging
import gc
import numpy as np

from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt

def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        if cfg.model.eval_type == 'non-ranking':
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            logger.update_stats(true=true.detach().cpu(),
                                pred=pred_score.detach().cpu(),
                                loss=loss.item(),
                                lr=0,
                                time_used=time.time() - time_start,
                                params=cfg.params) 

def train(loggers, loaders, model, optimizer, scheduler, exp_num=0):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
        # print('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))
        # print('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        # logging.info(f'[Exp{exp_num}]')
        logging.info(f'[Experiment]')
        # print(f'[Experiment]')
        loggers[0].write_epoch(cur_epoch)

        # print("print loggers[0]",loggers[0])

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
                # print("print loggers[0]",loggers[0])

        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
    # print('Task done, results saved in {}'.format(cfg.out_dir))

    # Collect unused memory
    del model
    for loader in loaders:
        for batch in loader:
            del batch
    gc.collect()
    torch.cuda.empty_cache()


    

