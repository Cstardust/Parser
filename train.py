from typing import *
from itertools import chain
import logging
import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.base_par import DepParser
from data_helper import Dependency, ConllDataset
from trainer import BasicTrainer
from utils import arc_rel_loss, uas_las, seed_everything
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./data/train.conll')
    parser.add_argument('--val_file', type=str, default='./data/val.conll')
    parser.add_argument('--plm', type=str, default='./plm/chinese-electra-180g-base-discriminator')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--plm_lr', type=float, default=2e-5)
    parser.add_argument('--head_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.33)   # 0.2
    parser.add_argument('--grad_clip', type=float, default=3)    # 2
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--num_early_stop', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=160)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--num_labels', type=int, default=40)
    parser.add_argument('--print_every_ratio', type=float, default=0.5)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--res_dir', type=str, default='results_v0')
    parser.add_argument('--eval_strategy', type=str, default='epoch')   # step
    parser.add_argument('--eval_every', type=int, default=800)
    args = parser.parse_args()
    return args

def load_conll(data_file: str, train_mode=True):
    sentence: List[Dependency] = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[8], toks[9])
                sentence.append(dep)


def load_conll_with_aug(data_file: str, train_mode=True):
    sentence: List[Dependency] = []
    f1 = open(data_file, 'r', encoding='utf-8')
    f2 = open('./aug/codt/codt_train_fixed.conll', 'r', encoding='utf-8')
    for line in chain(f1.readlines(), f2.readlines()):
        toks = line.split()
        if len(toks) == 0:
            yield sentence
            sentence = []
        elif len(toks) == 10:
            if toks[8] != '_':
                dep = Dependency(toks[0], toks[1], toks[8], toks[9])
            else:
                dep = Dependency(toks[0], toks[1], toks[6], toks[7])
            sentence.append(dep)

    f1.close()
    f2.close()

if __name__ == '__main__':

    CFG = parse_args()

    if not os.path.exists(CFG.res_dir):
        os.makedirs(CFG.res_dir)

    print(CFG.res_dir)

    pt_output_file = f'./{CFG.res_dir}/{CFG.plm.split("/")[-1].replace("-","_")}_base_par_new.pt'

    print(pt_output_file)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=f"./{CFG.res_dir}/train.log", mode='a')
    logger.addHandler(fh)

    time_now = datetime.datetime.now().isoformat()
    print(time_now)
    logger.info(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')

    seed_everything(CFG.random_seed)

    if torch.cuda.is_available() and CFG.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
    CFG.__dict__["tokenizer"] = tokenizer


    train_iter = None
    val_iter = None

    print('train_file', CFG.train_file)
    train_dataset = ConllDataset(CFG, fname=CFG.train_file, load_fn=load_conll_with_aug, train=True, logger=logger)
    train_iter = DataLoader(train_dataset, batch_size=CFG.batch_size)

    print('val_file', CFG.val_file)
    val_dataset = ConllDataset(CFG, fname=CFG.val_file, load_fn=load_conll_with_aug, train=True, logger=logger)
    val_iter = DataLoader(val_dataset, batch_size=CFG.batch_size)

    model = DepParser(CFG)
    model = model.cuda()

    trainer = BasicTrainer(model=model,
                           trainset_size=len(train_dataset),
                           loss_fn=arc_rel_loss,
                           metrics_fn=uas_las,
                           logger=logger,
                           config=CFG)

    best_res, best_state_dict = trainer.train(model=model,
                                              train_iter=train_iter,
                                              val_iter=val_iter)
    print(f'====================={time_now} best_res {best_res}=====================')
    logger.info(f'{time_now} best_res {best_res}')
    time_now = datetime.datetime.now().isoformat()

    pt_output_file = f'./{CFG.res_dir}/{CFG.plm.split("/")[-1].replace("-","_")}_base_par_new.pt'
    torch.save(best_state_dict, pt_output_file)

    sys.exit()