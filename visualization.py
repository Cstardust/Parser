from typing import *

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from trainer import BasicTrainer

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from model.base_par import DepParser
from data_helper import Dependency, ConllDataset, load_annoted, DialogDataset, tensor2dep
from utils import uas_las, seed_everything, to_cuda, arc_rel_loss, inner_inter_uas_las
import logging, datetime
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_file', type=str, default='./data/use_test.conll')
    parser.add_argument('--plm', type=str, default='./plm/bge-large-zh')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_length', type=int, default=160)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--num_labels', type=int, default=40)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--res_dir', type=str, default='visual')
    parser.add_argument('--data_file', type=str, default='./data/test.json')
    args = parser.parse_args()
    return args

CFG = parse_args()
seed_everything(CFG.random_seed)  # 设置随机种子

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename=f"./{CFG.res_dir}/try-03.txt", mode='a')
logger.addHandler(fh)
time_now = datetime.datetime.now().isoformat()
logger.info(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')
save_file = "./{}/test_save.txt".format(CFG.res_dir)

print(save_file)
if not os.path.exists(save_file):
    with open(save_file, 'a'):
        os.utime(save_file, None)

tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
CFG.tokenizer = tokenizer

# test0
# 加载test数据
def load_conll(data_file: str, train_mode=False):
    sentence: List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as fi_data:
        for line in fi_data.readlines():
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


test_dataset = ConllDataset(CFG, load_fn=load_conll, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)


words = test_dataset.get_words
for idx, line in enumerate(words):
    logger.info("wwords &&{}: {}&&".format(idx, line))
    if idx >= 5:
        break

# test1
# test_dataset = DialogDataset(CFG, data_file=CFG.data_file, data_ids=list(range(800)))
# test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)


print("start inference")

# 存储arc，rel的标签和模型输出
head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
# 平均loss
avg_loss = 0.0
offset_len = 0

total_batches = len(test_dataloader)
logger.info(f"total batches: {total_batches}")
for step, batch in enumerate(tqdm(test_dataloader, desc="Inference Progress")):
    # print('step ', step)
    # inputs, offsets, heads, rels, masks, words, deps = batch
    inputs, offsets, heads, rels, masks = batch
    
    # print('heads', type(heads), len(heads))
    # print('\thead 0', heads[0].size())
    # print('rels', type(rels), len(rels))
    # print('\trel 0', rels[0].size())
    # print('masks', type(masks), len(masks))
    # print('\tmask 0', masks[0].size())

    if step == 0 or step == 1 or step == total_batches - 1:
        logger.info(f"**************************step {step} {offset_len} visual*********************")
        dep_llist = tensor2dep(heads, rels, words[offset_len:], masks)
        # dep_llist 若干句子的依存关系
        logger.info('dep_llst len {}'.format(len(dep_llist)))
        for idx, dep_list in enumerate(dep_llist):
            # 一个句子的依存关系: dep_list
            logger.info('dep_lst len {}'.format(len(dep_list)))
            for jdx, dep in enumerate(dep_list):
                logger.info('dep visual {} {}'.format(jdx, dep.repr()))
        logger.info(f"**************************step {step} {offset_len} visual*********************")

    offset_len += len(heads)

        # - inputs: 输入数据列表，包含了编码后的对话输入
        # - offsets: 偏移量列表，记录了输入中单词的起始位置
        # - heads: 头部索引列表，记录了依存关系中的头部单词索引
        # - rels: 关系列表，记录了依存关系类型的编码
        # - masks: 掩码列表，标记了有效的输入位置

    # print('heads ', heads.size())
    # heads  torch.Size([256, 160])
    # print(heads[0])
    
    # print('rels ', rels.size())
    # rels  torch.Size([256, 160])
    # print(rels[0])
# heads  torch.Size([256, 160])
# tensor([0, 6, 6, 6, 6, 4, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
# rels  torch.Size([256, 160])
# tensor([ 0,  9,  9,  9,  9, 12,  0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
#        device='cuda:0')

    # 同上，将batch的arc和rel的Label拼接起来统一计算
    head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
    # print('head_whole', head_whole.size())
    # print('rel_whole', rel_whole.size())
    # head_whole torch.Size([1024, 160])
    # rel_whole torch.Size([1024, 160])
    mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)

# 计算相关指标
# arc_logits_ed / heads torch.Size([4241, 160])    # [i,j] 第i批, 以j词作为尾词. 其父节点的位置
# rel_logits_ed / rels  torch.Size([4241, 160])    # [i,j] 第i批, 以j词作为尾词. 其依存弧的标签