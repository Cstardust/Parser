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
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_file', type=str, default='./data/use_test.conll')
    parser.add_argument('--plm', type=str, default='./plm/chinese-electra-180g-base-discriminator')
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
    parser.add_argument('--format', type=str, default='json')
    parser.add_argument('--visual', type=bool, default=False)
    args = parser.parse_args()
    return args

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

### CONFIG ###
CFG = parse_args()
seed_everything(CFG.random_seed)  # 设置随机种子
### CONFIG ###

### LOGGER ###
logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename=f"./{CFG.res_dir}/try-08.txt", mode='a')
logger.addHandler(fh)
time_now = datetime.datetime.now().isoformat()
logger.info(f'=-=-=-=-=-=-=-=-={time_now}=-=-=-=-=-=-=-=-=-=')
save_file = "./{}/test_save.txt".format(CFG.res_dir)
print(save_file)
### LOGGER ###

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f'Using device: {device}')


if not os.path.exists(save_file):
    with open(save_file, 'a'):
        os.utime(save_file, None)

tokenizer = AutoTokenizer.from_pretrained(CFG.plm)
CFG.tokenizer = tokenizer

if CFG.format == 'conll':
    test_dataset = ConllDataset(CFG, load_fn=load_conll, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)
else:
    test_dataset = DialogDataset(CFG, data_file=CFG.data_file, data_ids=list(range(800)))
    test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)

model = DepParser(CFG)  # 创建依存解析器模型
# 加载Best Checkpoint
local_best_model_path = f'./{CFG.res_dir}/{CFG.plm.split("/")[-1].replace("-","_")}_base_par_new.pt'
print(f"load model checkpoint file path:{local_best_model_path}")
model.load_state_dict(torch.load(local_best_model_path))
model = model.cuda()
model.eval()


words = test_dataset.get_words
for idx, line in enumerate(words):
    logger.info("wwords &&{}: {}&&".format(idx, line))
    if idx >= 5:
        break

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
    if torch.cuda.is_available():
        inputs_cuda = {}
        for key, value in inputs.items():
            inputs_cuda[key] = value.cuda()
        inputs = inputs_cuda

        offsets, heads, rels, masks = to_cuda(data=(offsets, heads, rels, masks))

    with torch.no_grad():
        arc_logits, rel_logits = model(inputs,
                                       offsets,
                                       heads,
                                       rels,
                                       masks,
                                       evaluate=True)
    # print('heads', type(heads), len(heads))
    # print('\thead 0', heads[0].size())
    # print('rels', type(rels), len(rels))
    # print('\trel 0', rels[0].size())
    # print('masks', type(masks), len(masks))
    # print('\tmask 0', masks[0].size())
    
    # 计算arc和rel的综合loss
    loss = arc_rel_loss(arc_logits, rel_logits, heads, rels, masks)

    # 将batch的arc和rel模型输出拼接起来统一计算
    arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
    rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)

    # print('arc_logit_whole', arc_logit_whole.size())
    # print('rel_logit_whole', rel_logit_whole.size())

    # 同上，将batch的arc和rel的Label拼接起来统一计算
    head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
    mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)

    # 预估每个样本的loss和
    avg_loss += loss.item() * len(heads)  # times the batch size of data

    offset_len += len(heads)


# 计算相关指标
# arc_logits_ed / heads torch.Size([4241, 160])    # [i,j] 第i批, 以j词作为尾词. 其父节点的位置
# rel_logits_ed / rels  torch.Size([4241, 160])    # [i,j] 第i批, 以j词作为尾词. 其依存弧的标签
metrics, arc_logits_ed, rel_logits_ed = uas_las(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
print('metrics 0', metrics)
avg_loss /= len(test_dataloader.dataset)
results = [CFG.plm, round(avg_loss,4), metrics['UAS'], metrics['LAS']]
results = [str(x) for x in results]
with open(save_file, "a+") as f:
    f.write(",".join(results) + "\n")

print('offset len', offset_len)
print('head_whole', type(head_whole), len(head_whole))

if CFG.visual:
    logger.info(f"**************************Inference {offset_len} visual*********************")
    dep_llist = tensor2dep(head_whole, rel_whole, words, mask_whole)
    # dep_llist 若干句子的依存关系
    logger.info('dep_llst len {}'.format(len(dep_llist)))
    for idx, dep_list in enumerate(dep_llist):
        # 一个句子的依存关系: dep_list
        logger.info('dep_lst len {}'.format(len(dep_list)))
        for jdx, dep in enumerate(dep_list):
            logger.info('dep visual {} {}'.format(jdx, dep.repr()))
    logger.info(f"**************************Inference {offset_len} visual*********************")

sys.exit()

# metrics, _, _ = inner_inter_uas_las(arc_logit_whole, rel_logit_whole, head_whole, rel_whole,
#                   mask_whole)
# print('metrics 1', metrics)

# inter_uas, inner_uas,inter_las,inner_las = metrics['Inter-UAS'], metrics['Inner-UAS'],metrics['Inter-LAS'], metrics['Inner-LAS']

# avg_loss /= len(test_dataloader.dataset)

# logger.info("--Evaluation:")
# logger.info("Avg Loss: {}  Inter-UAS: {}  Inter-LAS: {} \Inner-UAS: {}  Inner-LAS: {} n".format(avg_loss, inter_uas, inter_las, inner_uas, inner_las))

# results = [CFG.plm, round(avg_loss,4), inter_uas, inter_las, inner_uas, inner_las]
# results = [str(x) for x in results]
# with open(save_file, "a+") as f:
#     f.write(",".join(results) + "\n")