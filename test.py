from typing import *

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from trainer import BasicTrainer

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from model.base_par import DepParser
from data_helper import Dependency, ConllDataset, load_annoted, InterDataset, load_codt_signal, DialogDataset
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
    parser.add_argument('--res_dir', type=str, default='results_v0')
    parser.add_argument('--data_file', type=str, default='./data/test.json')
    args = parser.parse_args()
    return args

CFG = parse_args()
seed_everything(CFG.random_seed)  # 设置随机种子

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(filename=f"./{CFG.res_dir}/test.log", mode='a')
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

# test1
# test_dataset = DialogDataset(CFG, data_file=CFG.data_file, data_ids=list(range(800)))
# test_dataloader = DataLoader(test_dataset, batch_size=CFG.batch_size)


model = DepParser(CFG)  # 创建依存解析器模型
# 加载Best Checkpoint
local_best_model_path = f'./{CFG.res_dir}/{CFG.plm.split("/")[-1].replace("-","_")}_base_par_new.pt'
print(f"load model checkpoint file path:{local_best_model_path}")
model.load_state_dict(torch.load(local_best_model_path))
model = model.cuda()
model.eval()

print("start inference")

# 存储arc，rel的标签和模型输出
head_whole, rel_whole, mask_whole = torch.Tensor(), torch.Tensor(), torch.Tensor()
arc_logit_whole, rel_logit_whole = torch.Tensor(), torch.Tensor()
# 平均loss
avg_loss = 0.0
# for step, batch in enumerate(test_dataloader):
for step, batch in enumerate(tqdm(test_dataloader, desc="Inference Progress")):
    inputs, offsets, heads, rels, masks = batch
        # - inputs: 输入数据列表，包含了编码后的对话输入
        # - offsets: 偏移量列表，记录了输入中单词的起始位置
        # - heads: 头部索引列表，记录了依存关系中的头部单词索引
        # - rels: 关系列表，记录了依存关系类型的编码
        # - masks: 掩码列表，标记了有效的输入位置
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
    
    # print('arc_logit.size ', arc_logits.size())
        # 模型推理出的弧的结果
            # 弧(Arc)预测：预测每个词与句子中其他词之间的依存关系，即句法树中的边。
            # 形状为 (batch_size, seq_len, seq_len)
            # arc_logit[i, j, k] 表示第 i 个句子中第 k 个词作为第 j 个词的父节点的概率得分. 是否存在依存关系
    # print(arc_logits[0][0])
        # 模型推理出的关系的结果
            # 关系(Relation)预测: 对于每一条弧，预测其所表示的依存关系类型。
            # 形状为 (batch_size, seq_len, seq_len, num_rels)
            # rel_logit_cond[i, j, k, m] 第i个句子中第j个词和第k个词之间所有关系的得分
    # print('rel_logits.size ', rel_logits.size())
#         arc_logit.size  torch.Size([256, 160, 160])
# tensor([  3.1154, -18.2024,  -4.5325, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024,
#         -18.2024, -18.2024, -18.2024, -18.2024, -18.2024, -18.2024],
#        device='cuda:0')
# rel_logits.size  torch.Size([256, 160, 40])
# tensor([ 13.2253, -13.7392,  -5.7025,  -8.5131,  -5.4425, -12.9642,  -2.5700,
#         -13.0239,  -2.1520,  -5.1711,  -8.3395,  -4.0683,  -8.1109, -13.3134,
#         -11.1747,  -8.5684,  -5.5008, -10.2341,  -2.2665, -14.1649, -12.8969,
#         -13.1050, -15.6647, -14.5756, -14.9270, -13.4337, -14.6572,  -7.9551,
#         -16.7591, -16.7550, -18.7497, -14.5212, -16.0033, -17.0037, -17.1893,
#         -17.2901, -16.4864, -14.5896, -12.2550, -13.5884], device='cuda:0')
    # print(rel_logits[0][0])

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

    # 计算arc和rel的综合loss
    loss = arc_rel_loss(arc_logits, rel_logits, heads, rels, masks)

    # 将batch的arc和rel模型输出拼接起来统一计算
    arc_logit_whole = torch.cat([arc_logit_whole, arc_logits.cpu()], dim=0)
    rel_logit_whole = torch.cat([rel_logit_whole, rel_logits.cpu()], dim=0)

    # print('arc_logit_whole', arc_logit_whole.size())
    # print('rel_logit_whole', rel_logit_whole.size())

    # 同上，将batch的arc和rel的Label拼接起来统一计算
    head_whole, rel_whole = torch.cat([head_whole, heads.cpu()], dim=0), torch.cat([rel_whole, rels.cpu()], dim=0)
    # print('head_whole', head_whole.size())
    # print('rel_whole', rel_whole.size())
    # head_whole torch.Size([1024, 160])
    # rel_whole torch.Size([1024, 160])
    mask_whole = torch.cat([mask_whole, masks.cpu()], dim=0)
    
    # 预估每个样本的loss和
    avg_loss += loss.item() * len(heads)  # times the batch size of data

# 计算相关指标
metrics = uas_las(arc_logit_whole, rel_logit_whole, head_whole, rel_whole, mask_whole)
print('metrics 0', metrics)
avg_loss /= len(test_dataloader.dataset)
results = [CFG.plm, round(avg_loss,4), metrics['UAS'], metrics['LAS']]
results = [str(x) for x in results]
with open(save_file, "a+") as f:
    f.write(",".join(results) + "\n")

# ####################################################################

metrics = inner_inter_uas_las(arc_logit_whole, rel_logit_whole, head_whole, rel_whole,
                  mask_whole)
print('metrics 1', metrics)

inter_uas, inner_uas,inter_las,inner_las = metrics['Inter-UAS'], metrics['Inner-UAS'],metrics['Inter-LAS'], metrics['Inner-LAS']

avg_loss /= len(test_dataloader.dataset)

logger.info("--Evaluation:")
logger.info("Avg Loss: {}  Inter-UAS: {}  Inter-LAS: {} \Inner-UAS: {}  Inner-LAS: {} n".format(avg_loss, inter_uas, inter_las, inner_uas, inner_las))

results = [CFG.plm, round(avg_loss,4), inter_uas, inter_las, inner_uas, inner_las]
results = [str(x) for x in results]
with open(save_file, "a+") as f:
    f.write(",".join(results) + "\n")