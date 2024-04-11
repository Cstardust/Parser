
from typing import *
import json
from constant import rel_dct, rel2id

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

# Dependency: {尾部单词索引, 单词内容, 头部单词索引, 关系类型}           
class Dependency():
    def __init__(self, idx, word, head, rel):
        self.id = idx
        self.word = word
        self.head = head
        self.rel = rel

    def __str__(self):
        # example:  1	上海	_	NR	NR	_	2	nn	_	_
        values = [str(self.idx), self.word, "_", "_", "_", "_", str(self.head), self.rel, "_", "_"]
        return '\t'.join(values)

    def __repr__(self):
        return f"({self.word}, {self.head}, {self.rel})"

    
def load_codt(data_file: str):
    sentence:List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[3], toks[6], toks[7])
                sentence.append(dep)

def load_codt_new(data_file: str):
    sentence:List[Dependency] = []

    with open(data_file, 'r', encoding='utf-8') as f:
        # data example: 1	上海	_	NR	NR	_	2	nn	_	_
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:
                dep = Dependency(toks[0], toks[1], toks[3], toks[8], toks[9])
                sentence.append(dep)

# 获取所有回合对话的依存关系列表 sample_list[list(Dependency)] 
# 其元素list(Dependency)是每个回合的依存关系列表
# Dependency: {尾部单词索引, 单词内容, 头部单词索引, 关系类型}           
def load_annoted(data_file, data_ids=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    sample_lst:List[List[Dependency]] = []
    
    # 每次遍历一个回合的对话
    for i, d in enumerate(data):
    # i: 0, 
    # d: {'dialog': [{'turn': 1, 'utterance': 'Hello!'}, {'turn': 2, 'utterance': 'How are you?'}], 'relationship': [['1-1', 'subj', '2-2'], ['2-1', 'obj', '1-2']]}
        # 如果第i条不是目标样本（不在之前规定好的在data_ids中）
        if data_ids is not None and i not in data_ids:
            continue
        # 用于存储当前对话的依存关系
        # 二层字典
            # 外层字典 rel_dct 的键是 head_uttr_idx(头部单词所在的回合索引)，对应的值是一个字典。
            # 内层字典的键是 tail_word_idx（尾部单词所在的位置索引），对应的值是一个列表，包含 [head_word_idx, rel]，表示头部单词所在的位置索引和关系类型。
        rel_dct = {}
        for tripple in d['relationship']:
            # head 头部单词在对话中的位置
            # rel  头部单词和尾部单词之间的关系
            # tail 尾部单词在对话中的位置
            head, rel, tail = tripple
            # 解析头部和尾部的单词索引
            # head.split('-')：将字符串 head 按照连字符 "-" 进行分割，得到一个包含两个字符串元素的列表，例如 ['1', '2']。
            # [int(x) for x in ...] 将字符串列表中的每个元素都转化成整数
            # head_uttr_idx, head_word_idx: 头部单词所在的回合索引, 和单词在该回合的位置索引
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            # 同理, 尾部单词位置.
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            # 如果头部和尾部不在同一个句子中，则跳过当前依存关系
            if head_uttr_idx != tail_uttr_idx:
                continue
            # 将当前依存关系添加到关系字典中
            # 依存关系按照头部单词的索引和尾部单词的索引组织起来
                # rel_dct.get(head_uttr_idx, None): 如果rel_dct中不存在rel_dct key, 则返回none
            if not rel_dct.get(head_uttr_idx, None):
                rel_dct[head_uttr_idx] = {tail_word_idx: [head_word_idx, rel]}
            else:
                rel_dct[head_uttr_idx][tail_word_idx] = [head_word_idx, rel]
            
        for item in d['dialog']:
            turn = item['turn']     # 第几轮对话
            utterance = item['utterance']   # 对话内容
            # dep_lst:List[Dependency] = [Dependency(0, '[root]', -1, '_')]
            dep_lst:List[Dependency] = []   # 存储当前回合单词之间的的依存关系列表

            # 遍历当前对话内容的每个单词
            for word_idx, word in enumerate(utterance.split(' ')):
                # word是作为依存关系中的尾部单词
                # 获取当前单词在依存关系中的头部单词索引和关系类型
                    # 如果字典中不存在 word_idx + 1 为键的项, 则返回默认值 [word_idx, 'adjct']
                    # 其中 word_idx 是当前单词在对话中的位置, 'adjct' 是默认的关系类型。
                    # 这个默认值的作用是处理一些在依存关系中未被标注的单词，使得代码能够正常运行并给出一个默认的依存关系。
                # head_word_id: 当前单词在依存关系中的头部单词的位置索引
                # rel: 词间的关系
                                                    # rel_dict中的单词位置是从1开始计数的，遍历时的word_idx是从0开始计数的，所以需要将 word_idx 加1以匹配字典中的键。
                head_word_idx, rel = rel_dct[turn].get(word_idx + 1, [word_idx, 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                # Dependency: {尾部单词索引, 单词内容, 头部单词索引, 关系类型}
                # print(Dependency(word_idx + 1, word, head_word_idx, rel))
                # 当前单词所在的依存关系Dependency存入本回合的依存关系列表dep_lst
                dep_lst.append(Dependency(word_idx + 1, word, head_word_idx, rel))  # start from 1
            # 将当前回合的依存关系列表Depdency list存入sample_lst
            sample_lst.append(dep_lst)
    
    # 返回整个对话的依存关系列表
    return sample_lst


class DialogDataset(Dataset):
    def __init__(self, cfg, data_file, data_ids):
        """
        初始化函数，创建 DialogDataset 类的实例。

        参数：
            - cfg: 配置对象，包含了模型和数据处理的相关参数
            - data_file: 数据文件路径，存储了对话数据
            - data_ids: 数据索引列表，指定需要读取的数据样本的索引
        """
        self.cfg = cfg
        self.data_file = data_file
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data(data_ids)
        
    def read_data(self, data_ids):
        """
        读取对话数据并进行预处理，返回处理后的输入、偏移量、头部索引、关系和掩码。

        参数：
            - data_ids: 数据索引列表，指定需要读取的数据样本的索引

        返回：
            - inputs: 输入数据列表，包含了编码后的对话输入
            - offsets: 偏移量列表，记录了输入中单词的起始位置
            - heads: 头部索引列表，记录了依存关系中的头部单词索引
            - rels: 关系列表，记录了依存关系类型的编码
            - masks: 掩码列表，标记了有效的输入位置
        """
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []

        # 遍历指定索引的数据样本. (json被处理成了Dependency列表)
            # data_file: test.json / train_50.json
            # tqdm: 用于显示for循环进度
            # deps: list{dependency, dependency, dependency...}
        for deps in tqdm(load_annoted(self.data_file, data_ids)):
            # another sentence
            # 当前回合对话的依存列表长度
            seq_len = len(deps)

            word_lst = [] # 空列表，用于存储当前处理的句子中的单词
            # 长度为 cfg.max_length 的一维数组
                # 头部单词索引
            head_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)  # same as root index is 0, constrainting by mask 
                # 依存关系（rel转化成int）
            rel_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
                # 掩码信息（依存关系是否有效）
            mask_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            # 遍历本回合对话的所有dependency
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== self.cfg.max_length:
                    break
                # 当前遍历的单词添加到word list
                word_lst.append(dep.word)
                # 如果头部单词索引无效, 那么置mask为无效
                if dep.head == -1 or dep.head + 1 >= self.cfg.max_length:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                # 保存头部单词索引, 置mask为有效1
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1
                
                # 将dep.relationship(依存关系)转化为int保存下来
                    # rel2id : {'root': 0, 'sasubj-obj': 1, 'sasubj': 2, 'dfsubj': 3, 'subj': 4, 'subj-in': 5, 'obj': 6, 'pred': 7, 'att': 8, 'adv': 9, 'cmp': 10, 'coo': 11, 'pobj': 12, 'iobj': 13, 'de': 14, 'adjct': 15, 'app': 16, 'exp': 17, 'punc': 18, 'frag': 19, 'repet': 20, 'attr': 21, 'bckg': 22, 'cause': 23, 'comp': 24, 'cond': 25, 'cont': 26, 'elbr': 27, 'enbm': 28, 'eval': 29, 'expl': 30, 'joint': 31, 'manner': 32, 'rstm': 33, 'temp': 34, 'tp-chg': 35, 'prob-sol': 36, 'qst-ans': 37, 'stm-rsp': 38, 'req-proc': 39}
                rel_tokens[i+1] = rel2id.get(dep.rel, 0)    # 如果关系不存在, 则置0
            
            # 对当前回合对话的所有单词列表进行编码
            tokenized = self.cfg.tokenizer.encode_plus(word_lst, 
                                              # 一律补pad到max length长度
                                              padding='max_length',
                                              # 当句子长度大于max length时,截断 
                                              truncation=True,
                                              max_length=self.cfg.max_length, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            # 编码后的结果填入inputs list
            # 为啥叫做input? 是指这个就是最终输入给模型的input吗? 还会经过pretrained的一层编码
            inputs.append({"input_ids": tokenized['input_ids'][0],          # 输入token的ID. 就是这一个list得单词的编码
                          "token_type_ids": tokenized['token_type_ids'][0], # token的类型ID
                           "attention_mask": tokenized['attention_mask'][0] # 注意力掩码
                          })
            # Tokenized: {'input_ids': tensor([[ 101,  872, 1962, 1086, 6224, 1103, 3299, 1305,  102,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]), 'offset_mapping': tensor([[[0, 0], [0, 1],
            #      [1, 2],
            #      [0, 1],
            #      [1, 2],
            #      [0, 1],
            #      [1, 2],
            #      [2, 3],
            #      [0, 0],
            #      [0, 0]]])}
            
            # 初始化一个空列表，用于存储每个单词在编码后序列中的位置索引
            sentence_word_idx = []
            # 遍历编码后序列的偏移量列表，从第一个单词开始（索引1），并获取每个单词的起始和结束位置
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                # 检查当前标记的起始字符偏移量是否为0, 并且结束字符偏移量不为0. 这意味着当前标记是一个词汇的起始标记, 因为它的起始字符偏移量为0, 但不是句子的起始标记.
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
            """
            在 tokenized.offset_mapping 中，每个元素都是一个元组，表示编码后的序列中每个字符在原始文本中的起始和结束位置。
                start 表示编码后的序列中字符对应的原始文本中的起始位置（包括）。
                end 表示编码后的序列中字符对应的原始文本中的结束位置（不包括）。
            offset_mapping = [
                (0, 0),     # 特殊标记 [CLS] 对应的位置
                (0, 5),     # 单词 "Hello" 对应的位置
                (6, 7),     # 逗号 "," 对应的位置
                (8, 11),    # 单词 "how" 对应的位置
                (12, 15),   # 单词 "are" 对应的位置
                (16, 19),   # 单词 "you" 对应的位置
                (20, 20)    # 特殊标记 [SEP] 对应的位置
            ]
            """
            if len(sentence_word_idx) < self.cfg.max_length - 1:
                sentence_word_idx.extend([0]* (self.cfg.max_length - 1 - len(sentence_word_idx)))

            # 将计算得到的offset列表转换为PyTorch张量，并添加到offsets列表中
            offsets.append(torch.as_tensor(sentence_word_idx))

            # 保存头部单词索引, 关系, 有效掩码
            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
        
        # inputs: 输入的句子拆分成词, 并得到的所有词的编码：
            # [ {[input_ids],[token_type_ids],[attention_mask]}, {[input_ids],[token_type_ids],[attention_mask]}]
            # "input_ids": 对于word_lst的编码.
            # "token_type_ids": 第一个句子和特殊符号的位置是0. 第二个句子位置是1.
            # "attention_mask": pad的位置是0, 其他位置是1
        # offsets: 包含了每个句子中单词的偏移量. [[tensor],[tensor],[tensor]]
        # heads: 依存关系中的头部单词索引. [[tensor],[tensor],[tensor]]
        # rels: 以当前单词作为尾部词语的依赖关系类型. [[tensor],[tensor],[tensor]]
        # masks: 每个句子中每个单词的掩码信息。掩码信息用于指示模型在处理输入时哪些单词是有效的，哪些是填充的。这个列表记录了每个单词的掩码值. [[tensor],[tensor],[tensor]]
        return inputs, offsets, heads, rels, masks

    # 返回一篇章对话(上文的一回合)的词的依存关系
    # 一次train_iter迭代器会多次调用__getitem__, 将他们纵向拼接成一个矩阵
    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
#     inputs {'input_ids': tensor([ 101, 2644, 1962, 8024, 4867, 2644, 1921, 1921, 1962, 2552, 2658, 6435,
#         2644, 4924, 5023, 8024, 3633, 1762,  711, 2644, 4802, 6371, 3634, 1184,
#         1486, 6418, 1079, 2159,  511,  102,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0]), 'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
# offsets tensor([ 0,  1,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21,
#         23, 25, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
#          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])
# heads [ 0  2  4  2  0  8  8  8  4  4  9 12 10 12 17 17 15  9 19 20 17 20  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
# rels [ 0  4 27 18  0  8  9  8  6 27  6  9  7 18  9  9 12 27  9  8  6 18  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
#   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
# masks [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0]

    def __len__(self):
        return len(self.rels)


class ConllDataset(Dataset):
    def __init__(self, cfg, load_fn, train):
        self.train = train
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.load_fn = load_fn
        self.inputs, self.offsets, self.heads, self.rels, self.masks = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks = [], [], [], []
    
        file = self.cfg.train_file if self.train else self.cfg.dev_file
        for deps in tqdm(self.load_fn(file, self.train)):
            seq_len = len(deps)
            
            word_lst = [] 
            rel_attr = {'input_ids':torch.Tensor(), 'token_type_ids':torch.Tensor(), 'attention_mask':torch.Tensor()}
            head_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            mask_tokens = np.zeros(self.cfg.max_length, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== self.cfg.max_length:
                    break
                    
                word_lst.append(dep.word)
                    
                if dep.head in ['_', '-1'] or int(dep.head) + 1 >= self.cfg.max_length:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1

                if self.train:
                    rel_tokens[i+1] = rel2id[dep.rel]
                else:
                    rel_tokens[i+1] = rel2id.get(dep.rel, rel2id['adjct'])

            tokenized = self.tokenizer.encode_plus(word_lst, 
                                                padding='max_length', 
                                                truncation=True,
                                                max_length=self.cfg.max_length, 
                                                return_offsets_mapping=True, 
                                                return_tensors='pt',
                                                is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                            "token_type_ids": tokenized['token_type_ids'][0],
                            "attention_mask": tokenized['attention_mask'][0]
                            })

            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
            if len(sentence_word_idx) < self.cfg.max_length - 1:
                sentence_word_idx.extend([0]* (self.cfg.max_length - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))
            
            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)


        return inputs, offsets, heads, rels, masks


    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    def __len__(self):
        return len(self.rels)


def load_codt_signal(data_file: str, return_two=False):
    sentence:List = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            toks = line.split()
            if len(toks) == 0 and len(sentence) != 0:
                yield sentence
                sentence = []
            elif len(toks) == 10:                
                if return_two:
                    sentence.append([int(toks[2]), int(toks[3])])
                else:
                    sentence.append(int(toks[2]))


def load_inter(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    signal_iter = load_codt_signal('mlm_based/diag_test.conll', return_two=True)

    sample_lst:List[List[Dependency]] = []
    # for d, pred_signals in tqdm(zip(data, load_codt_signal('../prompt_based/diag_test.conll', idx=3))):
    for d in data:
        rel_dct = {}
        for tripple in d['relationship']:
            head, rel, tail = tripple
            head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
            tail_uttr_idx, tail_word_idx = [int(x) for x in tail.split('-')]
            
            if rel == 'root' and head_uttr_idx != 0: # ignore root
                continue
                 
            if not rel_dct.get(tail_uttr_idx, None):
                rel_dct[tail_uttr_idx] = {tail_word_idx: [head, rel]}
            else:
                rel_dct[tail_uttr_idx][tail_word_idx] = [head, rel]
                
        sent_lens_accum = [1]
        for i, item in enumerate(d['dialog']):
            utterance = item['utterance']
            sent_lens_accum.append(sent_lens_accum[i] + len(utterance.split(' ')) + 1)
        sent_lens_accum[0] = 0
        
        dep_lst:List[Dependency] = []
        role_lst:List[str] = []
        weak_signal = []
        for item in d['dialog']:
            turn = item['turn']
            utterance = item['utterance']

            pred_signals = next(signal_iter)

            role = '[ans]' if item['speaker'] == 'A' else '[qst]'
            dep_lst.append(Dependency(sent_lens_accum[turn], role, -1, '_'))  
            
            tmp_signal = []
            for word_idx, word in enumerate(utterance.split(' ')):
                tail2head = rel_dct.get(turn, {1: [f'{turn}-{word_idx}', 'adjct']})
                head, rel = tail2head.get(word_idx + 1, [f'{turn}-{word_idx}', 'adjct'])  # some word annoted missed, padded with last word and 'adjct'
                head_uttr_idx, head_word_idx = [int(x) for x in head.split('-')]
                
                # only parse cross-utterance
                if turn != head_uttr_idx:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, sent_lens_accum[head_uttr_idx] + head_word_idx, rel))  # add with accumulated length
                else:
                    dep_lst.append(Dependency(sent_lens_accum[turn] + word_idx + 1, word, -1, '_')) 

                try:
                    signal1, signal2 = pred_signals[i]
                except IndexError:
                    signal1, signal2 = pred_signals[len(pred_signals) - 1]
                
                tmp_signal = [signal1, signal2]
                
                # if word in weak_signal_dct.keys():
                #     tmp_signal.append(weak_signal_dct[word])

            if len(tmp_signal) != 0:
                # weak_signal.append(tmp_signal[-1])  # choose the last
                weak_signal.append(tmp_signal)  # choose the last
            else:
                weak_signal.append(-1)
            role_lst.append(item['speaker'])        
        sample_lst.append([dep_lst, role_lst, weak_signal])
        
    return sample_lst


class InterDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer= cfg.tokenizer
        self.inputs, self.offsets, self.heads, self.rels, self.masks, self.speakers, self.signs = self.read_data()
        
    def read_data(self):
        inputs, offsets = [], []
        tags, heads, rels, masks, speakers, signs = [], [], [], [], [], []
                
        for idx, (deps, roles, sign) in enumerate(load_inter(self.cfg.data_file)):
            seq_len = len(deps)
            signs.append(sign)

            word_lst = [] 
            head_tokens = np.zeros(1024, dtype=np.int64)  # same as root index is 0, constrainting by mask 
            rel_tokens = np.zeros(1024, dtype=np.int64)
            mask_tokens = np.zeros(1024, dtype=np.int64)
            for i, dep in enumerate(deps):
                if i == seq_len or i + 1== 1024:
                    break

                word_lst.append(dep.word)
                
                if int(dep.head) == -1 or int(dep.head) + 1 >= 1024:
                    head_tokens[i+1] = 0
                    mask_tokens[i+1] = 0
                else:
                    head_tokens[i+1] = int(dep.head)
                    mask_tokens[i+1] = 1
#                     head_tokens[i] = dep.head if dep.head != '_' else 0
                rel_tokens[i+1] = rel2id.get(dep.rel, 0)

            tokenized = self.tokenizer.encode_plus(word_lst, 
                                              padding='max_length', 
                                              truncation=True,
                                              max_length=1024, 
                                              return_offsets_mapping=True, 
                                              return_tensors='pt',
                                              is_split_into_words=True)
            inputs.append({"input_ids": tokenized['input_ids'][0],
                          "token_type_ids": tokenized['token_type_ids'][0],
                           "attention_mask": tokenized['attention_mask'][0]
                          })

#                 sentence_word_idx = np.zeros(self.cfg.max_length, dtype=np.int64)
            sentence_word_idx = []
            for idx, (start, end) in enumerate(tokenized.offset_mapping[0][1:]):
                if start == 0 and end != 0:
                    sentence_word_idx.append(idx)
#                         sentence_word_idx[idx] = idx
            if len(sentence_word_idx) < 1024 - 1:
                sentence_word_idx.extend([0]* (1024 - 1 - len(sentence_word_idx)))
            offsets.append(torch.as_tensor(sentence_word_idx))
#                 offsets.append(sentence_word_idx)

            heads.append(head_tokens)
            rels.append(rel_tokens)
            masks.append(mask_tokens)
            speakers.append(roles)
                    
        return inputs, offsets, heads, rels, masks, speakers, signs

    def __getitem__(self, idx):
        return self.inputs[idx], self.offsets[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.speakers[idx], self.signs[idx]
    
    def __len__(self):
        return len(self.rels)