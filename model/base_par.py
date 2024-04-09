
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

from model.module import NonLinear, Biaffine
from utils import arc_rel_loss


class DepParser(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        mlp_arc_size:int = cfg.hidden_size
        mlp_rel_size:int = cfg.hidden_size
        dropout = cfg.dropout

        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(cfg.tokenizer))

        self.mlp_arc_dep = NonLinear(in_features=self.encoder.config.hidden_size, 
                                     out_features=mlp_arc_size+mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(in_features=self.encoder.config.hidden_size, 
                                      out_features=mlp_arc_size+mlp_rel_size, 
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((mlp_arc_size + mlp_rel_size) / 100)
        self.arc_num = int(mlp_arc_size / 100)
        self.rel_num = int(mlp_rel_size / 100)
        
        self.arc_biaffine = Biaffine(mlp_arc_size, mlp_arc_size, 1)
        self.rel_biaffine = Biaffine(mlp_rel_size, mlp_rel_size, cfg.num_labels)

        self.dropout = nn.Dropout(dropout)
     
    def feat(self, inputs):
        length = torch.sum(inputs["attention_mask"], dim=-1) - 2
        
        feats, *_ = self.encoder(**inputs, return_dict=False)   # batch_size, seq_len (tokenized), plm_hidden_size
           
        # remove [CLS] [SEP]
        word_cls = feats[:, :1]
        char_input = torch.narrow(feats, 1, 1, feats.size(1) - 2)
        return word_cls, char_input, length
        
    def forward(self, inputs, offsets, heads, rels, masks, evaluate=False):  # inputs: batch_size, seq_len
        # inputs = [batch size, seq_len] 输入（token id）
        # offsets = [batch size, seq_len] 每个词的第一个字在整个句子中的索引位置
        # heads = [batch size, seq_len] # 每个词依赖的head词的第一个字的索引位置
        # rels = [batch size, seq_len] # 每一个词对应的依赖关系的编码
        # masks = [batch size, seq_len] # 主要用于loss计算中将pad位置的loss值屏蔽掉

        # cls shape = [bs,1,768]
        # char feat = [bs,158,768]
        # word_len = [bs,1]
        cls_feat, char_feat, word_len = self.feat(inputs)

        # word_idx [bs,seq_len-1,768]
        # word_feat [bs,seq_len-1,768]
        word_idx = offsets.unsqueeze(-1).expand(-1, -1, char_feat.shape[-1])  # expand to the size of char feat
        word_feat = torch.gather(char_feat, dim=1, index=word_idx)  # embeddings of first char in each word

        # 拼接上cls位置的向量，可能原因是PLM的原因，因为PLM的输入有[CLS]，为了保证句子表示的完整性
        # word_cls_feat [bs,seq_len,768]
        # feats [bs,seq_len,768]
        word_cls_feat = torch.cat([cls_feat, word_feat], dim=1)
        feats = self.dropout(word_cls_feat)
    
        # all_dep [bs,seq_len,800]
        # all_head [bs,seq_len,800]
        all_dep = self.dropout(self.mlp_arc_dep(feats))
        all_head = self.dropout(self.mlp_arc_head(feats))

        # all_dep_splits [8,bs,seq_len,100]
        # all_head_splits [8,bs,seq_len,100]
        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        # arc_dep [bs,seq_len,400]
        # arc_head [bs,seq_len,400]
        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        # arc_logit = [bs,seq_len,seq_len]
        # 例子：“我 是 中国人” 在训练时我们并不知道“我”这个字的head是谁，所以就全猜，给每个pair对都赋值一个值:（我,我,value）,（我,是,value）,（我,中国人,value）
        # 对于“是”，“中国人”同样，所以总共需要9个值，这个句子的seq_len = 3所以用矩阵形式表示就是[3(seq_len)x3(seq_len)]，训练时再加上批维度，就是[bs,seq_len,seq_len]
        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        arc_logit = arc_logit.squeeze(3)

        # rel_dep [bs,seq_len,400]
        # rel_head [bs,seq_len,400]
        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)
        
        # rel_logit_cond [bs,seq_len,seq_len,num_rels]
        # 这个的解释同arc_logit的解释蕾丝，例如：“我 是 中国人” 在训练时我不知道“我”这个词到底与那个头具有何种关系，那就全猜，就有:
        # (我，我，关系1，value),(我，我，关系2，value),(我，我，关系3，value),.....
        # (我，是，关系1，value),(我，是，关系2，value),(我，是，关系3，value),.....
        # (我，中国人”，关系1，value),(我，中国人”，关系2，value),(我，中国人”，关系3，value),.....
        # ......
        # “是”和“中国人”这两个词依次类推，加上批维度，用张量表示就是[bs,seq_len,seq_len,num_rels]
        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, num_rels
        
        if evaluate:
            # 一个词可以作为多个词的head，但是一个词的head只能有一个，所以取Max就行，如果多对多这里需要取topk
            # 如果是评估阶段没有heads的标签，所以只能用predict的heads来进行依存关系的输入
            _, heads = arc_logit.max(2)  # change golden heads to the predicted
        
        # index = [batch_size, seq_len, seq_len, num_rels]
        # rel_logit = [batch_size, seq_len, num_rels]
        # 为什这里要按照index取，因为在上一步中我们已经知道每个词的head是谁了，就是知道“# (我，我，关系1，value),(我，我，关系2，value),(我，我，关系3，value),.....”
        # 这一行中“我”的真实head是谁，其他的pair对就可以丢弃掉，因此这里需要按照index去取值。
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        
        if evaluate:
            return arc_logit, rel_logit
        else:
            loss = arc_rel_loss(arc_logit, rel_logit, heads, rels, masks)
            return arc_logit, rel_logit, loss