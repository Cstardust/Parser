from typing import *
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def length_to_mask(length, max_len=None, dtype=None):
    """
    >>> lens = [3, 5, 4]
    >>> length_to_mask(length)
    >>> [[1, 1, 1, 0, 0],\
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 0]]
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    if max_len is None:
        max_len = max_len or torch.max(length)[0].item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype). \
               expand(length.shape[0], max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def seq_mask_by_lens(lengths: torch.Tensor,
                     max_len=None,
                     dtype=torch.bool,
                     device='cpu'):
    """
    giving sequence lengths, return mask of the sequence.
    example:
        input: 
        lengths = torch.tensor([4, 5, 1, 3])
        output:
        tensor([[ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [ True, False, False, False, False],
                [ True,  True,  True, False, False]])
    """
    if max_len is None:
        max_len = lengths.max()
    row_vector = torch.arange(start=0, end=max_len.item(), step=1).to(device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def to_cuda(data):
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError


def arc_rel_loss(
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_gt: torch.Tensor,  # ground truth
        rel_gt: torch.Tensor,
        mask: torch.Tensor, 
        wa = 0.8,
        wb = 1.2) -> torch.Tensor:
    flip_mask = mask.eq(0)

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss * wa + rel_loss * wb


# def arc_rel_loss_(arc_logits: torch.Tensor,
#                  rel_logits: torch.Tensor,
#                  arc_gt: torch.Tensor,  # ground truth
#                  rel_gt: torch.Tensor,
#                  mask: torch.Tensor) -> torch.Tensor:
#     flip_mask = mask.eq(0)

#     def one_loss(logits, gt):
#         tmp1 = logits.view(-1, logits.size(-1))
#         tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
#         return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

#     def loss_with_prob(logits, gt):
#         logits = logits.view(-1, logits.size(-1))
#         gt = gt.view(-1, gt.size(-1))
#         loss_fn = nn.CrossEntropyLoss(reduction='none')
#         loss = loss_fn(logits, gt)  # batch_size, seq_len
#         loss = loss.view(mask.size())
#         loss *= mask
#         loss = loss.sum(-1)
#         return loss.mean()

#     arc_loss = one_loss(arc_logits, arc_gt)
#     rel_loss = loss_with_prob(rel_logits, rel_gt)

#     return arc_loss + rel_loss


def uas_las(
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_gt: torch.Tensor,  # ground truth
        rel_gt: torch.Tensor,
        mask: torch.Tensor) -> Dict:
    """
    CoNLL:
    LAS(labeled attachment score): the proportion of “scoring” tokens that are assigned both the correct head and the correct dependency relation label.
    Punctuation tokens are non-scoring. In very exceptional cases, and depending on the original treebank annotation, some additional types of tokens might also be non-scoring.
    The overall score of a system is its labeled attachment score on all test sets taken together.

    UAS(Unlabeled attachment score): the proportion of “scoring” tokens that are assigned the correct head (regardless of the dependency relation label).
    """
    if len(arc_logits.shape) > len(arc_gt.shape):
        pred_dim, indices_dim = 2, 1
        arc_logits = arc_logits.max(pred_dim)[indices_dim]
        # print('uas_las arc_logits', arc_logits.size())
        # print('uas_las arc_logits', arc_logits)
            # uas_las arc_logits torch.Size([4241, 160])    [i,j] 第i批, 以j词作为尾词. 其父节点的位置
            # uas_las arc_logits tensor([[0, 4, 4,  ..., 4, 4, 4],
            #         [0, 5, 3,  ..., 5, 5, 5],
            #         [0, 3, 3,  ..., 3, 3, 3],
            #         ...,
            #         [0, 2, 0,  ..., 2, 2, 2],
            #         [0, 2, 0,  ..., 2, 2, 2],
            #         [0, 2, 3,  ..., 2, 2, 2]])
        
    if len(rel_logits.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_logits = rel_logits.max(pred_dim)[indices_dim]
        # print('uas_las rel_logits', rel_logits.size())
        # print('uas_las rel_logits', rel_logits)
            # uas_las rel_logits torch.Size([4241, 160])    #  [i,j] 第i批, 以j词作为尾词. 其依存弧的标签
            # uas_las rel_logits tensor([[ 0,  9,  4,  ...,  9,  9,  9],
            #         [ 0,  9,  8,  ...,  9,  9,  9],
            #         [ 0,  4,  9,  ...,  4,  4,  4],
            #         ...,
            #         [ 0,  4,  0,  ...,  4,  4,  4],
            #         [ 0,  4,  0,  ...,  4,  4,  4],
            #         [ 0, 18,  8,  ..., 18, 18, 18]])
    arc_logits_correct = (arc_logits == arc_gt).long() * mask
    rel_logits_correct = (rel_logits == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()
    return {'UAS': float(arc) / float(num), 'LAS': float(rel) / float(num)}, arc_logits, rel_logits

def inner_inter_uas_las(
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_gt: torch.Tensor,  # ground truth
        rel_gt: torch.Tensor,
        mask: torch.Tensor) -> Dict:
    """
    CoNLL:
    LAS(labeled attachment score): the proportion of “scoring” tokens that are assigned both the correct head and the correct dependency relation label.
    Punctuation tokens are non-scoring. In very exceptional cases, and depending on the original treebank annotation, some additional types of tokens might also be non-scoring.
    The overall score of a system is its labeled attachment score on all test sets taken together.

    UAS(Unlabeled attachment score): the proportion of “scoring” tokens that are assigned the correct head (regardless of the dependency relation label).
    """

    if len(arc_logits.shape) > len(arc_gt.shape):
        pred_dim, indices_dim = 2, 1
        arc_logits = arc_logits.max(pred_dim)[indices_dim]
        # print('uas_las arc_logits', arc_logits.size())
        # print('uas_las arc_logits', arc_logits)
            # uas_las arc_logits torch.Size([4241, 160])    [i,j] 第i批, 以j词作为尾词. 其父节点的位置
            # uas_las arc_logits tensor([[0, 4, 4,  ..., 4, 4, 4],
            #         [0, 5, 3,  ..., 5, 5, 5],
            #         [0, 3, 3,  ..., 3, 3, 3],
            #         ...,
            #         [0, 2, 0,  ..., 2, 2, 2],
            #         [0, 2, 0,  ..., 2, 2, 2],
            #         [0, 2, 3,  ..., 2, 2, 2]])
        
    if len(rel_logits.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_logits = rel_logits.max(pred_dim)[indices_dim]
        # print('uas_las arc_logits', rel_logits.size())
        # print('uas_las arc_logits', rel_logits)
            # uas_las arc_logits torch.Size([4241, 160])    #  [i,j] 第i批, 以j词作为尾词. 其依存弧的标签
            # uas_las arc_logits tensor([[ 0,  9,  4,  ...,  9,  9,  9],
            #         [ 0,  9,  8,  ...,  9,  9,  9],
            #         [ 0,  4,  9,  ...,  4,  4,  4],
            #         ...,
            #         [ 0,  4,  0,  ...,  4,  4,  4],
            #         [ 0,  4,  0,  ...,  4,  4,  4],
            #         [ 0, 18,  8,  ..., 18, 18, 18]])

    # arc_logits_correct = (arc_logits == arc_gt).long() * mask * (rel_gt >= 21).long()

    # print(arc_logits.shape)
    # print(arc_gt.shape)

    inner_uas = 0.0
    inner_las = 0.0
    inter_las = 0.0
    inter_uas = 0.0
    inter_total = 0.0
    inner_total = 0.0

    for idx in range(arc_logits.shape[0]):
        once_arc_logit = arc_logits[idx]
        once_rel_logit = rel_logits[idx]

        once_arc_gt = arc_gt[idx]
        once_rel_gt = rel_gt[idx]

        once_mask = mask[idx]

        for jdx,once_rel in enumerate(once_rel_logit):
            if once_mask[jdx]==0:
                continue

            if once_rel_gt[jdx]>21:
                inter_total+=1
                if once_rel>21 and once_arc_logit[jdx]==once_arc_gt[jdx]:
                    if once_rel==once_rel_gt[jdx]:
                        inter_las+=1
                    inter_uas+=1
            if once_rel_gt[jdx]<=21:
                inner_total+=1
                if once_rel<=21 and once_arc_logit[jdx]==once_arc_gt[jdx]:
                    if once_rel==once_rel_gt[jdx]:
                        inner_las+=1
                    inner_uas+=1

    return {'Inter-UAS': round(float(inter_uas) / float(inter_total),4),'Inner-UAS': round(float(inner_uas) / float(inner_total),4), 'Inter-LAS': round(float(inter_las) / float(inter_total),4),'Inner-LAS': round(float(inner_las) / float(inner_total),4)}


def arc_rel_loss_split(
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_gt: torch.Tensor,  # ground truth
        rel_gt: torch.Tensor,
        mask: torch.Tensor) -> torch.Tensor:
    batch_size, num_uttrs, seq_len = arc_gt.size()
    arc_gt = arc_gt.view(batch_size, -1)
    rel_gt = rel_gt.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    flip_mask = mask.eq(0)

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


def uas_las_split(
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_gt: torch.Tensor,  # ground truth
        rel_gt: torch.Tensor,
        mask: torch.Tensor) -> Dict:
    batch_size, num_uttrs, seq_len = arc_gt.size()
    arc_gt = arc_gt.view(batch_size, -1)
    rel_gt = rel_gt.view(batch_size, -1)
    mask = mask.view(batch_size, -1)

    if len(arc_logits.shape) > len(arc_gt.shape):
        pred_dim, indices_dim = 2, 1
        arc_logits = arc_logits.max(pred_dim)[indices_dim]

    if len(rel_logits.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_logits = rel_logits.max(pred_dim)[indices_dim]

    arc_logits_correct = (arc_logits == arc_gt).long() * mask
    rel_logits_correct = (rel_logits == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()

    return {'UAS': float(arc) / float(num), 'LAS': float(rel) / float(num)}


def seed_everything(seed):
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
