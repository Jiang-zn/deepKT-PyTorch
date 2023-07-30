# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional


class DKTLoss(nn.Module):
    def __init__(self, reduce=None):
        super(DKTLoss, self).__init__()
        self.reduce = reduce

    def forward(self, logits, targets, qid, mask, device="cpu"):
        preds = torch.sigmoid(logits)
        preds = torch.gather(preds, dim=2, index=qid)
        preds = torch.squeeze(preds)
        ones = torch.ones(targets.size(), device=device)
        total = torch.sum(mask) + 1
        loss = -torch.sum(
            mask * targets * torch.log(preds)
            + mask * (ones - targets) * torch.log(ones - preds)
        )

        if self.reduce is None or self.reduce == "mean":
            loss = loss / total

        if self.reduce is not None and self.reduce not in ["mean", "sum"]:
            raise ValueError("the reduce should be mean or sum")

        return loss


# logits是形状为 (batch_size, max_seq_len, n_skill) 的张量，表示学生在不同题目上的掌握程度的预测值
# targets是形状为 (batch_size, max_seq_len, n_skill) 的张量，是真实的学生答题情况，表示学生在不同题目上的实际掌握程度
# qid是形状为 (batch_size, max_seq_len) 的张量，表示学生在不同题目上的题目id
# mask是形状为 (batch_size, max_seq_len) 的张量，用于标记哪些位置是真实数据，哪些位置是填充数据
class DeepIRTLoss(nn.Module):
    def __init__(self, reduce="mean"):
        super(DeepIRTLoss, self).__init__()
        self.reduce = reduce

    def forward(self, logits, targets, qid, mask, device="cpu"):
        mask = mask.gt(0).view(-1)
        targets = targets.view(-1)

        logits = torch.masked_select(logits, mask)
        targets = torch.masked_select(targets, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction=self.reduce
        )
        return loss


def dkt_predict(logits, qid):
    preds = torch.sigmoid(logits)
    preds = torch.gather(preds, dim=2, index=qid)
    preds = torch.squeeze(preds)
    binary_preds = torch.round(preds)
    return (preds.view(preds.size()[0], preds.size()[1]),
            binary_preds.view(preds.size()[0], preds.size()[1]),)
