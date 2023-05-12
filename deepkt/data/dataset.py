# -*- coding:utf-8 -*-
import torch.utils.data
import torch.nn.utils
import numpy as np


class KTDataset(torch.utils.data.Dataset):
    def __init__(self, df, n_skill, max_len=64):
        super(KTDataset, self).__init__()
        self.df = df
        self.n_skill = n_skill
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        qids = self.df[0][idx].split(",")
        correct = self.df[1][idx].split(",")

        if len(qids) > self.max_len:
            qids = qids[-self.max_len :]
            correct = correct[-self.max_len :]

        qids = list(map(int, qids))
        correct = list(map(int, correct))

        qid, qa = torch.LongTensor(qids), torch.LongTensor(correct)
        qa = qid + qa * self.n_skill

        mask = np.ones((len(qid),), dtype=int)

        return (
            torch.cat((torch.LongTensor([2 * self.n_skill]), qa[:-1])),
            qid.view(-1, 1),
            torch.LongTensor(correct),
            torch.LongTensor(mask),
        )

# 将输入数据进行填充，使其成为可以传入神经网络的张量
def collate_fn(data, n_skill):
    qa = [x[0] for x in data]
    qid = [x[1] for x in data]
    qc = [x[2] for x in data]
    mask = [x[3] for x in data]
    qa = torch.nn.utils.rnn.pad_sequence(
        qa, batch_first=True, padding_value=n_skill * 2
    )
    qid = torch.nn.utils.rnn.pad_sequence(qid, batch_first=True, padding_value=n_skill)
    qc = torch.nn.utils.rnn.pad_sequence(qc, batch_first=True, padding_value=-1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

    return qa, qid, qc, mask
