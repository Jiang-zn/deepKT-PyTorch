# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class DeepDINA(nn.Module):
    def __init__(self, n_skill, q_embed_dim, qa_embed_dim, hidden_dim, kp_dim,
                 n_layer, dropout, device="cpu", cell_type="lstm", ):
        super(DeepDINA, self).__init__()
        self.n_skill = n_skill
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.hidden_dim = hidden_dim
        self.kp_dim = kp_dim
        self.n_layer = n_layer
        self.dropout = dropout

        self.device = device
        self.rnn = None

        #问题嵌入，答案嵌入，
        # skill_specific_ability表示每个知识点的能力的张量
        # student_ability_prediction预测学生知识水平
        # 猜错率，失误率
        # 问题难度预测，问题区分度预测
        self.q_embedding = nn.Embedding(n_skill + 1, q_embed_dim, padding_idx=n_skill)
        self.qa_embedding = nn.Embedding(2 * n_skill + 1, qa_embed_dim, padding_idx=2 * n_skill)
        self.skill_specific_ability = nn.Parameter(torch.randn(n_skill, q_embed_dim))
        self.student_ability_prediction = nn.Sequential(
            nn.Linear(q_embed_dim, qa_embed_dim),
            nn.ReLU(),
            nn.Linear(q_embed_dim, n_skill)
        )
        self.guessin_prediction = nn.Linear(self.q_embed_dim + self.qa_embed_dim, 1)
        self.slipping_prediction = nn.Linear(self.q_embed_dim + self.qa_embed_dim, 1)
        self.question_difficulty_prediction = nn.Linear(q_embed_dim, 1)
        self.question_discrimination_prediction = nn.Linear(q_embed_dim, 1)
        # self.q_kp_relation = nn.Linear(self.q_embed_dim, self.kp_dim)
        # self.q_difficulty = nn.Linear(self.q_embed_dim, self.kp_dim)
        # self.user_ability = nn.Linear(self.hidden_dim, self.kp_dim)

        if cell_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "rnn":
            self.rnn = nn.RNN(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )
        elif cell_type.lower() == "gru":
            self.rnn = nn.GRU(
                self.qa_embed_dim,
                self.hidden_dim,
                self.n_layer,
                batch_first=True,
                dropout=self.dropout,
            )
        if self.rnn is None:
            raise ValueError("cell type only support lstm, rnn or gru type.")

    def forward(self, q, qa):
        q_embed_data = self.q_embedding(q)
        qa_embed_data = self.qa_embedding(qa)
        batch_size = q.size(0)
        seq_len = q.size(1)

        student_ability = self.student_ability_prediction(q_embed_data)
        skill_specific_ability = self.skill_specific_ability
        guessing_probability = torch.sigmoid(self.guessing_prediction(torch.cat([q_embed_data, qa_embed_data], dim=1)))
        slipping_probability = torch.sigmoid(self.slipping_prediction(torch.cat([q_embed_data, qa_embed_data], dim=1)))
        question_difficulty = self.question_difficulty_prediction(q_embed_data)
        question_discrimination = self.question_discrimination_prediction(q_embed_data)
        p_learn = (1 - guessing_probability) * slipping_probability
        p_known = (1 - guessing_probability) * (1 - slipping_probability)
        p_g = guessing_probability
        p_s = 1 - (p_learn + p_known + p_g)

        log_p_learn = torch.log(p_learn + 1e-9)
        log_p_known = torch.log(p_known + 1e-9)
        log_p_g = torch.log(p_g + 1e-9)
        log_p_s = torch.log(p_s + 1e-9)
        item_response_prob = (p_learn * question_difficulty +
                              p_known * (1 - question_difficulty)) ** question_discrimination

        # # h0 = torch.zeros((q.size(0), self.n_layer, self.hidden_dim), device=self.device)
        # states, _ = self.rnn(qa_embed_data)
        # # states_before = torch.cat((h0, states[:, :-1, :]), 1)
        # user_ability = self.user_ability(states).view(batch_size * seq_len, -1)
        #
        # kp_relation = torch.softmax(
        #     self.q_kp_relation(q_embed_data.view(batch_size * seq_len, -1)), dim=1
        # )
        # item_difficulty = self.q_difficulty(q_embed_data.view(batch_size * seq_len, -1))
        #
        # logits = (user_ability - item_difficulty) * kp_relation

        # return logits.sum(dim=1), None
        # 学生在每个知识点上的掌握程度，猜错率，失误率，学生在回答每个问题时做对或做错的概率，问题的难度
        return student_ability, guessing_probability, slipping_probability, item_response_prob, question_difficulty
