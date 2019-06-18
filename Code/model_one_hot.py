import baseline as bl
import improvement as ip
import torch
import torch.nn as nn
import variables as var
from config import cf4


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.img_encoder = bl.get_baseline()  # a cnn to encode image to a 2048 dim vector
        self.img_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

        self.ques_conv = nn.Conv1d(var.top_vocab_num + 2, 256, 1, 1)  # convert Q to 256-dim vector
        self.ques_linear = nn.Linear(256, var.top_ans_num + 1)  # classification layer

    def forward(self, v, q):  # v is input image, q is input question from data_loader
        v = self.img_encoder(v)  # (B, 2048)
        v = self.img_linear(v)  # (B, var.top_ans_num + 1)

        q = self.ques_conv(q)  # (B, D=256, L=14)
        q = torch.sum(q, 2)  # (B, D=256)
        q = self.ques_linear(q)  # (B, var.top_ans_num + 1)

        return v + q


class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.img_encoder_1, self.img_encoder_2 = ip.get_baseline()  # a cnn to encode image to a 2048 dim vector
        self.img_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

        self.ques_conv = nn.Conv1d(var.top_vocab_num + 2, 256, 1, 1)  # convert Q to 256-dim vector

        # the second order question filter learned from CNN output
        self.so_filter = nn.Linear(cf4.ch * cf4.o_size * cf4.o_size, 256)
        self.drop_out = nn.Dropout(p=0.5)

        # init the second order output as 0(skip it at first)
        self.so_bn = nn.BatchNorm1d(num_features=256)
        self.so_bn.weight.data.zero_()
        self.so_bn.bias.data.zero_()

        # self.activation = nn.ReLU(inplace=False)
        self.ques_linear = nn.Linear(256, var.top_ans_num + 1)  # classification layer

    def forward(self, v, q):  # v is input image, q is input question from data_loader
        # image encoding
        v_1 = self.img_encoder_1(v)  # (B, ch, o, o)
        v_2 = self.img_encoder_2(v_1)  # (B, 2048, 1, 1)
        v_2 = v_2.reshape(v_2.shape[0], v_2.shape[1])  # (B, 2048)
        v_2 = self.img_linear(v_2)  # (B, top_ans_num + 1)

        # so question filter
        v_1 = v_1.reshape(v_1.shape[0], -1)  # (B, ch * o * o)
        so_filter = self.so_filter(v_1)  # (B, D=256)
        so_filter = self.drop_out(so_filter)
        so_filter = so_filter.reshape(so_filter.shape[0], so_filter.shape[1], 1)  # (B, D=256, 1)

        # question encoding
        q = self.ques_conv(q)  # (B, D=256, L=14)  go through first conv
        q_so = so_filter * q  # (B, D=256, L=14)

        q = q + self.so_bn(q_so)  # (B, D=256, L=14)
        q = torch.sum(q, 2)  # (B, D=256)
        q = self.ques_linear(q)  # (B, var.top_ans_num + 1)

        return v_2 + q
