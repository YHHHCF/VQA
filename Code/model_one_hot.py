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
        self.ques_pad = nn.ZeroPad2d((1, 1))  # pad on L dimension

        self.ques_filter = nn.Conv1d(256, 256, 3, 1, 1, groups=256)  # convert Q to 256-dim vector

        # init the second order output as 0(skip it at first)
        self.filter_bn = nn.BatchNorm1d(num_features=256)
        self.filter_bn.weight.data.zero_()
        self.filter_bn.bias.data.zero_()

        self.ques_linear = nn.Linear(256, var.top_ans_num + 1)  # classification layer

    def forward(self, v, q):  # v is input image, q is input question from data_loader
        v = self.img_encoder(v)  # (B, 2048)
        v = self.img_linear(v)  # (B, var.top_ans_num + 1)

        q = self.ques_conv(q)  # (B, D=256, L=14)
        q_f = self.ques_filter(q)
        q = q + self.filter_bn(q_f)

        q = torch.sum(q, 2)  # (B, D=256)
        q = self.ques_linear(q)  # (B, var.top_ans_num + 1)

        return v + q


class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.img_encoder_1, self.img_encoder_2 = ip.get_baseline(cf4.cut)  # a cnn to encode image to a 2048 dim vector
        self.img_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

        self.ques_conv = nn.Conv1d(var.top_vocab_num + 2, 256, 1, 1)  # convert Q to 256-dim vector
        self.ques_pad = nn.ZeroPad2d((1, 1))  # pad on L dimension

        # the second order question filter learned from CNN output
        self.so_filter = nn.Linear(cf4.ch * cf4.o_size * cf4.o_size, 256 * 3)  # 3 is kernel size
        self.drop_out = nn.Dropout(p=cf4.dp_rate)

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
        so_filter = self.so_filter(v_1)  # (B, D=256 * k_size=3)
        so_filter = self.drop_out(so_filter)
        so_filter = so_filter.reshape(so_filter.shape[0], -1, 3)  # (B, D=256, k_size=3)

        # question encoding
        q = self.ques_conv(q)  # (B, D=256, L=14)  go through first conv
        q_pad = self.ques_pad(q)  #  (B, D=256, L=16)

        # (B, D=256, L=14)
        q_so = so_filter[:,:,0:1] * q_pad[:,:,:-2]
        q_so = q_so + so_filter[:,:,1:2] * q_pad[:,:,1:-1]
        q_so = q_so + so_filter[:,:,2:3] * q_pad[:,:,2:]

        q = q + self.so_bn(q_so)  # (B, D=256, L=14)
        q = torch.sum(q, 2)  # (B, D=256)
        q = self.ques_linear(q)  # (B, var.top_ans_num + 1)

        return v_2 + q
