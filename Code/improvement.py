import torch.nn as nn
import torch
import torchvision.models as models
import variables as var
import torch.nn.functional as F
from data_loader import *
import config.cf2 as cf2


class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.img_encoder_1, self.img_encoder_2 = get_baseline()  # a cnn to encode image to a 2048 dim vector
        self.img_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

        self.ques_encoder = nn.Linear(var.top_vocab_num + 1, 2048)  # an fc to encode question to a 2048 dim vector

        # the second order conv filter learned from question embedding
        self.so_filter = nn.Linear(2048, 256 * 3 * 3)

        # init the second order output as 0(skip it at first)
        self.so_bn = nn.BatchNorm2d(num_features=256)
        self.so_bn.weight.data.zero_()
        self.so_bn.bias.data.zero_()

        self.activation = nn.ReLU(inplace=True)
        self.ques_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

    def forward(self, v, q):  # v is input image, q is input question from data_loader
        q = self.ques_encoder(q)  # (B, 2048)
        so_filter = self.so_filter(q)  # (B, (512*3*3))

        # get filter parameters
        so_filter = so_filter.reshape((-1, 256, 1, 3, 3))  # (B, 256, 1, 3, 3)
        q = self.activation(q)  # (B, 2048)
        q = self.ques_linear(q)  # (B, top_ans_num + 1)

        v = self.img_encoder_1(v)  # (B, 256, 56, 56)

        # go throught each so_filter batch by batch
        v_out = torch.zeros_like(v).to(var.device)  # (B, 256, 56, 56)

        for i in range(v.size(0)):
            v_batch = F.conv2d(v[i].unsqueeze(0), so_filter[i], padding=1, groups=256)
            v_out[i] = v_batch.squeeze(0)

        # a short cut
        v += self.so_bn(v_out)  # (B, 256, 56, 56)
        v = self.img_encoder_2(v)  # (B, 2048, 1, 1)
        v = v.reshape(v.shape[0], v.shape[1])  # (B, 2048)
        v = self.img_linear(v)  # (B, top_ans_num + 1)

        return v + q


class BaselineCNN(nn.Module):
    def __init__(self, model):
        super(BaselineCNN, self).__init__()
        self.layers = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        embedding = self.layers(x)
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1])

        return embedding


# copy the weights from pre_trained model to the new model
def get_baseline():
    pt_model = models.resnet50(pretrained=True)

    new_model = BaselineCNN(pt_model)
    params_pt = pt_model.state_dict()
    params_new = new_model.state_dict()

    params_list = []
    for param in params_pt:
        params_list.append(params_pt[param])

    idx = 0
    for param in params_new:
        params_new[param] = params_list[idx]
        idx += 1

    new_model.load_state_dict(params_new)

    # the place to cut res block
    # 5 is the end of conv2_x; 6 is the end of conv3_x
    cut = 5

    new_model_1 = new_model.layers[:cut]  # split from the end of conv3_x
    new_model_2 = new_model.layers[cut:]
    return new_model_1, new_model_2
