import torch.nn as nn
import torchvision.models as models
import variables as var


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.img_encoder = get_baseline()  # a cnn to encode image to a 2048 dim vector
        self.img_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

        self.ques_encoder = nn.Linear(var.top_vocab_num + 1, 2048)  # an fc to encode question to a 2048 dim vector
        self.activation = nn.ReLU(inplace=True)
        self.ques_linear = nn.Linear(2048, var.top_ans_num + 1)  # classification layer

    def forward(self, v, q):  # v is input image, q is input question from data_loader
        v = self.img_encoder(v)
        v = self.img_linear(v)

        q = self.ques_encoder(q)
        q = self.activation(q)
        q = self.ques_linear(q)

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

    return new_model
