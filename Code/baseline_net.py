import torch
import torch.nn as nn
import torchvision.models as models


class BaselineNet(nn.Module):
    def __init__(self, model):
        super(BaselineNet, self).__init__()
        self.layers = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        embedding = self.layers(x)
        embedding = embedding.reshape(embedding.shape[1], -1)

        return embedding


# copy the weights from pretrained model to the new model
def get_model():
    pt_model = models.resnet152(pretrained=True)

    new_model = BaselineNet(pt_model)
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


# check whether the parameters of the model has changed
def print_model(model):
    params = model.state_dict()
    cnt = 0
    for key in params.keys():
        if cnt < 20:
            print(key, torch.max(params[key]))
            cnt += 1


# save the parameters of the model
def save_ckpt(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
    }, path)
    return
