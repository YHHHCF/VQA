from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('../')
from API.PythonHelperTools.vqaTools.vqa import VQA
import variables as var
import torch
import os
import torchvision.transforms as trans
from PIL import Image
import datetime
import numpy as np


# resize and normalize training image before feeding into CNN
img_norm = trans.Compose([trans.Resize((224, 224)),
                          trans.ToTensor(),
                          trans.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])])


# get a reshaped and normalized image
def get_im(path):
    img = Image.open(path).convert("RGB")
    img = img_norm(img)
    return img


# the dataset of VQA
class VqaDataset(Dataset):
    def __init__(self, img_dir, img_name_pattern, ques_path, ques_embedding_path,
                 ann_path=None, ans_idxs_path=None):
        """
        Args:
            img_dir: Path to the directory with COCO images
            img_name_pattern: (eg "COCO_train2014_{}.jpg")
            img_proc_dir: preprocessed image dir
            ques_path: Path to question data json file
            ques_embedding_path: Path to question embedding
            ann_path: Path to annotations mapping images, questions, and answers together
            ans_idxs_path: Path to answer indices

        """
        self.vqa = VQA(ann_path, ques_path)
        self.img_dir = img_dir
        self.img_name_pattern = img_name_pattern

        # load and resize all the images
        image_ids = list(set(self.vqa.getImgIds()))  # all the image ids

        time_t = datetime.datetime.utcnow()

        img_proc_dir = os.path.join(img_dir, 'pre_process')

        # if img is not preprocessed, process it and save to disk
        if not os.path.exists(img_proc_dir):
            print("this line should only appear in img pre-processing!")
            os.makedirs(img_proc_dir)
            cnt = 0
            for img_id in image_ids:
                if cnt % 1000 == 0:
                    print(cnt)
                cnt += 1
                img_path = self.get_im_path(img_id)
                img = get_im(img_path)

                torch.save(img, os.path.join(img_proc_dir, str(img_id) + '.pt'))

        print('Image loaded (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        self.qIDs = self.vqa.getQuesIds()

        self.ques_embedding, self.ques_keys = load_dict(ques_embedding_path)

        if ans_idxs_path:
            self.ans_idxs, self.ans_keys = load_dict(ans_idxs_path)
        else:
            self.ans_idxs = None
            self.ans_keys = None

        return

    def __len__(self):
        # return len(self.vqa.dataset['annotations'])
        return len(self.qIDs)

    # return the input image, input question and answers corresponding to idx
    def __getitem__(self, idx):
        item = {}  # the item to be returned

        # get question_id
        q_id = self.qIDs[idx]

        # get image
        # print("debug: ", idx, q_id)
        # print("debug 1: ", self.vqa.getImgIds(quesIds=[q_id]))
        img_id = self.vqa.getImgIds(quesIds=[q_id])[0]
        if len(self.vqa.getImgIds(quesIds=[q_id])) > 1:
            print("debug")
        item['img'] = torch.load(os.path.join(self.img_dir, 'pre_process', str(img_id) + '.pt'))

        # get questions
        ques = self.ques_embedding[q_id]  # (L,) each element in [0, top_num + 1]
        ques = torch.tensor(ques).reshape(1, -1)
        ques_one_hot = torch.zeros(var.top_vocab_num + 2, ques.shape[1]).scatter_(0, ques, 1)
        item['ques'] = ques_one_hot

        # get answers
        if self.ans_idxs:
            ans = self.ans_idxs[q_id]
            item['ans'] = torch.tensor(ans)

        # put question id into item, used for evaluation
        item['qID'] = q_id

        return item

    # get image path from image_id
    def get_im_path(self, img_id):
        img_id = str(img_id)
        img_id = img_id.zfill(12)
        pattern = self.img_name_pattern.split('{}')
        im_name = pattern[0] + img_id + pattern[1]

        path = os.path.join(self.img_dir, im_name)
        return path


# check whether the parameters of the model has changed
def print_model(model):
    params = model.state_dict()
    key_1 = list(params.keys())[:10]
    key_2 = list(params.keys())[-10:]

    print("===============start===============")
    for key in key_1:
        print(key, torch.max(params[key]))

    for key in key_2:
        print(key, torch.max(params[key]))
    print("==============end================")


# save the parameters of the model
def save_ckpt(model, optimizer, loss, acc, path):
    torch.save({
        'param': model.state_dict(),
        'optim': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
    }, path)
    return


# load the parameters from a ckpt to a model and optimizer
def load_ckpt(model, optimizer, path):
    if not torch.cuda.is_available():  # run on cpu
        ckpt = torch.load(path, map_location='cpu')
    else:  # run on gpu
        ckpt = torch.load(path)

    print("Pre-trained loss is {}, accuracy is {}".format(ckpt['loss'], ckpt['acc']))

    print("Before loading")
    print_model(model)
    model.load_state_dict(ckpt['param'], strict=False)
    print("After loading")
    print_model(model)

    # if do not want to load optimizer, set it None
    if not optimizer:
        return model

    # if want to load pre-trained optimizer, pass it in
    else:
        optimizer.load_state_dict(ckpt['optim'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(var.device)
        return model, optimizer


def load_file(path):
    file = np.load(path, allow_pickle=True)
    file = file['arr_0']
    file = list(file)
    return file


def load_dict(path):
    dict = np.load(path, allow_pickle=True)
    dict = dict['arr_0']
    dict = dict.item()
    keys = dict.keys()
    keys = list(keys)
    return dict, keys


# demo code
if __name__ == "__main__":
    img_dir = var.val_img_path
    ques_path = var.val_ques_path
    ques_embbed_path = var.val_ques_idx_path
    ann_path = var.val_ann_path
    ans_idxs_path = var.val_ans_idxs_path
    img_name_pattern = var.val_img_name_pattern

    # dataset for VQA
    dataset = VqaDataset(img_dir, img_name_pattern, ques_path, ques_embbed_path, ann_path, ans_idxs_path)

    # dataloader for VQA
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16)

    time_t = datetime.datetime.utcnow()
    for batch_id, batch_data in enumerate(loader):
        print(batch_id)
        print(batch_data['img'].shape)
        print(batch_data['ques'].shape)
        print(batch_data['ans'].shape)

    print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))
