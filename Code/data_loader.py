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
            ques_path: Path to question data json file
            ques_embedding_path: Path to question embedding
            ann_path: Path to annotations mapping images, questions, and answers together
            ans_idxs_path: Path to answer indices

        """
        self.vqa = VQA(ann_path, ques_path)
        self.img_dir = img_dir
        self.img_name_pattern = img_name_pattern

        # load and resize all the images
        self.images = {}
        image_ids = list(set(self.vqa.getImgIds()))  # all the image ids

        time_t = datetime.datetime.utcnow()
        cnt = 0
        for img_id in image_ids:
            if cnt % 1000 == 0:
                print(cnt)
            cnt += 1
            img_path = self.get_im_path(img_id)
            img = get_im(img_path)
            self.images[img_id] = img

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
        return len(self.vqa.dataset['annotations'])

    # return the input image, input question and answers corresponding to idx
    def __getitem__(self, idx):
        item = {}  # the item to be returned

        # get question_id
        q_id = self.qIDs[idx]

        # get image
        img_id = self.vqa.getImgIds(quesIds=[q_id])[0]
        item['img'] = self.images[img_id]

        # get questions
        ques = self.ques_embedding[q_id]
        vec_ques = torch.zeros(var.top_vocab_num + 1)
        vec_ques[ques] = 1  # multi-hot BOW embedding of the question
        item['ques'] = vec_ques

        # get answers
        ans = self.ans_idxs[q_id]
        item['ans'] = torch.tensor(ans)

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
    cnt = 0
    for key in params.keys():
        if cnt < 20:
            print(key, torch.max(params[key]))
            cnt += 1


# save the parameters of the model
def save_ckpt(model, loss, acc, path):
    torch.save({
        'param': model.state_dict(),
        'loss': loss,
        'acc': acc,
    }, path)
    return


# load the parameters from a ckpt to a model
def load_ckpt(path, model):
    ckpt = torch.load(path)
    print("Loss is {}, accuracy is {}".format(ckpt['loss'], ckpt['acc']))

    model.load_state_dict(ckpt['param'])
    return model


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
    ques_embbed_path = var.val_ques_embedding_path
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

    print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))
