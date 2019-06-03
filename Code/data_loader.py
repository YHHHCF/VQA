from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('../')
from API.PythonHelperTools.vqaTools.vqa import VQA
from load_helper import *
import variables as var
import torch
import os
import torchvision.transforms as trans
from PIL import Image
import datetime

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
        im_id = str(self.vqa.getImgIds(quesIds=[q_id])[0])
        im_id = im_id.zfill(12)
        pattern = self.img_name_pattern.split('{}')
        im_name = pattern[0] + im_id + pattern[1]

        img = get_im(os.path.join(self.img_dir, im_name))

        item['img'] = img

        # get questions
        ques = self.ques_embedding[q_id]
        vec_ques = torch.zeros(var.top_vocab_num + 1)
        vec_ques[ques] = 1  # multi-hot BOW embedding of the question
        item['ques'] = vec_ques

        # get answers
        ans = self.ans_idxs[q_id]
        item['ans'] = torch.tensor(ans)

        return item

# the dataloader for VQA


if __name__ == "__main__":
    img_dir = var.val_img_path
    ques_path = var.val_ques_path
    ques_embbed_path = var.val_ques_embedding_path
    ann_path = var.val_ann_path
    ans_idxs_path = var.val_ans_idxs_path
    img_name_pattern = var.val_img_name_pattern

    dataset = VqaDataset(img_dir, img_name_pattern, ques_path, ques_embbed_path, ann_path, ans_idxs_path)

    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=16)

    time_t = datetime.datetime.utcnow()
    for batch_id, batch_data in enumerate(loader):
        print(batch_id)

    print('DONE (t=%0.2fs)' % ((datetime.datetime.utcnow() - time_t).total_seconds()))
