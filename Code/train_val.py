import variables as var
import baseline
import torch
import torch.nn as nn
from data_loader import *
import baseline


#
def train(model, loader):
    raise NotImplemented


def val(model, loader):
    raise NotImplemented


if __name__ == "__main__":
    model = baseline.BaselineModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=var.lr, weight_decay=var.wd)
    criterion = nn.CrossEntropyLoss()

    # dataset for train/val VQA
    # train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
    #                            var.train_ques_path, var.train_ques_embedding_path,
    #                            var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                             var.val_ques_path, var.val_ques_embedding_path,
                             var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    # train_loader = DataLoader(train_set, batch_size=var.train_batch_size, shuffle=True,
    #                               num_workers=var.num_workers)
    val_loader = DataLoader(val_set, batch_size=var.val_batch_size, shuffle=False, num_workers=var.num_workers)

    baselineModel = baseline.BaselineModel()
    baselineModel.to(var.device)

    for e in range(var.epoch):
        for batch_id, batch_data in enumerate(val_loader):
            print(batch_id)

            imgs = batch_data['img'].to(var.device)
            ques = batch_data['ques'].to(var.device)
            ans = batch_data['ans'].to(var.device)

            pred = baselineModel(imgs, ques)
            loss = criterion(pred, ans)

            print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
