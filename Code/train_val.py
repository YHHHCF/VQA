import torch.nn as nn
from data_loader import *
import baseline
from tensorboardX import SummaryWriter


def train(model, optimizer, criterion, train_loader, val_loader, writer):
    for e in range(var.epoch):
        for bID, data in enumerate(train_loader):
            imgs = data['img'].to(var.device)
            ques = data['ques'].to(var.device)
            ans = data['ans'].to(var.device)

            pred = model(imgs, ques)
            loss = criterion(pred, ans)

            writer.add_scalar('train/loss', loss.item(), bID)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_acc = val(model, criterion, val_loader)
        print("Epoch {}: loss is {}, acc is {}".format(e, val_loss, val_acc))
        writer.add_scalar('val/loss', val_loss.item(), bID)
        writer.add_scalar('val/acc', val_acc.item(), bID)

        # save model
        dir = os.path.join(var.model_dir, var.experiment, str(e))

        if not os.path.exists(dir):
            os.makedirs(dir)

        path = os.path.join(dir, var.model_name)
        save_ckpt(model, optimizer, val_loss.item(), val_acc.item(), path)


def val(model, criterion, loader):
    model.eval()
    with torch.no_grad():
        cnt = 0
        total_len = 0
        acc = 0.0
        val_loss = 0.0

        for bID, data in enumerate(loader):
            imgs = data['img'].to(var.device)
            ques = data['ques'].to(var.device)
            ans = data['ans'].to(var.device)

            pred = model(imgs, ques)

            val_loss += criterion(pred, ans)

            # use a different metric to calculate acc here (count whether the prediction meets the best answer)
            pred_idx = torch.argmax(pred, 1)
            acc += sum(pred_idx == ans)

            cnt += 1
            total_len += len(ans)

        acc = float(acc) / total_len
        val_loss /= cnt

    model.train()

    return val_loss, acc


if __name__ == "__main__":
    model = baseline.BaselineModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=var.lr, weight_decay=var.wd)
    criterion = nn.CrossEntropyLoss()

    # dataset for train/val VQA
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                               var.train_ques_path, var.train_ques_embedding_path,
                               var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    train_loader = DataLoader(train_set, batch_size=var.train_batch_size, shuffle=True,
                                  num_workers=var.num_workers)
    val_loader = DataLoader(val_set, batch_size=var.val_batch_size, shuffle=False, num_workers=var.num_workers)

    model.to(var.device)

    writer = SummaryWriter()

    train(model, optimizer, criterion, train_loader, val_loader, writer)
