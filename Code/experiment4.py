from train_val import *
from config import cf4
import model_one_hot


# train the baseline model
def run_experiment():
    model = model_one_hot.ImprovedModel()

    params = list(model.parameters())

    for param in params[:-6]:
        param.requires_grad = False

    optimizer = torch.optim.Adam(params[-6:], lr=cf4.lr, weight_decay=cf4.wd)
    criterion = nn.CrossEntropyLoss()

    if cf4.if_pretrain:
        path = cf4.ckpt_path
        model, optimizer = load_ckpt(model, optimizer, path)

    # dataset for train/val VQA
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                               var.train_ques_path, var.train_ques_idx_path,
                               var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_idx_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    train_loader = DataLoader(train_set, batch_size=cf4.train_batch_size, shuffle=True,
                                  num_workers=cf4.num_workers)
    val_loader = DataLoader(val_set, batch_size=cf4.val_batch_size, shuffle=False, num_workers=cf4.num_workers)

    model = model.to(var.device)

    writer = SummaryWriter()

    train(model, optimizer, criterion, train_loader, val_loader, writer, cf4.epoch, experiment=4)
