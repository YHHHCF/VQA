from train_val import *
from config import cf2
import improvement


# insert second order network block into CNN and consume training based on
# pre-trained network in experiment 1
def run_experiment():
    imp_model = improvement.ImprovedModel()

    optimizer = torch.optim.Adam(imp_model.parameters(), lr=cf2.lr, weight_decay=cf2.wd)
    criterion = nn.CrossEntropyLoss()

    if cf2.if_pretrain:
        path = cf1.ckpt_path
        model, optimizer = load_ckpt(model, optimizer, path)

    # dataset for train/val VQA
    # train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
    #                            var.train_ques_path, var.train_ques_embedding_path,
    #                            var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    # train_loader = DataLoader(train_set, batch_size=cf2.train_batch_size, shuffle=True,
    #                               num_workers=cf2.num_workers)
    val_loader = DataLoader(val_set, batch_size=cf2.val_batch_size, shuffle=False, num_workers=cf2.num_workers)

    imp_model = imp_model.to(var.device)

    writer = SummaryWriter()

    # train(imp_model, optimizer, criterion, train_loader, val_loader, writer, cf2.epoch)
    train(imp_model, optimizer, criterion, val_loader, val_loader, writer, cf2.epoch)
