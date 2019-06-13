from train_val import *
from config import cf1
import baseline


# train the baseline model
def run_experiment():
    model = baseline.BaselineModel()

    params = list(model.parameters())

    for param in params[:-6]:
        param.requires_grad = False

    optimizer = torch.optim.Adam(params[-6:], lr=cf1.lr, weight_decay=cf1.wd)
    criterion = nn.CrossEntropyLoss()

    if cf1.if_pretrain:
        path = cf1.ckpt_path
        model, optimizer = load_ckpt(model, optimizer, path)

    # dataset for train/val VQA
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                               var.train_ques_path, var.train_ques_embedding_path,
                               var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    train_loader = DataLoader(train_set, batch_size=cf1.train_batch_size, shuffle=True,
                                  num_workers=cf1.num_workers)
    val_loader = DataLoader(val_set, batch_size=cf1.val_batch_size, shuffle=False, num_workers=cf1.num_workers)

    model = model.to(var.device)

    writer = SummaryWriter()

    train(model, optimizer, criterion, train_loader, val_loader, writer, cf1.epoch, experiment=1)


def run_experiment_1():
    v_model = baseline.BaselineModelV()

    if cf1.if_pretrain:
        path = cf1.ckpt_path
        v_model = load_ckpt(v_model, None, path)

    params = list(v_model.parameters())

    for param in params[:-2]:
        param.requires_grad = False

    optimizer = torch.optim.Adam(params[-2:], lr=cf1.lr, weight_decay=cf1.wd)
    criterion = nn.CrossEntropyLoss()

    # dataset for train/val VQA
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                           var.train_ques_path, var.train_ques_embedding_path,
                           var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    train_loader = DataLoader(train_set, batch_size=cf1.train_batch_size, shuffle=True,
                              num_workers=cf1.num_workers)
    val_loader = DataLoader(val_set, batch_size=cf1.val_batch_size, shuffle=False, num_workers=cf1.num_workers)

    v_model = v_model.to(var.device)

    print(v_model)

    if cf1.if_pretrain:
      val_loss, val_acc = val(v_model, criterion, val_loader, experiment=11)
      print("Experiment 1.1: val loss is {}. val acc is {}".format(val_loss, val_acc))
    else:
      writer = SummaryWriter()
      train(v_model, optimizer, criterion, train_loader, val_loader, writer, cf1.epoch, experiment=11)


def run_experiment_2():
    q_model = baseline.BaselineModelQ()
    if cf1.if_pretrain:
        path = cf1.ckpt_path
        q_model = load_ckpt(q_model, None, path)

    params = list(q_model.parameters())

    for param in params[:-4]:
        param.requires_grad = False

    optimizer = torch.optim.Adam(params[-4:], lr=cf1.lr, weight_decay=cf1.wd)
    criterion = nn.CrossEntropyLoss()

    # dataset for train/val VQA
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                           var.train_ques_path, var.train_ques_embedding_path,
                           var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # data_loader for train/val VQA
    train_loader = DataLoader(train_set, batch_size=cf1.train_batch_size, shuffle=True,
                              num_workers=cf1.num_workers)
    val_loader = DataLoader(val_set, batch_size=cf1.val_batch_size, shuffle=False, num_workers=cf1.num_workers)

    q_model = q_model.to(var.device)

    print(q_model)

    if cf1.if_pretrain:
      val_loss, val_acc = val(q_model, criterion, val_loader, experiment=12)
      print("Experiment 1.2: val loss is {}. val acc is {}".format(val_loss, val_acc))
    else:
      writer = SummaryWriter()
      train(q_model, optimizer, criterion, train_loader, val_loader, writer, cf1.epoch, experiment=12)
