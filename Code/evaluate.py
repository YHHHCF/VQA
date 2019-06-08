import torch.nn as nn
import json

import baseline
from data_loader import *
import variables as var
from vqaEvaluation.vqaEval import VQAEval
import config.eval_cf as cfg


# get the result file by running a vqa on a pre-trained vqa model
def calculate_json_result(model, loader, epoch):
    qaRes = {}

    for bID, data in enumerate(loader):
        imgs = data['img'].to(var.device)
        ques = data['ques'].to(var.device)
        qID = data['qID']

        pred = model(imgs, ques)
        pred_idx = torch.argmax(pred, 1)

        for i in range(len(qID)):
            qaRes[qID[i].item()] = pred_idx[i].item()

        # debug
        break

    # get top anwsers
    top_ans = load_file(var.top_ans_path)

    # process answers
    results = []

    for key in qaRes.keys():
        result = {}
        result['question_id'] = key
        result['answer'] = top_ans[qaRes[key]]
        results.append(result)

    dir = os.path.join(cfg.result_dir, str(cfg.experiment), str(epoch))

    if not os.path.exists(dir):
        os.makedirs(dir)

    path = os.path.join(dir, cfg.result_name)

    with open(path, 'w') as file:
        json.dump(results, file)

    return path


# compute standard accuracy and perform analysis based on result file and annotation
def calculate_acc(vqa, resFile, quesFile):
    vqaRes = vqa.loadRes(resFile, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()

    acc = vqaEval.accuracy['overall']
    return acc


if __name__ == "__main__":
    model = baseline.BaselineModel()
    criterion = nn.CrossEntropyLoss()

    eval_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=False, num_workers=16)

    model.to(var.device)

    resFile = calculate_json_result(model, eval_loader, 0)

    acc = calculate_acc(eval_set.vqa, resFile, var.val_ques_path)
