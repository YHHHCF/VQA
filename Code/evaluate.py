import torch.nn as nn
import json

import baseline
from data_loader import *
import variables as var

import sys
sys.path.append('../')
from API.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

import model_one_hot

import config.eval_cf as cfg
from config import cf3
from config import cf4


# get the result file by running a vqa on a pre-trained vqa model
def calculate_json_result(model, loader, epoch):
    qaRes = {}

    for bID, data in enumerate(loader):
        if bID % 100 == 0:
            print(bID)

        imgs = data['img'].to(var.device)
        ques = data['ques'].to(var.device)
        qID = data['qID']

        pred = model(imgs, ques)
        pred_idx = torch.argmax(pred, 1)

        for i in range(len(qID)):
            qaRes[qID[i].item()] = pred_idx[i].item()

    # get top anwsers
    top_ans = load_file(var.top_ans_path)

    # process answers
    results = []

    for key in qaRes.keys():
        result = {}
        result['question_id'] = key

        if qaRes[key] < var.top_ans_num:
            result['answer'] = top_ans[qaRes[key]]
        else:
            result['answer'] = None

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
    eval_set = VqaDataset(var.test_img_path, var.test_img_name_pattern,
                         var.test_std_ques_path, var.test_std_ques_idx_path,
                         None, None)

    eval_loader = DataLoader(eval_set, batch_size=128, shuffle=False, num_workers=16)

    model = model_one_hot.BaselineModel()
    model.to(var.device)

    path = cf3.ckpt_path
    model = load_ckpt(model, optimizer=None, path=path)

    model.eval()
    with torch.no_grad():
        resFile = calculate_json_result(model, eval_loader, 0)

    # acc calculation is only for train/val, which has annotations
    # acc = calculate_acc(eval_set.vqa, resFile, var.test_dev_ques_path)

    print("acc: ", acc)

