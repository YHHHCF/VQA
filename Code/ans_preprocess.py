import sys
sys.path.append('../')
from API.PythonHelperTools.vqaTools.vqa import VQA
import numpy as np
import variables as var


# get the top answers using train/val set
def get_top_ans(ans_path, train_vqa, val_vqa, top_num):
    train_qa = train_vqa.qa
    val_qa = val_vqa.qa

    all_ans = []
    ans_cnt = {}  # a dictionary to count all the answers

    # count the number of all train/val answers and get top_num answers
    for q_id in train_qa.keys:
        answers = train_qa[q_id]['answers']
        all_ans.append(answers)
        for i in range(10):
            single_ans = answers[i]['answer']
            if single_ans in ans_cnt.keys():
                ans_cnt[single_ans] += 1
            else:
                ans_cnt[single_ans] = 1

    for q_id in val_qa.keys:
        answers = val_qa[q_id]['answers']
        all_ans.append(answers)
        for i in range(10):
            single_ans = answers[i]['answer']
            if single_ans in ans_cnt.keys():
                ans_cnt[single_ans] += 1
            else:
                ans_cnt[single_ans] = 1

    # get top_num answers
    sorted_ans_cnt = sorted(ans_cnt.items(), key=lambda kv: kv[1])
    top_answers = []
    for i in range(top_num):
        top_answers.append(sorted_ans_cnt[-i - 1][0])

    np.savez(ans_path, top_answers)
    print("saved {} top answers".format(len(top_answers)))
    print("saving path is:", ans_path)
    return top_answers


# encode the answers of a vqa obj and save it at the given path
def encode_answer(vqa, top_answers, save_path, top_num):
    qa = vqa.qa
    q_ids = qa.keys()

    # encode 10 answers to a (10, 2) tensor, with the 1st column being the ans_ids, the 2nd column being the count
    embed_answers = {}
    q_ids = list(q_ids)

    for i in range(len(q_ids)):
        if i % 10000 == 0:
            print(i)

        embed_ans = []
        q_id = q_ids[i]
        answers = qa[q_id]['answers']
        for j in range(10):
            answer = answers[j]['answer']
            if answer in top_answers:
                idx = top_answers.index(answer)
                embed_ans.append(idx)

        embed_ans = np.array(embed_ans)
        embed_ans = np.sort(embed_ans)  # <= 10 sorted index of answers(less than 10 if some answer not in top answers)

        # if all the answers are not in the top answers, index the gt answer to be top_num
        if len(embed_ans) == 0:
            embed_answers[q_id] = top_num

        # else find the answer with the highest count and let it be the gt answer
        else:
            start_pos = 0
            max_idx = embed_ans[0]
            max_len = 0

            for pos in range(len(embed_ans)):
                if embed_ans[pos] != embed_ans[start_pos]:
                    length = pos - start_pos
                    if length > max_len:
                        max_len = length
                        max_idx = embed_ans[start_pos]
                    start_pos = pos

            embed_answers[q_id] = max_idx

    # save the answer embedding as dictionary, key is answer_id, value is list ids
    np.savez(save_path, embed_answers)
    print("Saved {} embedded answers".format(len(embed_answers)))
    print("saving path is:", save_path)
    return


if __name__ == "__main__":
    train_vqa = VQA(var.train_ann_path, var.train_ques_path)
    val_vqa = VQA(var.val_ann_path, var.val_ques_path)

    # each element of the list is all the indices in the answer list which shows in top_ans_num
    # key is question_id, value is embedded answer
    top_ans = get_top_ans(var.top_ans_path, train_vqa, val_vqa, top_num=var.top_ans_num)

    # answer embedding for train/val set
    encode_answer(train_vqa, top_ans, var.train_ans_idxs_path, top_num=var.top_ans_num)
    encode_answer(val_vqa, top_ans, var.val_ans_idxs_path, top_num=var.top_ans_num)
