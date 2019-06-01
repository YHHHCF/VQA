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
        embed_ans = np.sort(embed_ans)
        vec_ans = np.full((10, 2), -1)

        mark = 0
        for ele in embed_ans:
            if ele in vec_ans[:, 0]:
                idx = np.where(vec_ans == ele)[0][0]
                vec_ans[idx][1] += 1
            else:
                vec_ans[mark][0] = ele
                vec_ans[mark][1] = 1
                mark += 1

        # sort according to the count
        vec_ans = -vec_ans
        vec_ans_sorted = vec_ans[vec_ans[:, 1].argsort()]
        vec_ans_sorted = -vec_ans_sorted

        # if all are -1, then replace the first row with (top_num, 10), meaning the answer is unknown
        if vec_ans_sorted[0][0] == -1:
            vec_ans_sorted[0][0] = top_num
            vec_ans_sorted[0][1] = 10

        embed_answers[q_id] = vec_ans_sorted

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
