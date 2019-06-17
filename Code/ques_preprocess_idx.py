import sys
sys.path.append('../')
from API.PythonHelperTools.vqaTools.vqa import VQA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import variables as var


# get the vocabularies using train/val set
def get_top_vocab(vocab_path, train_vqa, val_vqa, top_num):
    train_qqa = train_vqa.qqa
    val_qqa = val_vqa.qqa
    ques_all = []

    for key in train_qqa.keys():
        ques_all.append(train_qqa[key]['question'])

    for key in val_qqa.keys():
        ques_all.append(val_qqa[key]['question'])

    # get vocabulary
    vectorizer = CountVectorizer(max_features=top_num, vocabulary=None)
    vectorizer.fit_transform(ques_all)
    vocabulary = vectorizer.get_feature_names()

    np.savez(vocab_path, vocabulary)
    print("Saved a vocabulary with length {}".format(len(vocabulary)))
    return vocabulary


# take a list of questions and return the numeric version of them
def sentence2list(sentences, vocabulary, top_num):
    vectorizer = CountVectorizer(max_features=top_num, vocabulary=vocabulary)
    analyze = vectorizer.build_analyzer()

    sample = ['This is a sample.']
    vectorizer.fit_transform(sample)

    num_lists = []

    for sentence in sentences:
        seq = analyze(sentence)
        num_list = []

        for word in seq:
            num = vectorizer.vocabulary_.get(word)
            if num:
                num_list.append(num)
            else:
                num_list.append(top_num)

        # clip and pad to 14 words
        if len(num_list) > 14:
            num_list = num_list[:14]

        if len(num_list) < 14:
            while len(num_list) < 14:
                num_list.append(top_num + 1)

        num_list = np.array(num_list)
        num_lists.append(num_list)

    return num_lists


# encode the questions of a vqa obj and save it at the given path
# the encoded questions are saved as bag of word indices
def encode_question(vqa, vocabulary, save_path, top_num):
    qqa = vqa.qqa
    ques_all = []
    q_ids = qqa.keys()

    for key in q_ids:
        ques_all.append(qqa[key]['question'])

    ques_emb = sentence2list(ques_all, vocabulary, top_num)

    # save question as dictionary: key is question_id, value is encoded question
    ques_dict = {}
    q_ids = list(q_ids)
    for i in range(len(q_ids)):
        q_id = q_ids[i]
        emb = ques_emb[i]
        ques_dict[q_id] = emb

    np.savez(save_path, ques_dict)
    print("saved questions with length: ", len(ques_dict))
    print("saving path is: ", save_path)
    return


if __name__ == "__main__":
    train_vqa = VQA(var.train_ann_path, var.train_ques_path)
    val_vqa = VQA(var.val_ann_path, var.val_ques_path)
    test_dev_vqa = VQA(None, var.test_dev_ques_path)
    test_std_vqa = VQA(None, var.test_std_ques_path)

    # get the vocabulary from train and val questions
    vocab = get_top_vocab(var.ques_vocab_path, train_vqa, val_vqa, top_num=var.top_vocab_num)

    # question embedding for train/val/test/test_dev set
    encode_question(train_vqa, vocab, var.train_ques_idx_path, top_num=var.top_vocab_num)
    encode_question(val_vqa, vocab, var.val_ques_idx_path, top_num=var.top_vocab_num)
    encode_question(test_dev_vqa, vocab, var.test_dev_ques_idx_path, top_num=var.top_vocab_num)
    encode_question(test_std_vqa, vocab, var.test_std_ques_idx_path, top_num=var.top_vocab_num)
