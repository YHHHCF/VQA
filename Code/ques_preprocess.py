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


# encode the questions of a vqa obj and save it at the given path
# the encoded questions are saved as bag of word indices
def encode_question(vqa, vocabulary, save_path, top_num):
    qqa = vqa.qqa
    ques_all = []
    q_ids = qqa.keys()

    for key in q_ids:
        ques_all.append(qqa[key]['question'])

    # get top_num words in questions and encode them to vectors with top_num elements
    vectorizer = CountVectorizer(max_features=top_num, vocabulary=vocabulary)
    ques_BoW = vectorizer.fit_transform(ques_all).toarray()

    # save question as dictionary: key is question_id, value is encoded question
    ques_dict = {}
    q_ids = list(q_ids)
    for i in range(len(q_ids)):
        q_id = q_ids[i]
        bow = ques_BoW[i]
        ques_dict[q_id] = np.nonzero(bow)[0]  # get the indices of words(bag of words)

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
    encode_question(train_vqa, vocab, var.train_ques_embedding_path, top_num=var.top_vocab_num)
    encode_question(val_vqa, vocab, var.val_ques_embedding_path, top_num=var.top_vocab_num)
    encode_question(test_dev_vqa, vocab, var.test_dev_ques_embedding_path, top_num=var.top_vocab_num)
    encode_question(test_std_vqa, vocab, var.test_std_ques_embedding_path, top_num=var.top_vocab_num)
