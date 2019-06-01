import os

data_path = '../Data'

# paths for image/questions/annotations
train_img_path = os.path.join(data_path, 'Images', 'train2014')
val_img_path = os.path.join(data_path, 'Images', 'val2014')
test_img_path = os.path.join(data_path, 'Images', 'test2015')

train_ques_path = os.path.join(data_path, 'Questions', 'v2_OpenEnded_mscoco_train2014_questions.json')
val_ques_path = os.path.join(data_path, 'Questions', 'v2_OpenEnded_mscoco_test2015_questions.json')
test_dev_ques_path = os.path.join(data_path, 'Questions', 'v2_OpenEnded_mscoco_test-dev2015_questions.json')
test_std_ques_path = os.path.join(data_path, 'Questions', 'v2_OpenEnded_mscoco_val2014_questions.json')

train_ann_path = os.path.join(data_path, 'Annotations', 'v2_mscoco_train2014_annotations.jsonn')
val_ann_path = os.path.join(data_path, 'Annotations', 'v2_mscoco_val2014_annotations.json')

# paths for question embeddings
train_ques_embedding_path = os.path.join(data_path, 'Questions', 'train_ques_embedding.npz')
val_ques_embedding_path = os.path.join(data_path, 'Questions', 'val_ques_embedding.npz')
test_dev_ques_embedding_path = os.path.join(data_path, 'Questions', 'test_dev_ques_embedding.npz')
test_std_ques_embedding_path = os.path.join(data_path, 'Questions', 'test_std_ques_embedding.npz')

# paths for answer indices
train_ans_idxs_path = os.path.join(data_path, 'Annotations', 'train_ans_idxs.npz')
val_ans_idxs_path = os.path.join(data_path, 'Annotations', 'val_ans_idxs.npz')

# path for question vocabulary
ques_vocab_path = os.path.join(data_path, 'Questions', 'ques_vocab.npz')

# path for top answers
top_ans_path = os.path.join(data_path, 'Annotations', 'top_answers.npz')

# experiment related
top_vocab_num = 3000
top_ans_num = 3000


