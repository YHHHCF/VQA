from data_loader import *

if __name__ == "__main__":
    train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern,
                           var.train_ques_path, var.train_ques_embedding_path,
                           var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    test_set = VqaDataset(var.test_img_path, var.test_img_name_pattern,
                         var.test_std_ques_path, var.test_std_ques_embedding_path)
