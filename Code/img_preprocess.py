from data_loader import *

if __name__ == "__main__":
    train_vqa = VQA(var.train_ann_path, var.train_ques_path)
    val_vqa = VQA(var.val_ann_path, var.val_ques_path)
    test_dev_vqa = VQA(None, var.test_dev_ques_path)
    test_std_vqa = VQA(None, var.test_std_ques_path)

    # train_set = VqaDataset(var.train_img_path, var.train_img_name_pattern, var.train_img_proc_path,
    #                        var.train_ques_path, var.train_ques_embedding_path,
    #                        var.train_ann_path, var.train_ans_idxs_path)

    val_set = VqaDataset(var.val_img_path, var.val_img_name_pattern, var.val_img_proc_path,
                         var.val_ques_path, var.val_ques_embedding_path,
                         var.val_ann_path, var.val_ans_idxs_path)

    # test_dev_set = VqaDataset(var.test_img_path, var.test_dev_img_name_pattern, var.test_dev_img_proc_path,
    #                      var.test_dev_ques_path, var.test_dev_ques_embedding_path)
    #
    # test_dev_set = VqaDataset(var.test_img_path, var.test_std_img_name_pattern, var.test_std_img_proc_path,
    #                           var.test_std_ques_path, var.test_std_ques_embedding_path)
