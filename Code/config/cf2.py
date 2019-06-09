import os
import variables as var

experiment = 'exp2'

# experiment configuration
epoch = 30
train_batch_size = 128
val_batch_size = 128
num_workers = 16
lr = 3e-5
wd = 1e-5

if_pretrain = True
pt_exp_id = 1
pt_epoch = 0

pt_exp = 'exp' + str(pt_exp_id)
ckpt_path = os.path.join(var.model_dir, pt_exp, str(pt_epoch), var.model_name)
