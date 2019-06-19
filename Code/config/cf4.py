import os
import variables as var

# experiment configuration
epoch = 30
train_batch_size = 128
val_batch_size = 128
num_workers = 8
lr = 3e-5
wd = 1e-5
dp_rate = 0.75  # drop out rate of second order filter learning

if_pretrain = False
pt_exp_id = 1
pt_epoch = 29

pt_exp = 'exp' + str(pt_exp_id)
ckpt_path = os.path.join(var.model_dir, pt_exp, str(pt_epoch), var.model_name)

# the place to cut res block
# 5 is the end of conv2; 6 is the end of conv3
# 7 is the end of conv4; 8 is the end of conv5
cut = 6
o_size = 28
ch = 512
