## To create the project, go to Data/
#### bash ./make_data
####    

## To perform preprocessing, go to Code/
#### python img_preprocess.py
#### python ques_preprocess.py
#### python ans_preprocess.py
####    

## To run experiment, go to Code/
#### set configuration in Code/config/
#### run baseline experiment: python train_val.py -e 1
#### run improvement: python train_val.py -e 2
####    

## To show result on tensorBoard
#### tensorboard --logdir=VQA/Code/runs
####  
