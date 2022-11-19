
Requirements

Python 3.6 or 3.7
PyTorch version 1.4

Run Model Training and Evaluation

Semi-Supervised Training and Test

SSTSC：

python train_ssl.py --dataset_name CricketX --model_name train_SemiInterPF   --alpha  0.3  --label_ratio  [0.1 0.2 0.4 1.0]

Supervised Training and Test：

python train_ssl.py --dataset_name CricketX --model_name SupCE  --label_ratio  [0.1 0.2 0.4 1.0]

Check Results
After runing model training and evaluation, the checkpoints of the trained model are saved in the local [ckpt] directory, the training logs are saved in the local [log] directory, and all experimental results are saved in the local [results] directory.