This supplement only constains POL data (code/dataset/pol/) to show the reproduction ability.
To reset the random seed (line 11 in trainer.py) to zero to repeat the experiments, 

The other datasets can be got from the links presented in paper and preprocessed with preprocess.py first.

Notice that though POL dataset is not run for constrained models in the paper, you can still run the spatial constrained models for it.

steps:
```commandline
sh run_tensorflow_docker.sh

pip3 install -r requirements.txt

python3 preprocess.py

python3 trainer.py
```