#!/bin/bash

echo "y" | conda create -n mean_teacher python=3.5
CONDA_DIR=$(dirname `which conda`)
source $CONDA_DIR/activate mean_teacher
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip install torchvision numpy scipy sklearn tqdm pandas nltk  
git clone https://github.com/poojalnarayan/mean-teacher.git
cd mean-teacher/pytorch
git checkout fet
echo "Copying dataset from '/work/poojal/fet/figer_hw_runs/pytorch/data-local/fet' to 'pytorch/data-local' ..."
cp -a /work/poojal/fet/figer_hw_runs/pytorch/data-local/fet data-local
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=figer  --arch=simple_MLP_embed --epochs=10  --batch-size=256 --labels=100.0 --checkpoint-epochs=10  --exclude-unlabeled=true --random_seed=1 --run-name=mean_teacher_fet_figer_100p_10e_rs1_100kdata --consistency=0.3 --consistency-rampup=30 --weight-decay=0.0001
#echo "Please activate 'mean_teacher' environment before running the python command (via source activate mean_teacher)"
source $CONDA_DIR/deactivate mean_teacher
echo "Removing mean_teacher conda environment ..."
echo "y" | conda remove --name mean_teacher --all
