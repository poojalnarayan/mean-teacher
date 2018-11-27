#!/bin/bash

echo "y" | conda create -n mean_teacher python=3.5
CONDA_DIR=$(dirname `which conda`)
source $CONDA_DIR/activate mean_teacher
pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
pip install torchvision numpy scipy sklearn tqdm pandas nltk  
git clone https://github.com/poojalnarayan/mean-teacher.git
cd mean-teacher/pytorch
git checkout noise_expts0.4_fullctx
echo "Copying dataset from ' /work/poojal/noise_expts_ctx/mean-teacher/pytorch/data-local/nec' to 'pytorch/data-local' ..."
cp -a /work/poojal/noise_expts_ctx/mean-teacher/pytorch/data-local/nec data-local
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=ontonotes_ctx  --arch=custom_embed --epochs=1  --batch-size=64 --labels=440 --checkpoint-epochs=20  --hidden-size=300  --labeled-batch-size=16 --random_seed=30 --word-noise="drop:1"  --run-name=25nov_custom_onto_drop_1_c1_1e_440l_rs30 --consistency=1.0 --consistency-rampup=5 --weight-decay=0.0001
#echo "Please activate 'mean_teacher' environment before running the python command (via source activate mean_teacher)"
source $CONDA_DIR/deactivate mean_teacher
echo "Removing mean_teacher conda environment ..."
echo "y" | conda remove --name mean_teacher --all
