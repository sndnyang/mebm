# MA-EBM


## Usage

1. pip install -r requirements.txt
2. python one_center.py --dataset cifar10  # generate GM's mu and sigma

### Training

To train an MA-EBM model on CIFAR10 as in the paper, please refer to scripts/maebm_cifar10.sh

python train_maebm.py --dataset cifar10 \
     --lr .1 --optimizer sgd \
     --px 1.0 --pyx 0.0 \
     --sigma .0 --width 10 --depth 28 \
     --plot_uncond --warmup_iters 1000 \
     --model wrn \
     --norm batch \
     --print_every 100 \
     --n_epochs 200 --decay_epochs 60 120 180 \
     --n_steps 2     \
     --sgld_lr 1   \
     --sgld_std 0.001 \
     --l2_coeff 0.5   \
     --uncond \
     --gpu-id 0


To train an MA-JEM model on CIFAR10 as in the paper, please refer to scripts/majem_cifar10.sh


python train_maebm.py --dataset cifar10 \
     --lr .1 --optimizer sgd \
     --px 1.0 --pyx 1.0 \
     --sigma .0 --width 10 --depth 28 \
     --plot_uncond --warmup_iters 1000 \
     --model wrn \
     --norm batch \
     --print_every 100 \
     --n_epochs 200 --decay_epochs 60 120 180 \
     --n_steps 5      \
     --sgld_lr 1   \
     --sgld_std 0.001 \
     --l2_coeff 0.5   \
     --uncond \
     --gpu-id 0


### Evaluation

To evaluate the classifier (on CIFAR10), please refer to scripts/eval_ebm.sh

To evaluate the FID in the replay buffer (on CIFAR10):

python eval_maebm.py --eval fid --load_path /PATH/TO/YOUR/MODEL.pt --ratio 0.9
