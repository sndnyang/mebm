
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
 --sgld_lr 1     \
 --sgld_std 0.0  \
 --l2_coeff 0.1  \
 --uncond \
 --gpu-id 6

# n_steps=2 (K=2) is very small, set  l2_coeff 0.1 and sgld_std 0
# K>=5,  l2_coeff=0.5, sgld_std=0.001