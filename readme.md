# Energy-Based Models


## Usage

1. pip install -r requirements.txt
2. python one_center.py --dataset cifar10  # generate GM's mu and sigma

the pretrained model link https://drive.google.com/drive/folders/1akoGmrjnChUu0HcIVziq25k2PDKgdx6C?usp=sharing

## Training

**Note:** when n_steps(K) is extremely small (1 or 2), l2_coeff doesn't help or leads a negative effect.

To train an M-EBM model on CIFAR10 as in the paper, please refer to scripts/maebm_cifar10.sh

```bash
python train_maebm.py --dataset cifar10 \
     --lr .1 --optimizer sgd \
     --px 1.0 --pyx 1.0 \
     --data_root ../../data \
     --sigma .0 --width 10 --depth 28 \
     --plot_uncond --warmup_iters 1000 \
     --model wrn \
     --norm batch \
     --print_every 100 \
     --n_epochs 200 --decay_epochs 60 120 180 \
     --n_steps 5   \
     --sgld_lr 1   \
     --sgld_std 0.0 \
     --no_fid       \
     --no_wandb     \
     --l2_coeff 0.5   \
     --uncond \
     --gpu-id 3
```


To train an M-JEM model on CIFAR10 as in the paper, please refer to scripts/majem_cifar10.sh

```bash
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
     --sgld_std 0.0 \
     --l2_coeff 0.5   \
     --uncond \
     --gpu-id 0
```


To train an M-EBM model on CelebA-HQ as in the paper, please refer to scripts/maebm_cifar10.sh
The images should be stored in args.data_root/train/*/xxx.jpg

## Evaluation

Please check script/eval_all_in_one.sh

To evaluate the classifier (on CIFAR10), please refer to scripts/eval_ebm.sh

loss  |  accuracy
------|-----------
0.212 |   94.08

To evaluate the FID in the replay buffer (on CIFAR10):

python eval_maebm.py --eval fid --uncond --load_path /PATH/TO/YOUR/MODEL.pt --ratio 0.9

ratio  |   IS | FID
-------|------|---------
0.9    | 8.31 | 9.87
10000  | 8.18 | 10.38



## generate new samples

```
python eval_maebm.py --eval gen \
       --buffer_size 100 \
       --batch_size 100 \
       --n_sample_steps 400  \
       --uncond --n_steps 1 \
       --print_every 10 \
       --load_path $1
```

# t-SNE visualization

Left one: refer to the paper

Right figure: 
```
Using a non-final-epoch checkpoint of unconditional EBM trained on CIFAR-10 --- a final checkpoint should look similar
to visualize the t-SNE of learned features from different classes (10 classes - 10 colors)

The mixing of different classes may result in more 'manifold intrusion', an intuitive hypothesis. 
```



|  |  |
|--|--|
| ![all](https://github.com/sndnyang/mebm/blob/master/tsne.png)  | ![uncond_category](https://github.com/sndnyang/mebm/blob/master/uncond_categorial_tsne.png) |


# Ref
If you found this work useful and used it on your own research, please consider citing this paper.

```
@inproceedings{xiulong2023mebm,
 author = {Xiulong Yang, Shihao Ji},
 booktitle = {Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
 title = {M-EBM: Towards Understanding the Manifolds of Energy-Based Models},
 url = {https://arxiv.org/abs/2303.04343},
 year = {2023}
}
```
