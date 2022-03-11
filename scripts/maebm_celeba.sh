
python train_maebm_celeba.py --dataset=celeba128 --data_root celeba/data128x128/ \
      --lr=.02 --optimizer=sgd --px=1.0 --width=2 --depth=28 \
      --warmup_iters=1000 \
      --model=wrn --norm batch \
      --print_every=100 --n_epochs=200 --decay_epochs 80 120 160 \
      --n_steps=10 --sgld_lr=1 --sigma=.0 \
      --l2_coeff 0. --batch_size 64 \
      --gpu-id=3