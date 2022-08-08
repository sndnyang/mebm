


python eval_maebm.py --eval test_clf --uncond --load_path  $1
# test accuracy 94.08
# test loss 0.212

python eval_maebm.py --eval fid --uncond --load_path  $1  --ratio 0.9
# ratio  IS  FID
# 0.9   8.31  9.87
# 10000  8.18  10.38



# Qualitative Analysis
python eval_maebm.py --eval quality --uncond --load_path $1


# tSNE visualization

python eval_maebm.py --eval tsne --uncond --load_path  $1

# generate from scratch

python eval_maebm.py --eval gen --buffer_size 100 --batch_size 100 --n_sample_steps 1000  --uncond --n_steps 1 --print_every 10 --load_path $1

