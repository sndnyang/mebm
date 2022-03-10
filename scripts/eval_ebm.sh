
#python eval_maebm.py --eval test_clf --load_path $1
python eval_maebm.py --eval fid --load_path $1 --ratio 0.9
#
#python eval_maebm.py --eval gen --buffer_size 100 --batch_size 100 --n_sample_steps 1000 --load_path $1
