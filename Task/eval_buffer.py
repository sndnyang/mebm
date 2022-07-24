import os
import time
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
import numpy as np
from torch.utils.data import DataLoader
from .eval_quality import eval_is_fid


def norm_ip(img, min, max):
    temp = t.clamp(img, min=min, max=max)
    temp = (temp + -min) / (max - min + 1e-5)
    return temp


def cond_is_fid(f, new_buffer, args, device, ratio=0.1, eval='all'):
    n_it = new_buffer.size(0) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in range(n_it):
            x = new_buffer[i * 100: (i + 1) * 100].to(device)
            logits, _ = f.classify(x)
            if 'dis' in args.model:
                logits = logits[0][0]
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    each_class = [new_buffer[all_y == l] for l in range(args.n_classes)]
    each_class_probs = [probs[all_y == l] for l in range(args.n_classes)]
    print([len(c) for c in each_class])

    new_buffer = []
    for c in range(args.n_classes):
        each_probs = each_class_probs[c]
        # print("%d" % len(each_probs))
        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk)
        index_list = topks[1]
        images = each_class[c][index_list]
        new_buffer.append(images)

    new_buffer = t.cat(new_buffer, 0)
    print(new_buffer.shape)
    metrics = eval_is_fid((new_buffer + 1) * 127.5, args.dataset, args)
    inc_score, std, fid = metrics['inception_score_mean'], metrics['inception_score_std'], metrics['frechet_inception_distance']
    if eval in ['is', 'all']:
        print("Inception score of {} with std of {}".format(inc_score, std))
    if eval in ['fid', 'all']:
        print("FID of score {}".format(fid))
    return inc_score, std, fid
