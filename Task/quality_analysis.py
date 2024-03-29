import os.path

import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import plot


def show_top(each_class, each_class_probs, each_class_metrics, ratio, dir_path, metric='px'):
    new_buffer = []
    for c in range(10):
        each_probs = each_class_probs[c]
        # select
        each_metric = each_class_metrics[c]

        if abs(ratio) < 1:
            topk = abs(int(len(each_metric) * ratio))
        else:
            topk = abs(int(ratio))
        topk = min(topk, len(each_probs))
        if ratio > 0:
            topks = torch.topk(each_metric, topk, largest=ratio > 0)
            index_list = topks[1]
        else:
            topks = torch.topk(each_metric, topk, largest=ratio > 0)
            index_list = topks[1]

        print('P(x) min %.3f max %.3f' % (each_metric[index_list].max().item(), each_metric[index_list].min().item()))
        print('Prob(y|x) max %.3f min %.3f' % (each_probs[index_list].max().item(), each_probs[index_list].min().item()))
        images = each_class[c][index_list]
        new_buffer.append(images[:10])
        plot('{}/{}k/{}_{}.png'.format(dir_path, 'top' if ratio > 0 else 'bottom', metric, c), images)
    replay_buffer = torch.cat(new_buffer, 0)
    plot('{}/{}k_{}.png'.format(dir_path, 'top' if ratio > 0 else 'bottom', metric), replay_buffer)


def qualitative_analysis(f, buffer, buffer_path, dataset='cifar10'):
    import seaborn as sns
    sns.set()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # only for cifar10 or stl10
    names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    dir_path = os.path.dirname(buffer_path)
    # I think the size should be 10k or 50k
    f.eval()
    n = len(buffer)
    all_y = []
    probs = []
    px_set = []
    with torch.no_grad():
        for i in range(int(n / 100)):
            batch = buffer[i * 100:(i + 1) * 100].to(device)
            logits = f.classify(batch)[0]
            px = logits.logsumexp(dim=1)
            prob_v = F.softmax(logits.detach(), dim=1)
            y = logits.max(1)[1]
            prob = prob_v.max(1)[0]
            all_y.append(y)
            probs.append(prob)
            px_set.append(px)

    all_y = torch.cat(all_y, 0)
    probs = torch.cat(probs, 0)
    each_class = [buffer[all_y == i] for i in range(10)]
    each_class_probs = [probs[all_y == i] for i in range(10)]
    px_set = torch.cat(px_set, 0)
    each_class_pxs = [px_set[all_y == i] for i in range(10)]

    print([len(c) for c in each_class])

    fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, sharey=True, figsize=(20, 8))
    for i in range(10):
        r = int(i / 5)
        c = i % 5
        axes[r][c].hist(each_class_pxs[i].cpu().numpy(), bins=50, orientation='horizontal')
        # axes[r][c].set_ylim(2, 4.5)
        axes[r][c].set_title(names[r * 5 + c])
    axes[0][0].set_ylabel('LSE(x)')
    axes[1][0].set_ylabel('LSE(x)')
    fig.tight_layout()
    fig.savefig(f'{dir_path}/px_{dataset}.png', dpi=400)
    plt.close()

    fig, axes = plt.subplots(ncols=5, nrows=2, sharex=True, sharey=True, figsize=(20, 8))
    for i in range(10):
        r = int(i / 5)
        c = i % 5
        axes[r][c].hist(each_class_probs[i].cpu().numpy(), bins=50, orientation='horizontal')
        # axes[r][c].set_ylim(0, 1)
        axes[r][c].set_title(names[r * 5 + c])
    axes[0][0].set_ylabel('p(y|x)')
    axes[1][0].set_ylabel('p(y|x)')
    fig.tight_layout()
    fig.savefig(f'{dir_path}/pyx_{dataset}.png', dpi=400)
    plt.close()

    os.makedirs(f'{dir_path}/topk', exist_ok=True)
    os.makedirs(f'{dir_path}/bottomk', exist_ok=True)
    show_top(each_class, each_class_probs, each_class_pxs, 100, dir_path, metric='px')
    show_top(each_class, each_class_probs, each_class_pxs, -100, dir_path, metric='px')
    show_top(each_class, each_class_probs, each_class_probs, 100, dir_path, metric='pyx')
    show_top(each_class, each_class_probs, each_class_probs, -100, dir_path, metric='pyx')


def draw_evolution(num_list, avg_list, dir_path):
    names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    for i in range(10):
        plt.plot(np.arange(1001), num_list[:, i], label=f'{names[i]}')
    plt.legend()
    plt.ylabel('Number', fontsize=16)
    plt.xlabel('Diffusion Step', fontsize=16)
    plt.savefig(f'{dir_path}/sample_evolution_num.png', dpi=300)
    plt.close()

    for i in range(10):
        plt.plot(np.arange(1001), avg_list[:, i], label=f'{names[i]}')
    plt.legend()
    # plt.ylim(0, 1)
    plt.ylabel('Probability', fontsize=16)
    plt.xlabel('Diffusion Step', fontsize=16)
    plt.savefig(f'{dir_path}/sample_evolution_prob.png', dpi=300)
    plt.close()
    np.save(f'{dir_path}/num.npy', num_list)
    np.save(f'{dir_path}/prob.npy', avg_list)


def sample_evolution(f, arg):
    batches = 10
    bs = 100
    dir_path = os.path.dirname(arg.resume)
    num_list = np.zeros((1001, 10))
    avg_list = np.zeros((1001, 10))
    for i in range(batches):
        samples,  track_num, track_avg = f.sample(batch_size=bs, tracing=True)
        track_num = np.array(track_num)
        track_avg = np.array(track_avg)
        num_list += track_num
        avg_list += track_avg
    avg_list /= batches
    draw_evolution(num_list, avg_list, dir_path)
