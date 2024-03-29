# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import copy

import numpy as np
import torch
import torch as t
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from ExpUtils import *
from models.jem_models import get_model_and_buffer
from Task.eval_buffer import cond_is_fid
from Task.tSNE_trans import tsne_trans
from utils import eval_classification, checkpoint, get_data, plot, disable_running_stats, enable_running_stats

t.set_num_threads(2)
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
inner_his = []
conditionals = []


def init_random(args, bs):
    global conditionals
    n_ch = 3
    size = [3, 32, 32]
    im_sz = 32
    new = t.zeros(bs, n_ch, im_sz, im_sz)
    for i in range(bs):
        index = np.random.randint(len(conditionals))
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
    return t.clamp(new, -1, 1).cpu()


def init_random_uniform(args, bs):
    im_sz = 32
    n_ch = 3
    if args.dataset == 'tinyimagenet':
        im_sz = 64
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        im_sz = 224

    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def sample_p_0(replay_buffer, bs, y=None):
    if len(replay_buffer) == 0:
        return init_random(args, bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes
    inds = t.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
    buffer_samples = replay_buffer[inds]
    if args.init == 'gm':
        random_samples = init_random(args, bs)
    else:
        random_samples = init_random_uniform(args, bs)
    r_idx = t.rand(bs) < args.reinit_freq
    choose_random = r_idx.float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    new_step = t.zeros(bs)
    return samples.to(args.device), inds


def sample_q(f, replay_buffer, y=None, n_steps=10, args=None, save=True):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    # Batch norm uses train status
    # f.eval()
    # get batch size
    bs = args.batch_size if y is None else y.size(0)
    # generate initial samples and buffer index of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y)
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    # sgld

    prev_e = None
    for it in range(n_steps):
        energies, h, _ = f(x_k, y=y)
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        tmp_inp = x_k + t.clamp(eta, -1, 1) * args.sgld_lr

        x_k.data = tmp_inp.data

        if args.sgld_std > 0.0:
            x_k.data += args.sgld_std * t.randn_like(x_k)
        x_k.data = t.clamp(x_k.data, -1, 1)

    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0 and save:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def init_from_centers(args):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = args.buffer_size
    if args.dataset == 'tinyimagenet':
        size = [3, 64, 64]
    else:
        size = [3, 32, 32]
    centers = t.load('./%s_mean_one.pt' % args.dataset)
    covs = t.load('./%s_cov_one.pt' % args.dataset)

    buffer = []
    # for i in range(args.n_classes):
    mean = centers.to(args.device)
    cov = covs.to(args.device)
    dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(args.device))
    buffer.append(dist.sample((bs, )).view([bs] + size).cpu())
    conditionals.append(dist)
    return t.clamp(t.cat(buffer), -1, 1)


def main(arg):

    np.random.seed(arg.seed)
    t.manual_seed(arg.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(arg.seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    arg.device = device

    # datasets
    dload_train, dload_train_labeled, dload_valid, dload_test = get_data(arg)

    f, replay_buffer = get_model_and_buffer(arg, device)
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in f.parameters()) / 1e6))

    replay_buffer = init_from_centers(arg)
    if arg.px <= 0:
        replay_buffer = None

    # optimizer
    params = f.class_output.parameters() if arg.clf_only else f.parameters()
    if arg.optimizer == "adam":
        optim = t.optim.Adam(params, lr=arg.lr, betas=[.9, .999], weight_decay=arg.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=arg.lr, momentum=.9, weight_decay=arg.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0
    n_steps = arg.n_steps
    # trace learning rate
    new_lr = arg.lr
    prev_is = 0

    fid_init = {'celeba128': 395.3, 'img32': 415.6, 'cifar10': 220.3, 'cifar100': 208.9}
    if arg.dataset in fid_init:
        fid = fid_init[arg.dataset]
    else:
        _, _, fid = cond_is_fid(f, replay_buffer, arg, device, ratio=0.9, eval='fid')
    prev_fid = fid
    print('IS {}, fid {}'.format(0, fid))
    arg.writer.add_scalar('Gen/FID', fid, -1)

    bench_train = torch.load('temp/%s_train_100.pt' % args.dataset)
    train_x = bench_train['x'].to(device)
    train_y = bench_train['y']
    rand_indx = torch.randperm(len(train_x))
    train_x = train_x[rand_indx]
    train_y = train_y[rand_indx]
    print(train_x.shape)
    x0 = init_random(arg, 200).to(device)

    for epoch in range(arg.n_epochs):
        if epoch in arg.decay_epochs:
            for param_group in optim.param_groups:
                new_lr = param_group['lr'] * arg.decay_rate
                param_group['lr'] = new_lr
            print("Decaying lr to {}".format(new_lr))

        for i, (x_lab, y_lab, idx) in tqdm(enumerate(dload_train_labeled)):
            if cur_iter <= arg.warmup_iters:
                lr = arg.lr * cur_iter / float(arg.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            x_p_d, y_p_lab, x_idx = dload_train.__next__()
            x_p_d = x_p_d.to(device)
            y_p_lab = y_p_lab.to(device)
            x_p_d += arg.sigma * t.randn_like(x_p_d)

            loss = 0.
            if arg.px > 0:  # maximize log p(x)
                fp_all, h, logits = f(x_p_d)
                fp = fp_all.mean()

                if arg.class_cond_p_x_sample:
                    assert not arg.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, arg.n_classes, (arg.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, y=y_q, n_steps=n_steps, args=arg)
                else:
                    x_q = sample_q(f, replay_buffer, n_steps=n_steps, args=arg)  # sample from log-sumexp

                fq_all, h_x_neg, _ = f(x_q)
                fq = fq_all.mean()

                l_p_x = -(fp - fq)
                # l_p_x += arg.l2_coeff * (fp ** 2 + (fq_all ** 2).mean())
                # l_p_x += arg.l2_coeff * (torch.mean(fp_all ** 2) + torch.mean(fq ** 2))
                l_p_x += arg.l2_coeff * (torch.mean(fp_all ** 2) + fq ** 2)

                loss += arg.px * l_p_x

                # break if the loss diverged...easier for poppa to run experiments this way
                if loss.abs().item() > 1e5:
                    print("BAD BOIIIIIIIIII")
                    print("min {:>4.3f} max {:>5.3f}".format(x_q.min().item(), x_q.max().item()))
                    plot('{}/diverge_{}_{:>06d}.png'.format(arg.save_dir, epoch, i), x_q)
                    return

                if cur_iter % 100 == 0:
                    arg.writer.add_scalar('Train/NegData', fq.item(), cur_iter)
                    arg.writer.add_scalar('Train/PosData', fp.item(), cur_iter)
                    arg.writer.add_scalar('Train/CD', (fp - fq).item(), cur_iter)

                if cur_iter % arg.print_every == 0:
                    print('{} P(x) | {}:{:>d} f(x_p_d)={:>6.4f} f(x_q)={:>6.4f} d={:>6.4f}'.format(arg.pid, epoch, i, fp, fq, fp - fq))
                    if args.print_every == 1:
                        print('Neg min={:>6.4f}, max={:>6.4f}, Pos min={:>6.4f}, max={:>6.4f}'.format(fq_all.min().item(), fq_all.max().item(), fp_all.min().item(), fp_all.max().item()))
                        print('Neg avg={:>6.4f}, Pos avg={:>6.4f}'.format(fq_all.mean().item(), fp_all.mean().item()))

            if arg.pyx > 0:  # maximize log p(y | x)
                logits, h = f.classify(x_lab)
                ce_loss = nn.CrossEntropyLoss()(logits, y_lab)
                loss += ce_loss
                if cur_iter % arg.print_every == 0:
                    acc = (logits.max(1)[1] == y_lab).float().mean()
                    print('{} P(y|x) {}:{:>d} loss CE:{:>5.4f}, acc={:>5.4f}'.format(arg.pid, epoch, cur_iter, ce_loss.item(), acc.item()))

            optim.zero_grad()
            loss.backward()
            optim.step()
            cur_iter += 1

            if cur_iter % 100 == 0 and arg.px > 0:
                plot('{}/samples/x_q_y{}_{:>06d}.png'.format(arg.save_dir, epoch, i), x_q[:64])

        if epoch % arg.ckpt_every == 0 and arg.px > 0:
            checkpoint(f, replay_buffer, f'ckpt_{epoch}.pt', arg, device)

        with torch.no_grad():
            disable_running_stats(f)
            fp_all, h_x_pos, z_pos = f(train_x[:200])
            f0_all, h_x_0, z_0 = f(x0)
            enable_running_stats(f)
        if arg.px > 0:
            # tSNE visualization
            feats = torch.cat([h_x_pos, h_x_0, h_x_neg]).detach().cpu().numpy()
        else:
            feats = torch.cat([h_x_pos, h_x_0]).detach().cpu().numpy()
        np.save(arg.save_dir + f'/features_{epoch}.npy', feats)
        tsne_trans(feats, epoch, arg)

        if epoch % arg.eval_every == 0:
            f.eval()
            if arg.pyx > 0:
                with t.no_grad():
                    correct, loss = eval_classification(f, dload_test, 'Test', epoch, arg, wlog)

                if correct > best_valid_acc:
                    best_valid_acc = correct
                    print("Epoch {} Best Valid!: {}".format(epoch, correct))
                    checkpoint(f, replay_buffer, "best_valid_ckpt.pt", arg, device)

            if arg.px > 0:
                if not arg.no_fid and epoch % 5 == 0 or epoch == arg.n_epochs - 1:
                    # try:
                    print('eval is, fid')
                    ratio = 0.1 if arg.pyx else 0.9
                    inc_score, std, fid = cond_is_fid(f, replay_buffer, arg, device, ratio=ratio, eval='all')
                    if fid > 0:
                        prev_fid = fid
                    else:
                        fid = prev_fid
                    if inc_score > 0:
                        prev_is = inc_score
                    else:
                        inc_score = prev_is
                    print('IS {}, fid {}'.format(inc_score, fid))
                    arg.writer.add_scalar('Gen/IS', inc_score, epoch)
                    arg.writer.add_scalar('Gen/FID', fid, epoch)
                    # except BaseException:
                    #     pass
            f.train()
        checkpoint(f, replay_buffer, "last_ckpt.pt", arg, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Manifold-Aware Energy Based Models")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100", 'tinyimagenet', 'img32'])
    parser.add_argument("--data_root", type=str, default="./data")
    # optimization
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[60, 120, 180], help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting
    parser.add_argument("--px", type=float, default=1.)
    parser.add_argument("--pyx", type=float, default=1.)
    parser.add_argument("--l2_coeff", type=float, default=0)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0, help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    # network
    parser.add_argument("--norm", type=str, default="batch", choices=[None, "none", "batch", "instance", "layer", "act"], help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=10, help="number of steps of SGLD per iteration, 20 works for PCD")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=0.05)

    # SGLD
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=0)
    parser.add_argument("--init", type=str, default='gm', help='gm: Gaussian Mixture,  u: uniform')

    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--dir_path", type=str, default='./experiment')
    parser.add_argument("--log_dir", type=str, default='./runs')
    parser.add_argument("--log_arg", type=str, default='MAE-pyx-n_steps-sgld_lr')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--no_fid", action="store_true", help="If set, evaluate FID/Inception Score")
    parser.add_argument("--no_wandb", action="store_true", help="If set, evaluate FID/Inception Score")
    parser.add_argument("--n_valid", type=int, default=0)

    parser.add_argument("--eps", type=float, default=1, help="eps bound")
    parser.add_argument("--model", type=str, default='wrn')
    parser.add_argument("--novis", action="store_true", help="")
    parser.add_argument("--debug", action="store_true", help="")
    parser.add_argument("--exp_name", type=str, default="MAEBM", help="exp name, for description")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, default="0")
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()
    init_env(args, logger)
    args.save_dir = args.dir_path
    os.makedirs('{}/samples'.format(args.dir_path))
    print = wlog
    print(' '.join(sys.argv))
    print(args.dir_path)
    main(args)
    print(args.dir_path)
