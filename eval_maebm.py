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
import torch
import torch as t, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv, torchvision.transforms as tr
import argparse
from ExpUtils import *
from models.jem_models import F, CCF
from utils import plot, disable_running_stats
from one_center import ds_size, ds_grain, upsample, downsample
from Task.eval_buffer import cond_is_fid

# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3
n_classes = 10
correct = 0
print = wlog
conditionals = []


def init_random(arg, bs):
    global conditionals
    n_ch = 3
    im_sz = ds_size(arg)
    step = ds_grain(im_sz)
    size = [3, im_sz // step, im_sz // step]
    new = t.zeros(bs, n_ch, im_sz // step, im_sz // step)
    for i in range(bs):
        index = np.random.randint(len(conditionals))
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
    return upsample(t.clamp(new, -1, 1), im_sz)


def init_from_centers(arg):
    global conditionals
    from torch.distributions.multivariate_normal import MultivariateNormal
    bs = arg.buffer_size
    if arg.dataset == 'celeba128':
        size = [3, 28, 28]
    else:
        size = [3, 32, 32]
    if arg.dataset == 'cifar_test':
        arg.dataset = 'cifar10'
    centers = t.load('./%s_mean_one.pt' % arg.dataset)
    covs = t.load('./%s_cov_one.pt' % arg.dataset)

    mean = centers.to(arg.device)
    cov = covs.to(arg.device)
    dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * t.eye(int(np.prod(size))).to(arg.device))
    conditionals.append(dist)

    return init_random(arg, bs)


def sample_p_0(replay_buffer, bs, y=None, arg=None):
    if len(replay_buffer) == 0:
        return init_random(arg, bs), []
    buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // n_classes
    if buffer_size > bs:
        inds = t.randint(0, buffer_size, (bs,))
    else:
        inds = t.arange(0, bs)
    # if cond, convert inds to class conditional inds
    if y is not None:
        inds = y.cpu() * buffer_size + inds
        assert not arg.uncond, "Can't drawn conditional samples without giving me y"
    buffer_samples = replay_buffer[inds]
    random_samples = init_random(arg, bs)
    choose_random = (t.rand(bs) < arg.reinit_freq).float()[:, None, None, None]
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(arg.device), inds


def sample_q(f, replay_buffer, y=None, n_steps=10, arg=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """

    f.train()
    # get batch size
    bs = arg.batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(replay_buffer, bs=bs, y=y, arg=arg)
    x_k = t.autograd.Variable(init_sample, requires_grad=True).to(arg.device)
    # sgld

    eps = 1
    for it in range(n_steps):
        energies = f(x_k, y=y)[0]
        e_x = energies.sum()
        eta = t.autograd.grad(e_x, [x_k], retain_graph=True)[0]
        tmp_inp = x_k.data + eta * arg.sgld_lr
        x_k.data = tmp_inp.data

        if arg.sgld_std > 0.0:
            x_k.data += arg.sgld_std * t.randn_like(x_k)
        x_k.data = t.clamp(x_k.data, -1, 1)

    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        replay_buffer[buffer_inds] = final_samples.cpu()
    return final_samples


def uncond_samples(f, arg, device, save=True):

    init_from_centers(arg)
    replay_buffer = init_random(arg, arg.buffer_size)
    for i in range(arg.n_sample_steps):
        samples = sample_q(f, replay_buffer, n_steps=arg.n_steps, arg=arg)
        if i % arg.print_every == 0 and save:
            plot('{}/samples_{}.png'.format(arg.save_dir, i), samples)
        print(i)
    return replay_buffer


def cond_samples(f, replay_buffer, arg, device, fresh=False):

    if fresh:
        replay_buffer = uncond_samples(f, arg, device, save=True)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    for i in range(n_it):
        x = replay_buffer[i * 100: (i + 1) * 100].to(device)
        y = f.classify(x)[0].max(1)[1]
        all_y.append(y)

    all_y = t.cat(all_y, 0)
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])
    for i in range(100):
        this_im = []
        for l in range(10):
            this_l = each_class[l][i * 10: (i + 1) * 10]
            this_im.append(this_l)
        this_im = t.cat(this_im, 0)
        if this_im.size(0) > 0:
            plot('{}/samples_{}.png'.format(arg.save_dir, i), this_im)
        print(i)


def best_samples(f, replay_buffer, arg, device, fresh=False):
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))

    if fresh:
        replay_buffer = uncond_samples(f, arg, device, save=True)
    n_it = replay_buffer.size(0) // 100
    all_y = []
    all_px = []
    probs = []
    with t.no_grad():
        for i in tqdm(range(n_it)):
            x = replay_buffer[i * 100: (i + 1) * 100].to(device)
            logits, _ = f.classify(x)
            y = logits.max(1)[1]
            px = logits.logsumexp(1)
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)
            all_px.append(px)

    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    print(probs.min().item())
    print((probs < 0).sum().item())
    all_px = t.cat(all_px, 0)
    print("%f %f %f" % (probs.mean().item(), probs.max().item(), probs.min().item()))
    each_class = [replay_buffer[all_y == l] for l in range(10)]
    each_class_probs = [probs[all_y == l] for l in range(10)]
    each_class_px = [all_px[all_y == l] for l in range(10)]
    print([len(c) for c in each_class])

    new_buffer = []
    ratio = abs(arg.ratio)
    for c in range(10):
        each_probs = each_class_probs[c]
        # select
        each_metric = each_class_px[c]
        # each_metric = each_class_probs[c]

        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        if arg.ratio > 0:
            topks = t.topk(each_metric, topk, largest=arg.ratio > 0)
            index_list = topks[1]
        else:
            topks = t.topk(each_metric, topk, largest=arg.ratio > 0)
            index_list = topks[1]

        print('P(x) min %.3f max %.3f' % (-each_metric[index_list].max().item(), -each_metric[index_list].min().item()))
        print('Prob(y|x) max %.3f min %.3f' % (each_probs[index_list].max().item(), each_probs[index_list].min().item()))
        images = each_class[c][index_list]
        new_buffer.append(images)
        plot('{}/topk_{}.png'.format(arg.save_dir, c), images)

    replay_buffer = t.cat(new_buffer, 0)
    print(replay_buffer.shape)


def new_samples(f, arg, device, save=True):
    replay_buffer = init_from_centers(arg)
    plot('{}/samples_0.png'.format(arg.save_dir), replay_buffer)
    for i in range(arg.n_sample_steps):
        samples = sample_q(f, replay_buffer, n_steps=arg.n_steps, arg=arg)

        if i % 50 == 0:
            print(i)
        if (i + 1) % arg.print_every == 0 and save:
            print(i)
            plot('{}/samples_{}.png'.format(arg.save_dir, i + 1), samples)
            if arg.print_every < 50:
                continue
            # inc_score, std, fid = cond_is_fid(f, replay_buffer, args, device, ratio=100000)
            # print("sample more steps %d with %d" % (i, args.n_steps))
            # print("Inception score of {} with std of {}".format(inc_score, std))
            # print("FID of score {}".format(fid))
    inc_score, std, fid = cond_is_fid(f, replay_buffer, arg, device, ratio=100000)
    print("final sample more steps %d with %d" % (arg.n_sample_steps, arg.n_steps))
    print("Inception score of {} with std of {}".format(inc_score, std))
    print("FID of score {}".format(fid))


def logp_hist(f, arg, device):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.switch_backend('agg')
    def sample(x, n_steps=arg.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples
    def grad_norm(x):
        x_k = t.autograd.Variable(x, requires_grad=True)
        f_prime = t.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)
    def score_fn(x):
        if arg.score_fn == "px":
            return f(x)[0].detach().cpu()
        elif arg.score_fn == "py":
            logits, _ = f.classify(x)
            return nn.Softmax()(logits).max(1)[0].detach().cpu()
        else:
            return f.classify(x).max(1)[0].detach().cpu()
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + arg.sigma * t.randn_like(x)]
    )
    datasets = {
        "cifar10": tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False),
        "svhn": tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test"),
        "cifar100": tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False),
        "celeba": tv.datasets.CelebA(root="../../data", download=True, split="test",
                                          transform=tr.Compose([tr.Resize(32),
                                                                tr.ToTensor(),
                                                                tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                                lambda x: x + arg.sigma * t.randn_like(x)]))
    }

    score_dict = {}
    num_workers = 0 if arg.debug else 4
    for dataset_name in arg.datasets:
        print(dataset_name)
        dataset = datasets[dataset_name]
        dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=num_workers, drop_last=False)
        this_scores = []
        for x, _ in dataloader:
            x = x.to(device)
            scores = score_fn(x)
            this_scores.extend(scores.numpy())
        score_dict[dataset_name] = this_scores

    colors = ['green', 'red']
    for i, (name, scores) in enumerate(score_dict.items()):
        plt.hist(scores, label=name, bins=100, alpha=.5, color=colors[i])
    plt.legend(loc='upper left')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(arg.save_dir + "/jem_%s_logp.pdf" % arg.datasets[1], bbox_inches='tight', pad_inches=0.0)


def OODAUC(f, arg, device):
    print("OOD Evaluation")

    def grad_norm(x):
        x_k = t.autograd.Variable(x, requires_grad=True)
        f_prime = t.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
        grad = f_prime.view(x.size(0), -1)
        return grad.norm(p=2, dim=1)

    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + arg.sigma * t.randn_like(x)]
    )

    num_workers = 0 if arg.debug else 4
    dset_real = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)

    if arg.ood_dataset == "svhn":
        dset_fake = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    elif arg.ood_dataset == "cifar_100":
        dset_fake = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False)
    elif arg.ood_dataset == "celeba":
        dset_fake = tv.datasets.CelebA(root="../../data", download=True, split="test",
                                       transform=tr.Compose([tr.Resize(32),
                                                             tr.ToTensor(),
                                                             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
                                                             lambda x: x + arg.sigma * t.randn_like(x)]))
    else:
        dset_fake = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    dload_fake = DataLoader(dset_fake, batch_size=100, shuffle=True, num_workers=num_workers, drop_last=False)
    real_scores = []
    print("Real scores...")

    def score_fn(x):
        if arg.score_fn == "px":
            return f(x)[0].detach().cpu()
        elif arg.score_fn == "py":
            logits, _ = f.classify(x)
            return nn.Softmax()(logits).max(1)[0].detach().cpu()
        else:
            return -grad_norm(x).detach().cpu()

    for x, _ in dload_real:
        x = x.to(device)
        scores = score_fn(x)
        real_scores.append(scores.numpy())
    fake_scores = []
    print("Fake scores...")
    if arg.ood_dataset == "cifar_interp":
        last_batch = None
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            if i > 0:
                x_mix = (x + last_batch) / 2 + arg.sigma * t.randn_like(x)
                scores = score_fn(x_mix)
                fake_scores.append(scores.numpy())
            last_batch = x
    else:
        for i, (x, _) in enumerate(dload_fake):
            x = x.to(device)
            scores = score_fn(x)
            fake_scores.append(scores.numpy())
    real_scores = np.concatenate(real_scores)
    fake_scores = np.concatenate(fake_scores)
    real_labels = np.ones_like(real_scores)
    fake_labels = np.zeros_like(fake_scores)
    import sklearn.metrics
    scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([real_labels, fake_labels])
    score = sklearn.metrics.roc_auc_score(labels, scores)
    print('OOD scores %f of %s between cifar10 and %s using %s' % (score, arg.score_fn, args.ood_dataset, args.load_path))


def test_clf(f, arg, device):
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + t.randn_like(x) * arg.sigma]
    )

    def sample(x, n_steps=arg.n_steps):
        x_k = t.autograd.Variable(x.clone(), requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data += f_prime + 1e-2 * t.randn_like(x_k)
        final_samples = x_k.detach()
        return final_samples

    if arg.dataset == "cifar_train":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=True)
    elif arg.dataset == "cifar_test":
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    elif arg.dataset == "cifar100_train":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=True)
    elif arg.dataset == "cifar100_test":
        dset = tv.datasets.CIFAR100(root="../data", transform=transform_test, download=True, train=False)
    elif arg.dataset == "svhn_train":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="train")
    elif arg.dataset == "svhn_test":
        dset = tv.datasets.SVHN(root="../data", transform=transform_test, download=True, split="test")
    elif arg.dataset == 'stl10':
        dset = tv.datasets.STL10(root="../data", transform=transform_test, download=True, split="test")
    else:
        dset = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)

    num_workers = 0 if arg.debug else 4
    dload = DataLoader(dset, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)

    corrects, losses, pys, preds = [], [], [], []
    for x_p_d, y_p_d in tqdm(dload):
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        if arg.n_steps > 0:
            x_p_d = sample(x_p_d)
        with t.no_grad():
            logits, _ = f.classify(x_p_d)
        py = nn.Softmax(dim=1)(logits).max(1)[0].detach().cpu().numpy()
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y_p_d).cpu().detach().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
        pys.extend(py)
        preds.extend(logits.max(1)[1].cpu().numpy())

    loss = np.mean(losses)
    correct = np.mean(corrects)
    print('loss %.5g,  accuracy: %g%%' % (loss, correct * 100))
    return correct


def calibration(f, arg, device):
    from Task.calibration import reliability_diagrams
    from Task.calibration import ECELoss
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + arg.sigma * t.randn_like(x)]
    )

    num_workers = 0 if arg.debug else 4
    dset_real = tv.datasets.CIFAR10(root="../data", transform=transform_test, download=True, train=False)
    dload_real = DataLoader(dset_real, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    f.eval()
    real_scores = []
    labels = []
    pred = []
    ece_com = ECELoss(20)
    ece = 0
    c = 0
    logits_l = []
    for x, y in dload_real:
        x = x.to(device)
        labels.append(y.numpy())
        logits, _ = f.classify(x)
        logits_l.append(logits.detach())
        scores = nn.Softmax(dim=1)(logits).max(dim=1)[0].detach().cpu()
        preds = nn.Softmax(dim=1)(logits).argmax(dim=1).detach().cpu()
        real_scores.append(scores.numpy())
        pred.append(preds.numpy())
    logits_l = torch.cat(logits_l)
    temps = torch.LongTensor(np.concatenate(labels))
    ece = ece_com(logits_l, temps.to(device)).item()
    print("On Calibration of Modern Neural Networks code result: %f" % ece)
    real_scores = np.concatenate(real_scores)
    labels = np.concatenate(np.array(labels))
    pred = np.concatenate(pred)
    print(len(real_scores))
    # print(pred.shape)

    reliability_diagrams(list(pred), list(labels), list(real_scores), bin_size=0.05, title="Accuracy: %.2f%%" % (100.0 * correct), args=arg)


def get_h_and_e(model, inputs):
    e = []
    hl = []
    model.train()
    for i in range(0, len(inputs), 100):
        input_x = inputs[i:i+100].to(args.device)
        with t.no_grad():
            fp, h, logits = model(input_x)
            e.extend(fp.cpu().numpy())
            hl.extend(h.cpu().numpy())
    return e, hl


def plot_tsne(f, replay_buffer, arg):
    init_from_centers(arg)
    x0 = init_random(arg, 1000)
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform = tr.Compose(
        [
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ]
    )
    train_set = tv.datasets.CIFAR10(root='./data', transform=transform, download=True, train=True)
    torch.manual_seed(1234)
    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=2, drop_last=True)
    for x, y in train_loader:
        break

    f = f.train()
    disable_running_stats(f)
    e_ujem_d, h_ujem_d = get_h_and_e(f, x)
    e_ujem_n, h_ujem_n = get_h_and_e(f, replay_buffer)
    e_ujem_0, h_ujem_0 = get_h_and_e(f, x0)

    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=False)

    ax2.violinplot([e_ujem_d, e_ujem_n, e_ujem_0])
    # ax.violinplot([fp_ebm, rfp_ebm, fp_jem, rfp_jem])
    ax2.set_xticklabels(['', '$E(x^+)$', '', '$E(x^-)$', '', '$E(x_0)$'])
    ax2.set_title('MA-JEM')
    fig.tight_layout()
    fig.savefig(f'{arg.save_dir}/energy_hist.png')
    plt.close()

    all_h = np.concatenate([h_ujem_d, h_ujem_0, h_ujem_n[:1000]])
    print(all_h.shape)
    from sklearn import manifold
    vis_method = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = vis_method.fit_transform(all_h)
    # color = all_y
    fig = plt.figure(figsize=(8, 6))
    k = 1000
    s = 20
    plt.scatter(Y[2000:2000 + k, 0], Y[2000:2000 + k, 1], c='orange', marker='o', cmap=plt.cm.Spectral, s=s, label='MAJEM $x^0$')
    plt.scatter(Y[1000:1000 + k, 0], Y[1000:1000 + k, 1], c='g', marker='o', cmap=plt.cm.Spectral, s=s, label='MAJEM $x^-$')
    plt.scatter(Y[:0 + k, 0], Y[:0 + k, 1], c='brown', marker='o', cmap=plt.cm.Spectral, s=s, label='MAJEM $x^+$')

    lgnd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, fancybox=True)
    # for handle in lgnd.legendHandles:
    #     handle.set_sizes([25.0])
    plt.axis('tight')
    fig.savefig(f'{arg.save_dir}/tsne.png', dpi=200)
    plt.close()


def main(arg):
    global correct
    set_file_logger(logger, arg)
    arg.save_dir = arg.dir_path
    print(' '.join(sys.argv))
    print(arg.dir_path)

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    arg.device = device

    model_cls = F if arg.uncond else CCF
    f = model_cls(arg.depth, arg.width, arg.norm, n_classes=arg.n_classes, model=arg.model, args=arg)
    print(f"loading model from {arg.load_path}")

    # load em up
    ckpt_dict = t.load(arg.load_path)
    f.load_state_dict(ckpt_dict["model_state_dict"])
    replay_buffer = ckpt_dict["replay_buffer"]

    f = f.to(device)
    f.eval()

    if arg.eval == 'quality':
        from Task.quality_analysis import qualitative_analysis
        qualitative_analysis(f, replay_buffer, buffer_path=arg.load_path, dataset=args.dataset)

    if arg.eval == "OOD":
        OODAUC(f, arg, device)

    if arg.eval == "cali":
        correct = test_clf(f, arg, device)
        calibration(f, arg, device)

    if arg.eval == "test_clf":
        test_clf(f, arg, device)

    if arg.eval == "cond_samples":
        cond_samples(f, replay_buffer, arg, device, arg.fresh_samples)

    if arg.eval == 'gen':
        new_samples(f, arg, device, save=True)

    if arg.eval == "fid":
        print('eval is, fid')
        inc_score, std, fid = cond_is_fid(f, replay_buffer, args, device, ratio=args.ratio, eval='all')
        print('IS {}, fid {}'.format(inc_score, fid))

    if arg.eval == "uncond_samples":
        uncond_samples(f, arg, device)

    if arg.eval == "logp_hist":
        logp_hist(f, arg, device)

    if arg.eval == 'tsne':
        plot_tsne(f, replay_buffer, arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MA-EBM/JEM")
    parser.add_argument("--eval", default="OOD", type=str,
                        choices=["uncond_samples", "cond_samples", "best_samples", "logp_hist", "OOD", "test_clf", "fid", "cali", "gen", "quality", 'tsne'])
    parser.add_argument("--score_fn", default="px", type=str, choices=["px", "py", "pxgrad"], help="For OODAUC, chooses what score function we use.")
    parser.add_argument("--ood_dataset", default="svhn", type=str, choices=["svhn", "cifar_interp", "cifar_100", "celeba"], help="Chooses which dataset to compare against for OOD")
    parser.add_argument("--dataset", default="cifar_test", type=str, choices=["cifar_train", "cifar_test", "svhn_test", "svhn_train", "cifar100_test"],
                        help="Dataset to use when running test_clf for classification accuracy")
    parser.add_argument("--datasets", nargs="+", type=str, default=[], help="The datasets you wanna use to generate a log p(x) histogram")
    # optimization
    parser.add_argument("--batch_size", type=int, default=64)
    # regularization
    parser.add_argument("--sigma", type=float, default=0)
    # network
    parser.add_argument("--norm", type=str, default="batch", choices=[None, "none", "norm", "batch", "instance", "layer", "act"])
    parser.add_argument("--init", type=str, default='i', help='r random, i inform')
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--uncond", action="store_true")
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--reinit_freq", type=float, default=.0)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=0)

    parser.add_argument("--model", type=str, default='wrn')
    parser.add_argument("--ratio", type=float, default=900)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='jem_eval')
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--n_sample_steps", type=int, default=100)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true")
    parser.add_argument("--fresh_samples", action="store_true", help="If set, then we generate a new replay buffer from scratch for conditional sampling, Will be much slower.")
    parser.add_argument("--gpu-id", type=str, default="")

    args = parser.parse_args()
    assert args.eval != 'gen' or args.n_steps != 0

    auto_select_gpu(args)
    init_debug(args)
    run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
    if args.save_dir == 'jem_eval':
        # by default to eval the model
        args.dir_path = args.load_path + "_eval_%s_%s" % (args.eval, run_time)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.n_classes = 100 if "cifar100" in args.dataset else 10
    main(args)
    print(args.save_dir)
