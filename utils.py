import os
import torch
import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ExpUtils import AverageMeter


def sqrt(x):
    return int(t.sqrt(t.Tensor([x])))


def plot(p, x):
    return tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0)))


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)


class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


def cycle(loader):
    while True:
        for data in loader:
            yield data


def init_random(args, bs, im_sz=32, n_ch=3):
    return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })


def dataset_fn(args, train, transform):
    if args.dataset == "cifar10":
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.CIFAR10)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == "cifar100":
        args.n_classes = 100
        cls = dataset_with_indices(tv.datasets.CIFAR100)
        return cls(root=args.data_root, transform=transform, download=True, train=train)
    elif args.dataset == 'tinyimagenet':
        args.n_classes = 200
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    elif 'img' in args.dataset:
        args.n_classes = 1000
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    else:
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.SVHN)
        return cls(root=args.data_root, transform=transform, download=True, split="train" if train else "test")


def get_data(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(32),
             tr.RandomHorizontalFlip(),
             tr.ToTensor(),
             tr.Normalize(mean, std),
             ]
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize(mean, std),
         ]
    )
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        transform_train = tr.Compose([
            tr.RandomResizedCrop(224),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ])
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )
    elif 'img' in args.dataset:
        transform_train = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ])
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )
    else:
        transform_px = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean, std),
            ]
        )

    # get all training inds
    full_train = dataset_fn(args, True, transform_train)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid > args.n_classes:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)
    train_labeled_inds = train_inds

    if 'argment' in vars(args) and args.augment is False:
        transform_w = transform_px  # if args.dataset == 'cifar10' else transform_train
    else:
        transform_w = transform_train

    dset_train = DataSubset(dataset_fn(args, True, transform_px), inds=train_inds)
    dset_train_labeled = DataSubset(dataset_fn(args, True, transform_w), inds=train_labeled_inds)
    dset_valid = DataSubset(dataset_fn(args, True, transform_test), inds=valid_inds)

    num_workers = 0 if args.debug else 4
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    label_bs = 128
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=label_bs, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_train = cycle(dload_train)
    dset_test = dataset_fn(args, False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return dload_train, dload_train_labeled, dload_valid, dload_test


def get_train_test(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        transform_px = tr.Compose(
            [
                tr.Resize(256),
                tr.CenterCrop(224),
                tr.ToTensor(),
                tr.Normalize(mean, std),
             ]
        )
    else:
        transform_px = tr.Compose(
            [
                tr.ToTensor(),
                tr.Normalize(mean, std),
             ]
        )
    # get all training inds
    full_train = dataset_fn(args, True, transform_px)
    all_inds = list(range(len(full_train)))
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds)
    # seperate out validation set
    if args.n_valid > args.n_classes:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:]
    else:
        valid_inds, train_inds = [], all_inds
    train_inds = np.array(train_inds)

    dset_train = DataSubset(dataset_fn(args, True, transform_px), inds=train_inds)
    dset_valid = DataSubset(dataset_fn(args, True, transform_px), inds=valid_inds)
    dset_test = dataset_fn(args, False, transform_px)

    num_workers = 0 if args.debug else 4
    dload_train_labeled = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=num_workers, drop_last=False)
    return dload_train_labeled, dload_valid, dload_test


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def checkpoint(f, buffer, tag, args, device):
    if f is not None:
        f.cpu()
        ckpt_dict = {
            "model_state_dict": f.state_dict(),
            "replay_buffer": buffer,
        }
        t.save(ckpt_dict, os.path.join(args.save_dir, tag))
        f.to(device)
    else:
        ckpt_dict = {
            "model_state_dict": None,
            "replay_buffer": buffer,
        }
        t.save(ckpt_dict, os.path.join(args.save_dir, tag))


def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()


def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()


def eval_classification(f, dload, set_name, epoch, args=None, wlog=None):

    corrects, losses = [], []
    if args.n_classes >= 200:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

    for i, data in enumerate(dload):
        x, y = data[:2]
        x, y = x.to(args.device), y.to(args.device)
        logits = f.classify(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = nn.CrossEntropyLoss(reduction='none')(logits, y).detach().cpu().numpy()
        losses.extend(loss)
        if args.n_classes >= 200:
            acc1, acc5 = accuracy(logits, y, topk=(1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))
        else:
            correct = (logits.max(1)[1] == y).float().cpu().numpy()
            corrects.extend(correct)
        correct = (logits.max(1)[1] == y).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    if wlog:
        my_print = wlog
    else:
        my_print = print
    if args.n_classes >= 200:
        correct = top1.avg
        my_print("Epoch %d, %s loss %.5f, top1 acc %.4f, top5 acc %.4f" % (epoch, set_name, loss, top1.avg, top5.avg))
    else:
        correct = np.mean(corrects)
        my_print("Epoch %d, %s loss %.5f, acc %.4f" % (epoch, set_name, loss, correct))
    if args.vis:

        args.writer.add_scalar('%s/Loss' % set_name, loss, epoch)
        if args.n_classes >= 200:
            args.writer.add_scalar('%s/Acc_1' % set_name, top1.avg, epoch)
            args.writer.add_scalar('%s/Acc_5' % set_name, top5.avg, epoch)
        else:
            args.writer.add_scalar('%s/Accuracy' % set_name, correct, epoch)
    return correct, loss
