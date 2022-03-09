import os
import argparse
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset


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


def ds_grain(data_shape):
    step = 1
    if 1024 >= data_shape > 512:
        step = 16
    elif 512 >= data_shape > 256:
        step = 8
    elif 256 >= data_shape > 224:
        step = 4
    elif 224 >= data_shape >= 100:
        step = 4
    elif 100 > data_shape >= 64:
        step = 2
    return step


def ds_size(args):
    im_sz = 32
    if args.dataset == 'celeba256':
        im_sz = 256
    if args.dataset == 'img256' or args.dataset == 'imagenet':
        im_sz = 224
    if args.dataset == 'img128' or args.dataset == 'celeba128':
        im_sz = 128
    if args.dataset == 'img64' or args.dataset == 'tinyimagenet':
        im_sz = 64
    return im_sz


def downsample(x, data_shape=32):
    step = ds_grain(data_shape)
    down = torch.zeros([len(x), 3, data_shape // step, data_shape // step])

    for i in range(0, data_shape, step):
        for j in range(0, data_shape, step):
            v = x[:, :, i:i+step, j:j+step].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            ii, jj = i // step, j // step
            down[:, :, ii:ii+1, jj:jj+1] = v
    return down


def upsample(x, data_shape=32):
    step = ds_grain(data_shape)
    up = torch.zeros([len(x), 3, data_shape, data_shape])

    for i in range(0, data_shape, step):
        for j in range(0, data_shape, step):
            ii, jj = i // step, j // step
            up[:, :, i:i+step, j:j+step] = x[:, :, ii:ii+1, jj:jj+1]
    return up


def category_mean(train_loader, arg):
    import time
    start = time.time()
    im_test, target_test = [], []
    for im, targ, idx in train_loader:
        im_test.append(im.detach())
        target_test.append(targ.detach())
    im_test, target_test = torch.cat(im_test), torch.cat(target_test)
    print(im_test.shape)

    # conditionals = []
    im_sz = ds_size(arg)
    imc = im_test
    imc = downsample(imc, im_sz).view(len(imc), -1)
    mean = imc.mean(dim=0)
    sub = imc - mean.unsqueeze(dim=0)
    cov = sub.t() @ sub / len(imc)
    print(time.time() - start)
    print(mean.shape, cov.shape)
    torch.save(mean, './%s_mean_one.pt' % arg.dataset)
    torch.save(cov, './%s_cov_one.pt' % arg.dataset)


def dataset_fn(args, train, transform):
    if 'celeba' in args.dataset:
        args.n_classes = 10
        cls = dataset_with_indices(tv.datasets.ImageFolder)
        return cls(root=os.path.join(args.data_root, 'train' if train else 'val'), transform=transform)
    
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
        assert False, "%s is not contained" % args.dataset


def get_train(args):
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_px = tr.Compose(
        [
            tr.ToTensor(),
            tr.Normalize(mean, std),
        ]
    )
    full_train = dataset_fn(args, True, transform_px)
    all_inds = list(range(len(full_train)))
    dset_train = DataSubset(dataset_fn(args, True, transform_px), inds=all_inds)
    num_workers = 0 if args.debug else 4
    if ',' in args.gpu_id:
        num_workers *= len(args.gpu_id.split(','))
    loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models")
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "svhn", "cifar100", 'tinyimagenet', 'img32', 'img128', 'img256', 'celeba128'])
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--gpu-id", type=str, default="../data")
    args = parser.parse_args()
    args.seed = 1
    args.batch_size = 100
    args.debug = False
    args.n_valid = 0
    if 'celeba' in args.dataset:
        data_loader = get_train(args)
    else:
        data_loader, _, test_loader = get_train_test(args)

    if 'img' in args.dataset:
        data_loader = test_loader
    category_mean(data_loader, args)
