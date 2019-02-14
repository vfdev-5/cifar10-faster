import numpy as np

from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision.datasets.cifar import CIFAR10


__all__ = ["get_basic_train_test_loaders", "get_fast_train_test_loaders"]


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x, y = self.ds[index]
        return self.transform_fn(x), y


def get_basic_train_test_loaders(path, batch_size, num_workers, device):

    train_set_raw = CIFAR10(root=path, train=True, download=False)
    test_set_raw = CIFAR10(root=path, train=False, download=False)

    train_transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = TransformedDataset(train_set_raw, train_transforms)
    test_set = TransformedDataset(test_set_raw, test_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              pin_memory="cuda" in device, drop_last=True)

    test_loader = DataLoader(test_set, batch_size=batch_size * 4, num_workers=num_workers,
                             pin_memory="cuda" in device, drop_last=False)

    return train_loader, test_loader

################################################################################
#  Code below taken from : https://github.com/davidcpage/cifar10-fast.git
################################################################################

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

#####################
# # data augmentation
#####################


from collections import namedtuple


class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return C, self.h, self.w


class DynamicCrop(object):

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join(["{}={}".format(k, v)
                                          for k, v in self.__dict__.items()]))

    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return C, self.h, self.w


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class DynamicCutout(object):

    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               ", ".join(["{}={}".format(k, v)
                                          for k, v in self.__dict__.items()]))

    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0:y0 + self.h, x0:x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}


class Transform(object):
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


default_train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]


def get_fast_train_test_loaders(path, batch_size, num_workers, device,
                                train_transforms=default_train_transforms):

    train_set_raw = CIFAR10(root=path, train=True, download=False)
    test_set_raw = CIFAR10(root=path, train=False, download=False)

    print('Preprocessing training data')
    train_data = train_set_raw.train_data if hasattr(train_set_raw, "train_data") else train_set_raw.data
    train_labels = train_set_raw.train_labels if hasattr(train_set_raw, "train_labels") else train_set_raw.targets
    train_set = list(zip(transpose(normalise(pad(train_data, 4))), train_labels))
    print('Preprocessing test data')
    test_data = test_set_raw.test_data if hasattr(test_set_raw, "test_data") else test_set_raw.data
    test_labels = test_set_raw.test_labels if hasattr(test_set_raw, "test_labels") else test_set_raw.targets
    test_set = list(zip(transpose(normalise(test_data)), test_labels))

    train_set = Transform(train_set, train_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory="cuda" in device, drop_last=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0,
                             shuffle=False, pin_memory="cuda" in device, drop_last=False)

    return train_loader, test_loader
