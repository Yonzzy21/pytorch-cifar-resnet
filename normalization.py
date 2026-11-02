'''Input normalization shared by main.py (train/val) and predict.py (test).'''
import math

import torchvision.transforms as transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class PerImageStandardize:
    '''Normalize each image by its own mean and std, over all pixels and channels together.

    Removes global brightness and contrast differences between images while keeping
    their color balance (per-channel statistics would erase it). The std is floored
    at 1/sqrt(num_values), as in tf.image.per_image_standardization, so flat images
    don't blow up.
    '''

    def __call__(self, x):
        return (x - x.mean()) / x.std().clamp_min(1.0 / math.sqrt(x.numel()))

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_normalize(data_cfg):
    '''data.normalization: 'dataset' (default, fixed CIFAR mean/std) or 'per_image'.'''
    if data_cfg.get('normalization', 'dataset') == 'per_image':
        return PerImageStandardize()
    return transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
