'''Compare trained checkpoints on the held-out validation split (and shifted copies of it).

For each model: plain and flip-TTA accuracy on the clean val set and on copies of it with
mild brightness/contrast, blur and JPEG shifts. For each pair: an exact McNemar test, which
uses only the images where exactly one of the two models is right (more sensitive than
comparing accuracies, because both models are scored on the same images).

Usage: python compare_models.py --configs configs/a.yaml configs/b.yaml [...]
Each config's checkpoint (checkpoint.dir/filename) must exist locally.
'''
import argparse
import io
import itertools
import math
import os

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageFilter

import models
from dataset import CIFAR10Kaggle
from normalization import get_normalize


def brightness_contrast(img):
    # Roughly the measured train -> test shift: brighter, higher contrast, more clipped pixels
    x = np.asarray(img, dtype=np.float32) / 255
    x = np.clip((x - x.mean()) * 1.1 + x.mean() + 0.03, 0, 1)
    return Image.fromarray((x * 255).round().astype(np.uint8))


def jpeg(img, quality=70):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return Image.open(buf).convert('RGB')


SHIFTS = {
    'clean': lambda img: img,
    'bright+contrast': brightness_contrast,
    'blur': lambda img: img.filter(ImageFilter.GaussianBlur(0.7)),
    'jpeg70': jpeg,
}


def load(config_path):
    cfg = yaml.safe_load(open(config_path))
    model_cfg = cfg['model']
    net = getattr(models, model_cfg.get('name', 'CustomResNet'))(**{k: v for k, v in model_cfg.items() if k != 'name'})
    ckpt = torch.load(os.path.join(cfg['checkpoint']['dir'], cfg['checkpoint']['filename']),
                      map_location='cpu', weights_only=True)
    net.load_state_dict({k.removeprefix('module.'): v for k, v in ckpt['net'].items()})
    return cfg, net.eval()


def mcnemar_p(b, c):
    # Exact two-sided McNemar test: b, c = images where only model 1 / only model 2 is right
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--configs', nargs='+', required=True)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count())

    loaded = {os.path.splitext(os.path.basename(p))[0]: load(p) for p in args.configs}
    first_cfg = next(iter(loaded.values()))[0]
    data_cfg = first_cfg['data']
    labels_csv = os.path.join(data_cfg['data_dir'], 'train_labels.csv')
    img_dir = os.path.join(data_cfg['data_dir'], 'train_images')

    # Same split as main.py (random_split seeded by the config seed); all models must share it
    for name, (cfg, _) in loaded.items():
        assert (cfg['seed'], cfg['data']['val_ratio']) == (first_cfg['seed'], data_cfg['val_ratio']), name
    raw = CIFAR10Kaggle(labels_csv, img_dir, transform=None)
    n_val = int(len(raw) * data_cfg['val_ratio'])
    _, val = torch.utils.data.random_split(raw, [len(raw) - n_val, n_val],
                                           generator=torch.Generator().manual_seed(first_cfg['seed']))
    images = [raw[i][0] for i in val.indices]
    labels = torch.tensor([raw[i][1] for i in val.indices])

    correct = {}  # (model, shift, tta) -> bool tensor
    for shift, fn in SHIFTS.items():
        shifted = [fn(img) for img in images]
        for name, (cfg, net) in loaded.items():
            tf = T.Compose([T.ToTensor(), get_normalize(cfg['data'])])
            x_all = torch.stack([tf(img) for img in shifted])
            plain, tta = [], []
            with torch.no_grad():
                for x in x_all.split(250):
                    p = torch.softmax(net(x), 1)
                    q = (p + torch.softmax(net(torch.flip(x, dims=[3])), 1)) / 2
                    plain.append(p.argmax(1)); tta.append(q.argmax(1))
            correct[name, shift, False] = torch.cat(plain) == labels
            correct[name, shift, True] = torch.cat(tta) == labels

    names = list(loaded)
    print(f'\nAccuracy on the {n_val} validation images (plain / flip TTA)')
    print(f'{"model":24s}' + ''.join(f'{s:>20s}' for s in SHIFTS))
    for name in names:
        print(f'{name:24s}' + ''.join(
            f'{100 * correct[name, s, False].float().mean():>11.2f} / {100 * correct[name, s, True].float().mean():5.2f}'
            for s in SHIFTS))

    print('\nPaired McNemar tests (flip TTA): images only model 1 / only model 2 got right, p-value')
    for a, b in itertools.combinations(names, 2):
        cells = []
        for s in SHIFTS:
            ca, cb = correct[a, s, True], correct[b, s, True]
            n1, n2 = (ca & ~cb).sum().item(), (~ca & cb).sum().item()
            cells.append(f'{s}: {n1}/{n2} p={mcnemar_p(n1, n2):.2f}')
        print(f'{a} vs {b}:  ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
