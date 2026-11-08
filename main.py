'''Train CIFAR10 with PyTorch and YAML Config.'''
import os
import argparse
import contextlib
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import wandb
import models
from cutmix import cutmix_batch
from dataset import CIFAR10Kaggle
from normalization import get_normalize
from utils import progress_bar

# 1. Parse Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', type=str, required=True,
                    help='Path to configuration YAML file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='Resume from checkpoint')
args = parser.parse_args()

# 2. Load Config File
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # Best test accuracy
start_epoch = 0  # Start epoch
epochs_no_improve = 0

experiment_name = config.get('experiment_name', 'default')
print(f'===> Experiment: {experiment_name}')

# 3. Data Transformations & Loaders
print('==> Preparing data..')
data_cfg = config['data']
aug_cfg = data_cfg.get('augmentation', {})

train_transform_list = []

# 1. Spatial / Geometry augmentations on PIL Image
if aug_cfg.get('random_crop', True):
    train_transform_list.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))

if aug_cfg.get('horizontal_flip', True):
    train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

# Optional: AutoAugment policy optimized for CIFAR-10
if aug_cfg.get('autoaugment', False):
    train_transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

# Optional: TrivialAugment (Müller & Hutter, 2021): one random op at a random strength per image
if aug_cfg.get('trivialaugment', False):
    train_transform_list.append(transforms.TrivialAugmentWide())

# Optional: AugMix (Hendrycks et al., 2020): mixes several augmentation chains per image.
# torchvision's version omits the paper's Jensen-Shannon consistency loss.
if aug_cfg.get('augmix', False):
    train_transform_list.append(transforms.AugMix())

# Optional: Color Jitter (lighting / contrast variations)
if aug_cfg.get('color_jitter', False):
    train_transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

# 2. Convert to Tensor & Normalize (data.normalization: 'dataset' or 'per_image')
train_transform_list.extend([
    transforms.ToTensor(),
    get_normalize(data_cfg),
])

# 3. Tensor augmentations: Cutout / RandomErasing (must run on Tensor)
if aug_cfg.get('cutout', True):
    p = aug_cfg.get('cutout_prob', 0.5)
    train_transform_list.append(
        transforms.RandomErasing(p=p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    )

transform_train = transforms.Compose(train_transform_list)

# 4. Batch-level augmentation: CutMix (mixes pairs of images in the training loop)
use_cutmix = aug_cfg.get('cutmix', False)
cutmix_alpha = aug_cfg.get('cutmix_alpha', 1.0)
cutmix_prob = aug_cfg.get('cutmix_prob', 0.5)
if use_cutmix:
    print(f'   CutMix enabled (alpha={cutmix_alpha}, prob={cutmix_prob})')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    get_normalize(data_cfg),
])

# Check for fast in-memory dataset staged in /dev/shm
user = os.environ.get('USER', 'yd3245')
shm_data_dir = f"/dev/shm/{user}/data"
if os.path.isdir(os.path.join(shm_data_dir, 'train_images')):
    effective_data_dir = shm_data_dir
    print(f'==> Using fast in-memory RAM dataset from: {effective_data_dir}')
else:
    effective_data_dir = data_cfg['data_dir']

full_trainset = CIFAR10Kaggle(
    csv_file=os.path.join(effective_data_dir, 'train_labels.csv'),
    img_dir=os.path.join(effective_data_dir, 'train_images'),
    transform=transform_train,
)

val_ratio = data_cfg.get('val_ratio', 0.15)
total = len(full_trainset)

# Fast DataLoader settings: keep worker processes alive and prefetch asynchronously
loader_kwargs = {
    'num_workers': data_cfg['num_workers'],
    'pin_memory': True,
}
if data_cfg['num_workers'] > 0:
    loader_kwargs['persistent_workers'] = True
    loader_kwargs['prefetch_factor'] = 4

if val_ratio and val_ratio > 0.0:
    val_size = int(total * val_ratio)
    train_size = total - val_size

    trainset, valset = torch.utils.data.random_split(
        full_trainset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )

    # Validation set uses test-time transforms (no augmentation)
    valset_eval = CIFAR10Kaggle(
        csv_file=os.path.join(effective_data_dir, 'train_labels.csv'),
        img_dir=os.path.join(effective_data_dir, 'train_images'),
        transform=transform_test,
    )
    valset_eval = torch.utils.data.Subset(valset_eval, valset.indices)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=data_cfg['train_batch_size'],
        shuffle=True,
        **loader_kwargs
    )

    valloader = torch.utils.data.DataLoader(
        valset_eval,
        batch_size=data_cfg['test_batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    print(f'   Train: {train_size} samples | Val: {val_size} samples')
else:
    # Full dataset training (100% of samples used for training)
    trainloader = torch.utils.data.DataLoader(
        full_trainset,
        batch_size=data_cfg['train_batch_size'],
        shuffle=True,
        **loader_kwargs
    )
    valloader = None
    print(f'   Full Training Mode: using all {total} samples for training (no validation split)')

# 4. Dynamic Model Building
def build_model(model_cfg):
    model_name = model_cfg.get('name', 'CustomResNet')
    if not hasattr(models, model_name):
        raise ValueError(f"Model '{model_name}' not found in models module.")
    model_cls = getattr(models, model_name)
    # Pass kwargs if CustomResNet, else standard call
    if model_name == 'CustomResNet':
        return model_cls(**{k: v for k, v in model_cfg.items() if k != 'name'})
    return model_cls()


model_cfg = config['model']
print(f"==> Building model: {model_cfg.get('name', 'CustomResNet')}..")
net = build_model(model_cfg)

# Check and print parameter budget before training
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
max_budget = config.get('max_params', 5_000_000)
print(f'==> Model Parameters: {total_params:,} (Limit: {max_budget:,})')
if total_params > max_budget:
    raise ValueError(
        f"PARAMETER BUDGET EXCEEDED: Model has {total_params:,} parameters, "
        f"which is strictly greater than the allowed budget of {max_budget:,}!"
    )

net = net.to(device)
if device == 'cuda':
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
        print(f'==> Using DataParallel across {torch.cuda.device_count()} GPUs')
    cudnn.benchmark = True

checkpoint_dir = config['checkpoint']['dir']
checkpoint_path = os.path.join(checkpoint_dir, config['checkpoint']['filename'])
last_path = os.path.join(checkpoint_dir, 'last.pth')  # Full training state, overwritten every epoch

# Mixed precision (bf16 autocast): ~2x faster on A100/L4, no loss scaling needed
use_amp = config['training'].get('amp', False)
if use_amp:
    print('==> Mixed precision (bf16 autocast) enabled')


def autocast():
    return torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp)


# Knowledge distillation: a frozen teacher's logits are the student's training target
distill_cfg = config.get('distillation', {})
use_distill = distill_cfg.get('enabled', False)
if use_distill:
    with open(distill_cfg['teacher_config'], 'r') as f:
        teacher_config = yaml.safe_load(f)
    teacher = build_model(teacher_config['model'])
    teacher_ckpt = torch.load(distill_cfg['teacher_checkpoint'], map_location=device, weights_only=True)
    # Strip the DataParallel 'module.' prefix
    teacher.load_state_dict({k.removeprefix('module.'): v for k, v in teacher_ckpt['net'].items()})
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    distill_loss_type = distill_cfg.get('loss', 'mse')
    distill_temperature = distill_cfg.get('temperature', 4.0)
    ce_weight = distill_cfg.get('ce_weight', 0.0)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"==> Distillation: teacher {distill_cfg['teacher_checkpoint']} "
          f"({teacher_params:,} params, val acc {teacher_ckpt.get('acc', float('nan')):.2f}% "
          f"at epoch {teacher_ckpt.get('epoch', '?')}) | loss={distill_loss_type} | ce_weight={ce_weight}")


def distill_loss(student_logits, teacher_logits):
    s, t = student_logits.float(), teacher_logits.float()
    if distill_loss_type == 'mse':
        # Match raw logits (mean over batch and classes)
        return F.mse_loss(s, t)
    # Classic Hinton KD on temperature-softened probabilities
    T = distill_temperature
    return F.kl_div(F.log_softmax(s / T, dim=1), F.log_softmax(t / T, dim=1),
                    log_target=True, reduction='batchmean') * T * T

# Setup Weights & Biases (W&B)
wandb_cfg = config.get('wandb', {})
use_wandb = wandb_cfg.get('enabled', False)
if use_wandb:
    wandb.init(
        project=wandb_cfg.get('project', 'cifar10-resnet-challenge'),
        entity=wandb_cfg.get('entity', None),
        name=experiment_name,
        config={
            **config,
            'total_params': total_params
        }
    )
    wandb.watch(net, log='all', log_freq=100)
    print(f'==> W&B initialized for experiment: {experiment_name}')

# 6. Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
# training.optimizer: 'sgd' (default), 'adamw', or 'sam' (SGD with sharpness-aware steps)
optimizer_name = config['training'].get('optimizer', 'sgd')
if optimizer_name == 'adamw':
    # Decoupled weight decay (Loshchilov & Hutter); needs its own lr / weight_decay
    optimizer = optim.AdamW(
        net.parameters(),
        lr=config['training']['lr'],
        weight_decay=float(config['training']['weight_decay']),
    )
elif optimizer_name in ('sgd', 'sam'):
    optimizer = optim.SGD(
        net.parameters(),
        lr=config['training']['lr'],
        momentum=config['training']['momentum'],
        weight_decay=float(config['training']['weight_decay']),
    )
else:
    raise ValueError(f"Unknown optimizer '{optimizer_name}'")
use_sam = optimizer_name == 'sam'
sam_rho = config['training'].get('sam_rho', 0.05)
print(f"==> Optimizer: {optimizer_name}" + (f' (rho={sam_rho})' if use_sam else ''))

# Auxiliary rotation loss (Hendrycks et al., 2019): the rotation head learns to tell how each
# training image was rotated; this self-supervised task improves robustness to distribution shift
rotation_weight = config['training'].get('rotation_weight', 0.0)
use_rotation = rotation_weight > 0
if use_rotation:
    assert config['model'].get('rotation_head', False), 'rotation_weight needs model.rotation_head: true'
    assert not use_sam and not use_cutmix, 'the rotation loss is not implemented together with SAM or CutMix'
    print(f'==> Rotation auxiliary loss enabled (weight={rotation_weight})')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['training']['epochs'],
)

# Resume: restore weights, optimizer (momentum) and LR schedule from last.pth,
# so a run cut off by a session limit continues exactly where it stopped
if args.resume:
    assert os.path.isfile(last_path), f'Error: no checkpoint found at {last_path}!'
    checkpoint = torch.load(last_path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_acc = checkpoint['best_acc']
    epochs_no_improve = checkpoint['epochs_no_improve']
    start_epoch = checkpoint['epoch'] + 1
    print(f'==> Resumed from {last_path}: continuing at epoch {start_epoch} '
          f'(lr {scheduler.get_last_lr()[0]:.6f}, best val acc so far {best_acc:.2f}%)')


def sam_second_pass(inputs, batch_loss):
    '''SAM (Foret et al., 2021): move the weights to the worst point within radius rho
    (along the gradient), recompute the gradient there, then move back. optimizer.step()
    then applies that gradient at the original weights, steering toward flat minima.'''
    params = [p for p in net.parameters() if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    scale = sam_rho / (grad_norm + 1e-12)
    eps = []
    with torch.no_grad():
        for p in params:
            e = p.grad * scale
            p.add_(e)
            eps.append(e)
    optimizer.zero_grad(set_to_none=True)
    # Keep the BatchNorm running statistics from the first pass
    with frozen_bn_stats(), autocast():
        batch_loss(net(inputs)).backward()
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.sub_(e)


@contextlib.contextmanager
def frozen_bn_stats():
    '''Forward passes inside this block don't update BatchNorm running statistics (momentum 0).'''
    bn_layers = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mom in zip(bn_layers, saved):
            m.momentum = mom


def rotation_loss(inputs):
    '''Rotate each image by a random multiple of 90 degrees and score the rotation head on it.
    BN running statistics are not updated, so they keep describing upright images only.'''
    k = torch.randint(0, 4, (inputs.size(0),), device=inputs.device)
    rotated = inputs.clone()
    for r in (1, 2, 3):
        rotated[k == r] = torch.rot90(inputs[k == r], r, dims=(2, 3))
    with frozen_bn_stats(), autocast():
        _, rot_logits = net(rotated, with_rotation=True)
        return criterion(rot_logits, k)


# 7. Train Loop
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    train_rot_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        mixed = use_cutmix and np.random.rand() < cutmix_prob
        if mixed:
            inputs, targets_a, targets_b, lam = cutmix_batch(inputs, targets, cutmix_alpha)
        if use_distill:
            # The teacher sees the same augmented batch as the student
            with torch.no_grad(), autocast():
                teacher_outputs = teacher(inputs)

        def batch_loss(outputs):
            if mixed:
                # Loss and accuracy are weighted by how much of each image is original vs. pasted
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)
            if use_distill:
                loss = distill_loss(outputs, teacher_outputs) + ce_weight * loss
            return loss

        with autocast():
            outputs = net(inputs)
            loss = batch_loss(outputs)
        loss.backward()
        if use_sam:
            sam_second_pass(inputs, batch_loss)
        if use_rotation:
            rot_loss = rotation_loss(inputs)
            (rotation_weight * rot_loss).backward()
            train_rot_loss += rot_loss.item()
        optimizer.step()

        _, predicted = outputs.max(1)
        if mixed:
            correct += (lam * predicted.eq(targets_a).sum().item()
                        + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets).sum().item()
        train_loss += loss.item()
        total += targets.size(0)

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))

    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    epoch_rot_loss = train_rot_loss / len(trainloader)
    return epoch_loss, epoch_acc, epoch_rot_loss


# 8. Validation Loop
def validate(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with autocast():
                outputs = net(inputs)
            loss = criterion(outputs.float(), targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1),
                            100. * correct / total, correct, total))

    epoch_loss = val_loss / len(valloader)
    epoch_acc = 100. * correct / total

    # Save checkpoint
    if epoch_acc > best_acc:
        print('Saving checkpoint..')
        state = {
            'net': net.state_dict(),
            'acc': epoch_acc,
            'epoch': epoch,
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state, checkpoint_path)
        best_acc = epoch_acc

    return epoch_loss, epoch_acc


# 9. Main Execution
early_stopping_patience = config['training'].get('early_stopping_patience', None)

for epoch in range(start_epoch, config['training']['epochs']):
    train_loss, train_acc, train_rot_loss = train(epoch)
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    metrics = {
        'epoch': epoch,
        'train/loss': train_loss,
        'train/acc': train_acc,
        'learning_rate': current_lr,
        **({'train/rot_loss': train_rot_loss} if use_rotation else {}),
    }

    if valloader is not None:
        prev_best = best_acc
        val_loss, val_acc = validate(epoch)

        if val_acc > prev_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        metrics.update({
            'val/loss': val_loss,
            'val/acc': val_acc,
            'best_val_acc': best_acc,
            'epochs_no_improve': epochs_no_improve,
        })

        if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
            print(f'\n==> Early stopping triggered: Validation accuracy has not improved for {early_stopping_patience} consecutive epochs.')
            print(f'==> Stopping at epoch {epoch}. Best validation accuracy: {best_acc:.2f}%')
            break
    else:
        # Full training mode: save latest weights every epoch
        state = {
            'net': net.state_dict(),
            'acc': train_acc,
            'epoch': epoch,
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(state, checkpoint_path)

    # Full training state for --resume; write-then-rename so a kill mid-save can't corrupt it
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'epochs_no_improve': epochs_no_improve,
    }, last_path + '.tmp')
    os.replace(last_path + '.tmp', last_path)

    if use_wandb:
        wandb.log(metrics)

if valloader is not None:
    print(f'\nBest validation accuracy: {best_acc:.2f}%')
    if use_wandb:
        wandb.summary['best_val_acc'] = best_acc
else:
    print(f'\nFull training complete! Saved final model to: {checkpoint_path}')

if use_wandb:
    wandb.finish()
