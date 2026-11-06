'''Train CIFAR10 with PyTorch and YAML Config.'''
import os
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import wandb
import models
from dataset import CIFAR10Kaggle
from utils import progress_bar

# 1. Parse Arguments
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', type=str, required=True,
                    help='Path to configuration YAML file')
parser.add_argument('--resume', '-r', action='store_true',
                    help='Resume from checkpoint')

parser.add_argument('--optimizer', type=str, default=None,
                    help="Override optimizer: 'adamw' or 'sgd'")
parser.add_argument('--lr', type=float, default=None,
                    help="Override learning rate")
args = parser.parse_args()

# 2. Load Config File
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # Best test accuracy
start_epoch = 0  # Start epoch

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

# Optional: Color Jitter (lighting / contrast variations)
if aug_cfg.get('color_jitter', False):
    train_transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))

# 2. Convert to Tensor & Normalize
train_transform_list.extend([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 3. Tensor augmentations: Cutout / RandomErasing (must run on Tensor)
if aug_cfg.get('cutout', True):
    p = aug_cfg.get('cutout_prob', 0.5)
    train_transform_list.append(
        transforms.RandomErasing(p=p, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
    )

transform_train = transforms.Compose(train_transform_list)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

# Fast DataLoader kwargs: keep workers alive and prefetch batches
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
model_cfg = config['model']
model_name = model_cfg.get('name', 'CustomResNet')
print(f'==> Building model: {model_name}..')

if hasattr(models, model_name):
    model_cls = getattr(models, model_name)
    # Pass kwargs if CustomResNet, else standard call
    if model_name == 'CustomResNet':
        kwargs = {k: v for k, v in model_cfg.items() if k != 'name'}
        net = model_cls(**kwargs)
    else:
        net = model_cls()
else:
    raise ValueError(f"Model '{model_name}' not found in models module.")

# Check and print parameter budget before training
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
max_budget = config.get('max_params', 5_000_000)
print(f'==> Model Parameters: {total_params:,} (Limit: {max_budget:,})')
if total_params > max_budget:
    raise ValueError(
        f"PARAMETER BUDGET EXCEEDED: Model has {total_params:,} parameters, "
        f"which is strictly greater than the allowed budget of {max_budget:,}!"
    )

# Optional: Load Self-Supervised / Pretrained Backbone Weights
pretrained_path = config.get('pretrained_backbone', None)
if pretrained_path:
    print(f'==> Loading pretrained backbone weights from: {pretrained_path}..')
    assert os.path.isfile(pretrained_path), f'Error: pretrained checkpoint not found at {pretrained_path}'
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('net', ckpt)

    # Clean 'module.' prefix and strip classifier head (e.g. 4-class rotation head)
    backbone_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith('module.') else k
        if not clean_key.startswith('linear.'):
            backbone_dict[clean_key] = v

    missing, unexpected = net.load_state_dict(backbone_dict, strict=False)
    print(f'==> Backbone weights transferred! Initialized fresh 10-class head ({missing}).')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint_dir = config['checkpoint']['dir']
checkpoint_path = os.path.join(checkpoint_dir, config['checkpoint']['filename'])

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

# 5. Resume logic
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(checkpoint_path), \
        f'Error: no checkpoint found at {checkpoint_path}!'
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# 6. Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer_name = (args.optimizer or config['training'].get('optimizer', 'sgd')).lower()
lr = float(args.lr if args.lr is not None else config['training']['lr'])
weight_decay = float(config['training'].get('weight_decay', 5e-4))
if optimizer_name == 'adamw':
    betas = tuple(config['training'].get('betas', [0.9, 0.999]))
    optimizer = optim.AdamW(
        net.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    print(f'==> Using AdamW optimizer: lr={lr}, weight_decay={weight_decay}, betas={betas}')
elif optimizer_name == 'sgd':
    momentum = float(config['training'].get('momentum', 0.9))
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    print(f'==> Using SGD optimizer: lr={lr}, momentum={momentum}, weight_decay={weight_decay}')
else:
    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Choose 'sgd' or 'adamw'.")


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['training']['epochs'],
)


# 7. Train Loop
def train(epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1),
                        100. * correct / total, correct, total))

    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# 8. Validation Loop
def validate(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

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
epochs_no_improve = 0

for epoch in range(start_epoch, config['training']['epochs']):
    train_loss, train_acc = train(epoch)
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()

    metrics = {
        'epoch': epoch,
        'train/loss': train_loss,
        'train/acc': train_acc,
        'learning_rate': current_lr,
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
