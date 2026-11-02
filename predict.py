'''Inference script to generate Kaggle test predictions.'''
import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

import models
from dataset import CIFAR10KaggleTest
from normalization import get_normalize


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle CIFAR10 Submission')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (defaults to best.pth from config)')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Path where to save the submission CSV')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Inference batch size')
    parser.add_argument('--tta', action='store_true',
                        help='Test-time augmentation: average softmax over each image and its mirror')
    parser.add_argument('--bn_adapt', action='store_true',
                        help='Re-estimate BatchNorm statistics on the (unlabeled) test images before predicting')
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'==> Using device: {device}')

    # 2. Determine Checkpoint Path
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = config['checkpoint']['dir']
        checkpoint_filename = config['checkpoint']['filename']
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    assert os.path.isfile(checkpoint_path), f'Error: No checkpoint found at {checkpoint_path}'
    print(f'==> Loading checkpoint: {checkpoint_path}')

    # 3. Build Model with Configured Architecture
    model_cfg = config['model']
    model_name = model_cfg.get('name', 'CustomResNet')
    print(f'==> Building model: {model_name}')

    if hasattr(models, model_name):
        model_cls = getattr(models, model_name)
        if model_name == 'CustomResNet':
            kwargs = {k: v for k, v in model_cfg.items() if k != 'name'}
            net = model_cls(**kwargs)
        else:
            net = model_cls()
    else:
        raise ValueError(f"Model '{model_name}' not found in models module.")

    net = net.to(device)

    # 4. Load Trained Weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('net', checkpoint)

    # Handle torch.nn.DataParallel 'module.' prefix if saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    if 'acc' in checkpoint:
        print(f"==> Checkpoint achieved validation accuracy: {checkpoint['acc']:.2f}% (Epoch {checkpoint.get('epoch', '?')})")

    # 5. Prepare Test Data
    data_cfg = config['data']
    data_dir = data_cfg['data_dir']
    
    # Check test CSV and test image directory
    test_csv = os.path.join(data_dir, 'test.csv')
    test_img_dir = os.path.join(data_dir, 'test_images')
    
    assert os.path.isfile(test_csv), f"Error: test.csv not found in {data_dir}"
    assert os.path.isdir(test_img_dir), f"Error: test_images folder not found in {data_dir}"

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        get_normalize(config['data']),  # must match training
    ])

    testset = CIFAR10KaggleTest(
        csv_file=test_csv,
        img_dir=test_img_dir,
        transform=transform_test,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.bn_adapt:
        # Replace the training-set BN statistics with statistics of the test images.
        # No labels are used; weights are unchanged.
        bn_layers = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        for m in bn_layers:
            m.reset_running_stats()
            m.momentum = None  # cumulative average over all test batches
            m.train()
        with torch.no_grad():
            for inputs, _ in testloader:
                net(inputs.to(device))
        net.eval()
        print(f'==> BN adaptation: re-estimated statistics of {len(bn_layers)} BatchNorm layers on the test images')

    print(f'==> Running inference on {len(testset)} test images'
          f'{" (TTA: original + mirror)" if args.tta else ""}..')

    # 6. Predict Loop
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, ids in tqdm(testloader, desc='Predicting'):
            inputs = inputs.to(device)
            probs = torch.softmax(net(inputs), dim=1)
            if args.tta:
                probs = (probs + torch.softmax(net(torch.flip(inputs, dims=[3])), dim=1)) / 2
            _, predicted = probs.max(1)

            all_ids.extend(ids.tolist())
            all_preds.extend(predicted.cpu().tolist())

    # 7. Save Submission CSV
    sub_df = pd.DataFrame({
        'id': all_ids,
        'label': all_preds,
    })

    # Ensure sorted by id
    sub_df = sub_df.sort_values(by='id').reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub_df.to_csv(args.output, index=False)

    print(f'\n==> Submission saved successfully to: {args.output}')
    print(f'==> Total rows: {len(sub_df)}')
    print('\n==> Class prediction breakdown:')
    print(sub_df['label'].value_counts().sort_index())


if __name__ == '__main__':
    main()

