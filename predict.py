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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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

    print(f'==> Running inference on {len(testset)} test images..')

    # 6. Predict Loop
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for inputs, ids in tqdm(testloader, desc='Predicting'):
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)

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

