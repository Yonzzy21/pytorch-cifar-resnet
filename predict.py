'''Inference script to generate Kaggle test predictions.'''
import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import models
from dataset import CIFAR10KaggleTest


def predict_with_10crop_tta(net, inputs):
    """Center-Weighted 10-Crop Test-Time Augmentation (TTA) with Logit Averaging.

    Why 10-Crop?
    Convolutional Neural Networks are not strictly translation invariant (due to strided convolutions
    and spatial pooling). When an image object is shifted by even 1-2 pixels, the resulting feature map
    activations change. By sampling 5 spatial crops (center + 4 corner shifts of 2 pixels) along with
    their horizontal mirror flips (5 x 2 = 10 views total), we average out boundary misalignments.

    Why Center-Weighted?
    In CIFAR-10, images are generally roughly object-centered. If all 10 views had equal weight (0.10 each),
    the 8 shifted crops (80% of total vote) could overpower a well-framed center image.
    We therefore allocate:
      - 50% total weight to Center views (0.25 original + 0.25 horizontal flip)
      - 50% total weight distributed across the 4 shifted corner views (0.0625 each for 4 corners x 2 flips)

    Why Logit Averaging?
    Averaging raw logits before softmax behaves as a geometric mean over probabilities.
    It prevents any single distorted crop from outputting an extreme probability (>99%) and
    overruling the consensus of the remaining views.
    """
    # 1. Pad input image by 2 pixels on all 4 borders using reflection padding.
    #    [Batch, 3, 32, 32] -> [Batch, 3, 36, 36]
    padded = F.pad(inputs, (2, 2, 2, 2), mode='reflect')

    # 2. Extract 5 spatial crops of size 32x32:
    # Center crop: exactly recovers the original unshifted inputs [2:34, 2:34]
    crop_center = padded[:, :, 2:34, 2:34]

    # 4 corner crops shifted by 2 pixels in each diagonal direction
    crop_tl = padded[:, :, 0:32, 0:32]  # Shifted 2px up, 2px left
    crop_tr = padded[:, :, 0:32, 4:36]  # Shifted 2px up, 2px right
    crop_bl = padded[:, :, 4:36, 0:32]  # Shifted 2px down, 2px left
    crop_br = padded[:, :, 4:36, 4:36]  # Shifted 2px down, 2px right

    # 3. Weights: 50% Center, 50% Corners
    w_center = 0.25     # 0.25 * 2 = 0.50
    w_corner = 0.0625   # 0.0625 * 8 = 0.50

    # 4. Forward pass for Center views (original + horizontal flip)
    logits = w_center * net(crop_center)
    logits += w_center * net(torch.flip(crop_center, dims=[3]))

    # 5. Forward pass for the 4 shifted corner views (original + horizontal flip)
    for crop in (crop_tl, crop_tr, crop_bl, crop_br):
        logits += w_corner * net(crop)
        logits += w_corner * net(torch.flip(crop, dims=[3]))

    return logits


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
                        help='Enable Test-Time Augmentation')
    parser.add_argument('--tta_mode', type=str, default='10crop', choices=['10crop', 'flip'],
                        help="TTA method: '10crop' (Center-weighted 10-view spatial shift TTA) or 'flip' (2-view flip TTA)")
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
    if args.tta:
        print(f"==> Test-Time Augmentation (TTA) active: Mode='{args.tta_mode}'")
    else:
        print("==> TTA Disabled (single standard forward pass)")

    # 6. Predict Loop
    all_ids = []
    all_preds = []

    with torch.no_grad():
        desc = 'Predicting' + (f' (TTA: {args.tta_mode})' if args.tta else '')
        for inputs, ids in tqdm(testloader, desc=desc):
            inputs = inputs.to(device)
            if args.tta:
                if args.tta_mode == '10crop':
                    # Center-weighted 10-crop TTA (Center original+flip: 50%, 4 corner shifts original+flip: 50%)
                    # Uses logit averaging to prevent overconfidence distortion
                    logits = predict_with_10crop_tta(net, inputs)
                    _, predicted = logits.max(1)
                else:
                    # Classic 2-view flip TTA (original + horizontal flip)
                    probs_orig = net(inputs).softmax(dim=1)
                    probs_flip = net(torch.flip(inputs, dims=[3])).softmax(dim=1)
                    probs = 0.5 * (probs_orig + probs_flip)
                    _, predicted = probs.max(1)
            else:
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

