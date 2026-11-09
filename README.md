# CIFAR-10 ResNet Under 5M Parameters

Team **The Distillers**: Jonathan David and Matan Tsafrir (Deep Learning, Prof. Chinmay Hegde).

This repository trains a configurable ResNet in PyTorch for the course's Kaggle CIFAR-10 competition, under a budget of 5M trainable parameters. Our final model reached **87.7%** accuracy on the private Kaggle leaderboard (rank 21 of 81; 89.9% on the public leaderboard). The experiments and design decisions behind it are described in our report.

## Acknowledgment

This code builds on [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (MIT license, see `LICENSE`), which we extended with a configurable ResNet, average-pooling shortcuts, config-driven augmentation, full-data training, a parameter-budget check, W&B logging, Kaggle data loading and test-time augmentation.

## Deliverables

* **Notebook:** [`evaluation_and_plots.ipynb`](evaluation_and_plots.ipynb), executed with all outputs. It builds the final model from its config, prints its parameter count, loads the trained checkpoint (strict loading), states the final test accuracy, plots our progress on the Kaggle leaderboard, and checks the predicted class distribution on the test set.
* **Trained checkpoint:** `checkpoints/fulldata_arch_B_sgd_1200_low_weight_decay/best.pth` (19 MB).
* **Final submission:** `Final_submission_fulldata_arch_B_1200_low_weight_decay.csv`, the predictions behind our final score.
* **Trainable parameters:** **4,976,530** (< 5,000,000).
* **Final test accuracy:** **87.7%** on the private Kaggle leaderboard (rank 21 of 81).

## Final model

* **Architecture ("B"):** `CustomResNet` with blocks per stage `[3, 7, 6, 1]`, channels `[40, 80, 160, 320]`, 3x3 convolutions, and average-pooling shortcuts (2x2 average pooling followed by a 1x1 convolution) in the downsampling blocks.
* **Data:** all 50,000 labeled training images (no validation split).
* **Training:** 1200 epochs, SGD with momentum 0.9, learning rate 0.05 with a cosine schedule, weight decay 2e-4, label smoothing 0.1 (set in `main.py`).
* **Augmentation:** random crop with 4-pixel reflection padding, horizontal flip, AutoAugment (CIFAR-10 policy) and Cutout (random erasing, p = 0.7).
* **Test-time augmentation:** each test image's prediction is averaged with that of its horizontally flipped copy.
* **Config:** `configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml`. The second final model (weight decay 5e-4, 0.870 private) uses `configs/fulldata_arch_B_SGD_1200.yaml`, and the 600-epoch run uses `configs/fulldata_arch_B_SGD_600.yaml`.

## Other experiments

* **Self-supervised pretraining (RotNet):** `pretrain.py`, `pretrain_dataset.py` and the `configs/pretrain*.yaml` / `configs/fine_tune*.yaml` configs (on this branch).
* **SAM, knowledge distillation, deeper layouts and robustness checks:** on the [`matan/experiments`](https://github.com/Yonzzy21/pytorch-cifar-resnet/tree/matan/experiments) branch, with a results log in its `BRANCH_NOTES.md`.

---

##  Repository Structure

```
pytorch-cifar-resnet/
├── evaluation_and_plots.ipynb     # Deliverable: parameter count, checkpoint loading, test accuracy, plots
├── Final_submission_fulldata_arch_B_1200_low_weight_decay.csv  # Final Kaggle submission
├── best_config_wandb.yaml         # W&B record of the final run's config
├── best_config_log.log            # Training log of the final run
├── configs/                       # Experiment YAML configurations
├── models/
│   ├── __init__.py                # Model exports
│   └── resnet.py                  # CustomResNet and CustomBlock definitions
├── checkpoints/                   # Final model weights (best.pth)
├── dataset.py                     # Kaggle dataset loaders (train/val split, test set)
├── main.py                        # Training & validation loop
├── predict.py                     # Inference and Kaggle submission (with optional TTA)
├── pretrain.py                    # Self-supervised RotNet pretraining
├── pretrain_dataset.py            # Rotation dataset for pretraining
├── train.sbatch.example           # Slurm template (train.sbatch is git-ignored)
├── utils.py                       # Progress bar & helper utilities
└── LICENSE                        # MIT license (from kuangliu/pytorch-cifar)
```

---

##  Architecture & Hyperparameters

The model `CustomResNet` implements the customizable ResNet specification:
* **$N$ (`num_layers`)**: Number of residual stages (default: 4)
* **$B_i$ (`blocks_per_layer`)**: Number of residual blocks per stage (e.g., `[2, 2, 2, 2]`)
* **$C_i$ (`channels`)**: Channel widths for each stage (e.g., `[40, 80, 160, 320]`)
* **$F_i$ (`conv_kernel_sizes`)**: Kernel size for main convolutions (e.g., `[3, 3, 3, 3]`)
* **$K_i$ (`skip_kernel_sizes`)**: Kernel size for shortcut projection convolutions (`1` or `3`)
* **$P$ (`pool_size`)**: Spatial pooling size before the linear classifier (default: 4)

### Parameter Budget Enforcer
Before training starts, `main.py` counts all trainable parameters. If `total_params > max_params` (5,000,000), it raises an immediate error.

---

##  Reproducing Our Results

We trained on NYU's HPC cluster with Slurm, and the commands below assume it. Outside Slurm, run `python main.py --config <config>` directly instead of `sbatch train.sbatch <config>`.

### 1. Setup

* **Packages:** PyTorch, torchvision, PyYAML, pandas, Pillow, tqdm, matplotlib and wandb.
* **Data:** place the Kaggle competition files in a `data/` folder next to the repository (configs use `data_dir: "../data"`):
  ```
  data/
  ├── train_images/
  ├── train_labels.csv
  ├── test_images/
  └── test.csv
  ```
* **Weights & Biases:** the configs log to our team's W&B project. To run without it, set `wandb.enabled: false` in the config or run with `WANDB_MODE=disabled`.
* **Slurm:** copy `train.sbatch.example` to `train.sbatch` and fill in your account, overlay, environment and (optionally) W&B key.

### 2. Train the final model

```bash
cd /scratch/<your-netid>/pytorch-cifar-resnet
sbatch train.sbatch configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml
```

The weights are saved to `checkpoints/fulldata_arch_B_sgd_1200_low_weight_decay/best.pth`. Any other config in `configs/` runs the same way.

### 3. Generate the Kaggle submission

```bash
python predict.py --config configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml --tta --tta_mode flip
```

* `--tta --tta_mode flip` averages each prediction with that of the mirrored image, as in our final submission. With the committed checkpoint, this command reproduces `Final_submission_fulldata_arch_B_1200_low_weight_decay.csv` exactly. (`--tta_mode 10crop` is also available.)
* By default, it loads `best.pth` from the checkpoint directory in the config. Use `--checkpoint <path>` and `--output <file>` to change either.

### 4. Check the result without training

Run `evaluation_and_plots.ipynb`. It loads the committed checkpoint and works on a CPU.

---

##  Creating a New Experiment

1. **Copy a config:**
   ```bash
   cp configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml configs/my_new_experiment.yaml
   ```

2. **Edit it.** The main settings, for example:
   ```yaml
   experiment_name: "my_new_experiment"
   data:
     val_ratio: 0.1                      # 0 = train on all 50k images; 0.1 = hold out 10% for validation
     augmentation:
       autoaugment: true
       cutout_prob: 0.7
   model:
     blocks_per_layer: [3, 7, 6, 1]      # B_i
     channels: [40, 80, 160, 320]        # C_i
     shortcut_type: 'avgpool'            # 'avgpool' or 'conv'
   training:
     optimizer: 'sgd'                    # 'sgd' or 'adamw'
     lr: 0.05
     weight_decay: 2e-4
     epochs: 1200
   checkpoint:
     dir: "./checkpoints/my_new_experiment"
   ```
   The model settings are explained in [Architecture & Hyperparameters](#architecture--hyperparameters); `configs/` holds complete examples.

3. **Launch it:**
   ```bash
   sbatch train.sbatch configs/my_new_experiment.yaml
   ```

---
