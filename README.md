# CIFAR-10 ResNet Under 5M Parameters

Team **The Distillers**: Jonathan David and Matan Tsafrir (Deep Learning, Prof. Chinmay Hegde).

This repository trains a configurable ResNet in PyTorch for the course's Kaggle CIFAR-10 competition, under a budget of 5M trainable parameters. Every architectural and training choice is set in a YAML config file. Our final model reached **87.7%** accuracy on the private Kaggle leaderboard (rank 21 of 81; 89.9% on the public leaderboard). The experiments and design decisions behind it are described in our report.

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

##  Running Experiments

### 1. Run the Baseline Config
```bash
cd /scratch/yd3245/projects/DeepLearning/Resnet/pytorch-cifar-resnet
sbatch train.sbatch
```
*(Defaults to `configs/base_config.yaml`)*

### 2. Run a Specific Config
Pass the path to whichever config YAML you want to run - this is to run our best config:
```bash
sbatch train.sbatch configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml
```

---

##  Generating Kaggle Submissions

To run inference on the test set and generate a `submission.csv` for Kaggle:

```bash
python predict.py --config configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml --tta --tta_mode flip
```

* `--tta --tta_mode flip` averages each prediction with that of the mirrored image, as in our final submission. (`--tta_mode 10crop` is also available.)

* By default, it loads `best.pth` from the checkpoint directory specified in the config.
* To specify a custom checkpoint or output filename:
  ```bash
  python predict.py \
      --config configs/resnet18_k3_skip.yaml \
      --checkpoint checkpoints/resnet18_k3_skip/best.pth \
      --output submissions/sub_k3_skip.csv
  ```

---

##  Creating a New Experiment

To test a new architecture or set of hyperparameters:

1. **Create a new config file** in `configs/`:
   ```bash
   cp configs/base_config.yaml configs/my_new_experiment.yaml
   ```

2. **Edit the parameters** inside `configs/my_new_experiment.yaml`:
   ```yaml
   experiment_name: "my_new_experiment"
   seed: 42
   max_params: 5000000                   # Strict budget: raises error if exceeded

   # --- Weights & Biases (W&B) Logging ---
   wandb:
     enabled: true                       # Set to true to log to W&B
     project: "cifar10-resnet-challenge"
     entity: "the_distillers"            # W&B team/username

   # --- Data & Augmentation ---
   data:
     data_dir: "../data"                 # Path to train_images/ and train_labels.csv
     train_batch_size: 128
     test_batch_size: 100
     num_workers: 8
     val_ratio: 0.0                      # 0.0 = full 50k train set, 0.10 = 10% validation split

     # Augmentation Pipeline
     augmentation:
       random_crop: true                 # Random crop with 4-pixel reflection padding
       horizontal_flip: true             # 50% chance horizontal flip
       autoaugment: true                 # CIFAR-10 AutoAugment policy
       cutout: true                      # RandomErasing / Cutout
       cutout_prob: 0.7                  # Probability of applying Cutout (e.g. 0.5 - 0.7)
       color_jitter: false               # Lighting/contrast adjustments

   # --- Model Architecture ---
   model:
     name: "CustomResNet"
     num_layers: 4                       # N: Number of residual layers
     blocks_per_layer: [3, 7, 6, 1]      # B_i: Blocks per layer (e.g. [3, 7, 6, 1] for Arch B)
     channels: [40, 80, 160, 320]        # C_i: Channels tuned to stay < 5M parameters
     conv_kernel_sizes: [3, 3, 3, 3]     # F_i: 3x3 conv kernel throughout
     skip_kernel_sizes: [1, 1, 1, 1]     # K_i: 1x1 skip connections
     pool_size: 4                        # P: Spatial pooling kernel size before linear head
     num_classes: 10
     shortcut_type: 'avgpool'            # 'avgpool' (avoids spatial loss) or 'conv'

   # --- Training & Optimization ---
   training:
     optimizer: 'sgd'                    # 'sgd' or 'adamw'
     lr: 0.05                            # Learning rate (e.g. 0.05 - 0.1 for SGD, 1e-3 for AdamW)
     momentum: 0.9                       # Momentum factor (used if optimizer: 'sgd')
     betas: [0.9, 0.999]                 # Betas (used if optimizer: 'adamw')
     weight_decay: 2e-4                  # L2 regularization factor
     epochs: 1200                        # CosineAnnealingLR schedules over this duration

   # --- Checkpointing ---
   checkpoint:
     dir: "./checkpoints/my_new_experiment"
     filename: "best.pth"
   ```

3. **Launch the training job**:
   ```bash
   sbatch train.sbatch configs/my_new_experiment.yaml
   ```

---
