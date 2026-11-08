# CIFAR-10 ResNet Architecture Exploration (< 5M Parameters)

This repository implements a modular, configurable ResNet architecture in PyTorch for a Kaggle CIFAR-10 competition. Every hyperparameter and architectural choice is driven by YAML config files, while maintaining the parameter count under 5M.

---

##  Repository Structure

```
pytorch-cifar-resnet/
├── configs/                       # Experiment YAML configurations
├── models/
│   ├── __init__.py                # Model exports
│   └── resnet.py                  # CustomResNet and CustomBlock definitions
├── checkpoints/                   # Saved best model weights (.pth) [ignored by git]
├── dataset.py                     # Custom Kaggle dataset loader (train/val split)
├── main.py                        # Training & validation loop
├── train.sbatch.example           # Slurm template (train.sbatch is git-ignored for security)
├── utils.py                       # Progress bar & helper utilities
└── .gitignore                     # Ignores weights, logs, data, and wandb
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
python predict.py --config configs/fulldata_arch_B_SGD_1200_low_weight_decay.yaml
```

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
