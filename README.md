# CIFAR-10 ResNet Architecture Exploration (< 5M Parameters)

This repository implements a modular, configurable ResNet architecture in PyTorch for the Kaggle CIFAR-10 competition. Every hyperparameter and architectural choice is driven by YAML config files, with an automatic parameter budget enforcer (< 5,000,000 parameters) and Weights & Biases tracking.

---

## 📁 Repository Structure

```
pytorch-cifar-resnet/
├── configs/                       # Experiment YAML configurations
│   ├── base_config.yaml           # Baseline (1x1 skip, 4.37M params)
│   └── resnet18_k3_skip.yaml      # 3x3 skip connections (4.91M params)
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

## 🏗️ Architecture & Hyperparameters

The model `CustomResNet` implements the customizable ResNet specification:
* **$N$ (`num_layers`)**: Number of residual stages (default: 4)
* **$B_i$ (`blocks_per_layer`)**: Number of residual blocks per stage (e.g., `[2, 2, 2, 2]`)
* **$C_i$ (`channels`)**: Channel widths for each stage (e.g., `[40, 80, 160, 320]`)
* **$F_i$ (`conv_kernel_sizes`)**: Kernel size for main convolutions (e.g., `[3, 3, 3, 3]`)
* **$K_i$ (`skip_kernel_sizes`)**: Kernel size for shortcut projection convolutions (`1` or `3`)
* **$P$ (`pool_size`)**: Spatial pooling size before the linear classifier (default: 4)

### Parameter Budget Enforcer
Before training starts, `main.py` counts all trainable parameters. If `total_params > max_params` (5,000,000), it raises an immediate error to prevent wasted compute.

---

## 🚀 Running Experiments

### 1. Run the Baseline Config
```bash
cd /scratch/yd3245/projects/DeepLearning/Resnet/pytorch-cifar-resnet
sbatch train.sbatch
```
*(Defaults to `configs/base_config.yaml`)*

### 2. Run a Specific Config
Pass the path to whichever config YAML you want to run:
```bash
sbatch train.sbatch configs/resnet18_k3_skip.yaml
```

---

## 🔮 Generating Kaggle Submissions

To run inference on the test set and generate a `submission.csv` for Kaggle:

```bash
python predict.py --config configs/resnet18_k3_skip.yaml
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

## 🧪 Creating a New Experiment

To test a new architecture or set of hyperparameters:

1. **Create a new config file** in `configs/`:
   ```bash
   cp configs/base_config.yaml configs/my_new_experiment.yaml
   ```

2. **Edit the parameters** inside `configs/my_new_experiment.yaml`:
   ```yaml
   experiment_name: "my_new_experiment"
   
   wandb:
     enabled: true
     project: "cifar10-resnet-challenge"
     entity: "the_distillers"

   model:
     name: "CustomResNet"
     num_layers: 4
     blocks_per_layer: [2, 2, 2, 2]
     channels: [40, 80, 160, 320]     # Keep total parameters < 5M
     conv_kernel_sizes: [3, 3, 3, 3]
     skip_kernel_sizes: [1, 1, 1, 1]
     pool_size: 4
     num_classes: 10

   training:
     epochs: 200
     lr: 0.1
     momentum: 0.9
     weight_decay: 0.0005

   checkpoint:
     dir: "./checkpoints/my_new_experiment"
     filename: "best.pth"
   ```

3. **Launch the training job**:
   ```bash
   sbatch train.sbatch configs/my_new_experiment.yaml
   ```

---

## 📊 Monitoring Jobs & Logs

* **Check running jobs**:
  ```bash
  squeue -u yd3245
  ```

* **Cancel a job**:
  ```bash
  scancel <JOB_ID>
  ```

* **View live training logs**:
  ```bash
  tail -f /scratch/yd3245/projects/DeepLearning/Resnet/logs/train_<JOB_ID>.out
  ```

* **Weights & Biases**:
  View live curves (loss, accuracy, learning rate) at your team dashboard:
  [https://wandb.ai/the_distillers/cifar10-resnet-challenge](https://wandb.ai/the_distillers/cifar10-resnet-challenge)
