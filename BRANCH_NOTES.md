# Branch notes: `matan/experiments`

Running log of changes on this branch, relative to `master` at `f2f1308` ("added avgpool in shortcut").

---

## 1. New shortcut option: `shortcut_type: 'auto'`

**File:** `models/resnet.py` (`CustomBlock`)

**Problem.** `shortcut_type: 'avgpool'` replaces *every* downsampling shortcut with AvgPool + 1×1 conv, regardless of `skip_kernel_sizes`. But a strided conv only loses information when its kernel is smaller than its stride:

| K | Stride | Pixels the shortcut conv sees |
|---|---|---|
| 1 | 2 | ¼ (1×1 windows jump over 3 of every 4 pixels) |
| 2 | 2 | all (windows tile exactly) |
| 3 | 2 | all (windows overlap) |

So with `avgpool`, a config that says K=3 silently gets a 1×1 conv instead. The config no longer describes the model, and the param count stays at the K=1 value.

**Change.** Added a third option, `'auto'`: use AvgPool + 1×1 **only when K < stride** (where pixels would be skipped); otherwise keep the K×K conv shortcut. `'conv'` and `'avgpool'` behave exactly as before, so existing configs are unaffected.

```python
# A K x K conv with stride s skips pixels exactly when K < s.
use_avgpool = stride > 1 and (
    shortcut_type == 'avgpool'
    or (shortcut_type == 'auto' and skip_kernel < stride)
)
```

**Verified** (forward pass on a 32×32 batch, baseline layout [2,2,2,2], C=[40,80,160,320]):

| shortcut_type | K | Downsampling shortcuts built | Params |
|---|---|---|---|
| conv | 1 | 1×1 conv | 4,368,690 |
| conv | 3 | 3×3 conv | 4,906,290 |
| avgpool | 1 | AvgPool + 1×1 | 4,368,690 |
| avgpool | 3 | AvgPool + 1×1 (K ignored) | 4,368,690 |
| auto | 1 | AvgPool + 1×1 | 4,368,690 |
| auto | 3 | 3×3 conv | 4,906,290 |

**Experiment to decide keep vs. revert.** With K=1, `auto` builds the same model as `avgpool`, so the only meaningful comparison is:
- `shortcut_type: 'auto'`, K=3 → 3×3 conv shortcut (no info loss, +538k params)
- `shortcut_type: 'avgpool'`, K=1 → AvgPool + 1×1 (no info loss, 0 extra params)

If `auto`/K=3 is not better, revert this change.

Configs for this comparison (identical recipe; only `shortcut_type`, `skip_kernel_sizes`, names and checkpoint dirs differ):
- `configs/shortcut_avgpool_k1.yaml`: 4,368,690 params
- `configs/shortcut_auto_k3.yaml`: 4,906,290 params

Recipe (from `base_config.yaml`): [2,2,2,2], C=[40,80,160,320], batch 128, lr 0.1, SGD momentum 0.9, wd 5e-4, cosine over 100 epochs, crop + flip + cutout, 10% val split, **early stopping disabled** (`early_stopping_patience: null`) so the cosine schedule completes.

Caveat: torch is not seeded yet (only the train/val split is), so expect run-to-run noise of a few tenths of a percent. A smaller difference between the two runs is not meaningful.

**Known limitation (unchanged from `avgpool`).** AvgPool rounds odd sizes down while the padded stride-2 conv rounds up (7×7 → 3×3 vs. 4×4), so an odd feature-map size would crash on the addition. Doesn't occur with 32→16→8→4.

**Result:** `shortcut_avgpool_k1` reached **95.56%** best val acc; `shortcut_auto_k3`: TBD (the two were very close at epoch 85). Since a tie favors the cheaper option, `avgpool` (K=1) is kept as the shortcut for the next experiments.

---

## 2. Architecture experiment: shift capacity out of stage 4

**Motivation.** A block's parameter cost grows with C², so one stage-4 block (C=320) costs ~1.84M params, as much as ~4 stage-3 blocks or ~16 stage-2 blocks. Stage 4 runs on a 4×4 map whose receptive field already covers the whole image. Moving blocks to stages 2–3 buys more depth for the same budget. The question is how best to spend the freed budget: wider or deeper.

**Configs** (avgpool shortcut, K=1; recipe identical to section 1, so `shortcut_avgpool_k1` is the baseline):

| Config | Blocks | Channels | Params |
|---|---|---|---|
| `shortcut_avgpool_k1` (baseline) | [2, 2, 2, 2] | [40, 80, 160, 320] | 4,368,690 |
| `arch_A_shift_widen` | [2, 4, 4, 1] | [48, 96, 184, 376] | 4,996,266 |
| `arch_B_shift_deepen` | [3, 7, 6, 1] | [40, 80, 160, 320] | 4,976,530 |

Each config's header lists its stage-by-stage layout and per-stage parameter counts. Both build, pass the 5M check and run a forward pass.

Both use ~99.5–99.9% of the 5M budget, so A vs. B compares wide vs. deep at (nearly) equal parameter count. Widths in A are multiples of 8 (slightly more GPU-friendly); stage 3 is 184 rather than 192 to stay under 5M.

Note: A and B are deeper than the baseline (22 and 34 convs in blocks vs. 16), so each epoch takes longer.

**Result** (best val acc, 5k validation images, 100 epochs, 2× A100 run in parallel):

| Config | Params | Best val acc |
|---|---|---|
| `shortcut_avgpool_k1` (baseline) | 4,368,690 | 95.56% |
| `arch_A_shift_widen` | 4,996,266 | 95.68% |
| `arch_B_shift_deepen` | 4,976,530 | **96.00%** |

B is best (+0.44% over baseline, +0.32% over A), but all gaps are within run-to-run noise (~±0.5%, unseeded). B is adopted as the working architecture; the win is probable, not proven. All three land within 0.5%, which suggests larger gains will come from the training recipe (augmentation, longer training, distillation) rather than further layout changes.

---

## 3. Batch-size experiment on architecture B

**Question.** Does batch 64 beat batch 128 for B? Batch size and lr interact (linear scaling rule: lr ∝ batch size), so two runs:

| Config | Batch | lr | Tests |
|---|---|---|---|
| `arch_B_shift_deepen` (reference) | 128 | 0.1 | — (96.00%) |
| `arch_B_bs64_lr0.05` | 64 | 0.05 | batch size alone (lr scaled linearly) |
| `arch_B_bs64_lr0.1` | 64 | 0.1 | batch 64 at unchanged lr |

Everything else is identical to `arch_B_shift_deepen`. Batch 64 doubles the steps per epoch (~704 vs. ~352), so these runs are slower.

**Result** (best val acc):

| Config | Batch | lr | Best val acc |
|---|---|---|---|
| `arch_B_shift_deepen` | 128 | 0.1 | **96.00%** |
| `arch_B_bs64_lr0.05` | 64 | 0.05 | 95.88% |
| `arch_B_bs64_lr0.1` | 64 | 0.1 | 95.82% |

All three are within 0.2%, well inside run-to-run noise (~±0.5%). Batch 64 costs ~2× the steps per epoch for no measurable gain, so **batch 128, lr 0.1 is kept**. The lr choice at batch 64 (0.05 vs. 0.1) made no difference either.

---

## 4. Architecture experiment: B vs. Jonathan's [3, 3, 3, 2]

**Question.** Jonathan's layout [3,3,3,2] uses almost the same budget as B but keeps 2 blocks in stage 4. His earlier runs used a different recipe (lr 0.01, batch 64, 80 epochs, early stopping, AutoAugment), so they can't be compared with B directly. This run trains his layout with our recipe.

| Config | Blocks | Channels | Params | Convs in blocks | Best val acc |
|---|---|---|---|---|---|
| `arch_B_shift_deepen` | [3, 7, 6, 1] | [40, 80, 160, 320] | 4,976,530 | 34 | **96.00%** |
| `arch_J_3332` | [3, 3, 3, 2] | [40, 80, 160, 320] | 4,974,610 | 22 | 95.74% |
| `arch_B_repeat` (noise check) | [3, 7, 6, 1] | [40, 80, 160, 320] | 4,976,530 | 34 | 95.98% |

Only `blocks_per_layer` (and the names/checkpoint dir) differ; avgpool shortcut, K=1, batch 128, lr 0.1, 100 epochs, no early stopping. Builds, passes the 5M check, runs a forward pass.

`arch_B_repeat` is an exact rerun of `arch_B_shift_deepen` (same config and split). torch is unseeded, so the gap between the two B runs is a direct measurement of run-to-run noise, which tells us whether the B vs. [3,3,3,2] gap is meaningful.

**Result.** The two B runs agree closely (96.00% and 95.98%), so the noise between identical runs looks smaller than the ±0.5% assumed so far (one pair is not enough to pin it down). [3,3,3,2] is 0.24–0.26% below both B runs. **B is kept** as the architecture.

(The notebook's monitor cell first reported `arch_B_repeat` as crashed, but the run completed. The monitor flags any `Traceback` in the log, including non-fatal warnings.)

---

## 5. Augmentation experiment on B: AutoAugment vs. CutMix (200 epochs)

**Question.** Which augmentation, added to B's current pipeline (crop + flip + cutout), helps more? The two work differently:
- **AutoAugment** (per image): rotations, shears, color and tone changes from the CIFAR-10 policy. Already supported in the code.
- **CutMix** (per batch): pastes a box from another image in the batch and splits the label by area (e.g. 70% cat / 30% dog). New code, see below.

**Epochs.** Both run 200 epochs (cosine schedule stretched to 200). Stronger augmentation makes each epoch's data more varied, so it usually needs more passes to pay off. There is no exact "dataset multiplier" for augmentation; 2× is a conventional choice. Because epochs change too, the fair comparisons are:

| Config | Augmentation | Epochs | Best val acc |
|---|---|---|---|
| `arch_B_shift_deepen` | crop + flip + cutout | 100 | 96.00% |
| `arch_B_e200` | crop + flip + cutout | 200 | 96.52% |
| `arch_B_autoaug_e200` | + AutoAugment | 200 | **96.60%** |
| `arch_B_cutmix_e200` | + CutMix | 200 | TBD |
| `arch_B_e300` | crop + flip + cutout | 300 | 96.58% |
| `arch_B_autoaug_e300` | + AutoAugment | 300 | **97.06%** (epoch 294; session ended during epoch 299 of 300, lr already ≈ 0, so the missing epochs don't matter) |

AutoAugment vs. CutMix compare directly (same epochs). Whether either beats plain B at equal epochs needs `arch_B_e200`, which also answers "does longer training help?".

**Result so far.** Longer training explains most of the gain: B goes from 96.00% (100 epochs) to 96.52% (200 epochs), +0.52%. AutoAugment adds only +0.08% on top at 200 epochs (96.60%), which is too small to call a real effect from one run each. `arch_B_autoaug_e300` tests whether AutoAugment needs even more epochs to pay off; `arch_B_e300` is its control: if AutoAugment's gain over plain B grows from +0.08% (200 epochs) at 300 epochs, it needs more epochs to pay off; if not, it is dropped.

**Result at 300 epochs.** AutoAugment's gain over plain B grew from +0.08% (200 epochs) to **+0.48%** (300 epochs: 97.06% vs. 96.58%), while plain B barely moved from 200 to 300 epochs (96.52% → 96.58%). So plain B has plateaued, and AutoAugment needs the longer training to pay off. **B + AutoAugment, 300 epochs is the current best recipe.** One run per setting (the B repeat pair differed by 0.02%, so a 0.48% gap is likely real).

**Code change: CutMix** (`cutmix.py`, new; `main.py`, training loop).
- Enabled per config: `data.augmentation.cutmix: true`, `cutmix_alpha` (default 1.0), `cutmix_prob` (default 0.5). Configs without these keys behave exactly as before.
- With probability `cutmix_prob`, a batch is mixed: each image gets a box from a random partner in the batch, box area ~ Beta(α, α); `lam` is recomputed from the actual (border-clipped) box area. Loss = `lam · CE(y_a) + (1 − lam) · CE(y_b)`.
- Train accuracy on mixed batches is area-weighted the same way, so CutMix runs show lower train accuracy; validation is untouched.
- α = 1.0, p = 0.5 are the CutMix paper's CIFAR settings. Applied after cutout (both active), as in common pipelines.
- Verified: unit test (pasted area matches `lam` exactly; the original batch is not modified) and a 1-epoch smoke run of `main.py`. On startup the log prints `CutMix enabled (alpha=1.0, prob=0.5)`.

---

## Kaggle submissions and the validation → test gap

| Submission | Best val acc | Public leaderboard (1k images) |
|---|---|---|
| `arch_B_shift_deepen` (100 epochs) | 96.00% | 85.60% |
| `arch_B_autoaug_e200` | 96.60% | 86.70% |

The public score is ~10 points below validation for both, far more than leaderboard noise (~±1.4% on 1k images). The pipeline checks out: test images have the same format (32×32 RGB PNG), the same loader and test transform, and class predictions are balanced (a label or id mix-up would give ~10%, not ~86%). But the test images' pixel statistics differ from the training set's (mean 0.513/0.498/0.462 vs. 0.492/0.484/0.447; std ~0.27 vs. ~0.25): they are slightly brighter and higher in contrast.

So the test set most likely comes from a **shifted distribution**: new images of the same 10 classes, not held-out CIFAR-10 images. A 2,000-image test set fits CIFAR-10.1 (Recht et al., 2019), on which CIFAR-10 models typically lose several points. This is an inference, not confirmed.

Implications:
- Validation accuracy still ranks models (in Recht et al., gains on the original test set carried over to the new one), so model selection by validation stays.
- Changes that improve **robustness** (augmentation, especially color/contrast; TTA) may help more on the test set than validation shows. AutoAugment gained +0.08% on validation (vs. B at 200 epochs) but +1.1% on the public board vs. B at 100 epochs; that comparison mixes epochs and is within leaderboard noise, so it is a hint only.

---

## 6. Knowledge distillation: B × 2 teacher → B student (logit matching)

**Idea.** The parameter limit applies only to the submitted model. A larger teacher, trained without the limit, provides richer training targets for B: its raw logits encode which classes look alike (e.g. cat vs. dog), which a one-hot label does not.

**Plan.**
1. **Teacher** `teacher_B2x_aa_e300`: B's layout with channels doubled, C=[80,160,320,640] (19.9M params, 4.0× B's compute), trained with our best recipe (AutoAugment, 300 epochs) on the same 45k split, so the 5k val images stay unseen. Proceed only if it clearly beats the student's 97.06%.
2. **Student** `student_B_kd_e300` (45k): measures the gain over `arch_B_autoaug_e300` (97.06%).
3. **Final** `student_B_kd_e300_full` (all 50k), used only if step 2 shows a gain.

**Loss: MSE on raw logits** (Ba & Caruana, 2014), the student matches the teacher's logits, not its predicted class or its softened probabilities. Classic Hinton KD approaches logit matching at high temperature, and Kim et al. (2021) found MSE on logits at least as good as KL on CIFAR. `ce_weight: 0`, the student learns only from the teacher. The teacher sees the same augmented batch as the student (logits are computed on the fly, since AutoAugment creates a new variant every epoch). lr 0.1 is kept from the CE recipe; the MSE (mean over batch and classes) gives gradients of similar size, but this is not tuned.

**Why B × 2 rather than WRN-28-10.** WRN-28-10 has a published 97.4% with AutoAugment, but costs 11.3× B's compute (teacher ~2× longer to train, student ~2× slower), which does not fit before the deadline. B × 2 uses the recipe we know works, and a teacher close in structure to the student tends to distill better (Mirzadeh et al., 2020).

**Code changes** (`main.py`; all off unless a config enables them, so existing configs train as before):
- `distillation:` config block: `teacher_config`, `teacher_checkpoint`, `loss` (`mse` | `kl`), `temperature` (kl only), `ce_weight`. The teacher is frozen (eval mode, no gradients).
- `training.amp: true`: bf16 mixed precision (autocast), ~2× faster on A100. No loss scaler is needed with bf16. Used by the teacher and students; the earlier runs were fp32.
- **Proper resume.** Every epoch writes `last.pth` with weights, optimizer (momentum), LR schedule, epoch and best acc (written to a temp file, then renamed, so a kill mid-save can't corrupt it). `--resume` now loads `last.pth` and continues at the next epoch with the correct learning rate. Previously it reloaded only `best.pth` weights and restarted the LR schedule.
- `build_model()` factored out so the teacher is built the same way as the student.

**Verified locally** (tiny models, CPU, bf16 autocast on): a teacher killed after epoch 0 resumed at epoch 1 with exactly the expected cosine lr (0.075 of 3 epochs) and its best acc restored; students trained from its `best.pth` with both `mse` and `kl` losses (`ce_weight: 0`, so the teacher's logits were the only signal) and reached above-chance val accuracy after one epoch.

| Config | Params | Best val acc |
|---|---|---|
| `arch_B_autoaug_e300` (student baseline) | 4,976,530 | 97.06% |
| `teacher_B2x_aa_e300` | 19,879,450 | 97.18% (final 97.10%) |
| `student_B_kd_e300` | 4,976,530 | 96.96% (97.26% with flip TTA) |

**Teacher result.** 4× the parameters and compute bought only +0.12% over the student (97.18% vs. 97.06%), below the 97.4% bar set beforehand. With this recipe on 45k images, capacity is no longer the bottleneck. A single B × 2 teacher is therefore a weak teacher for B; next step is to measure an ensemble (B × 2 + `arch_B_autoaug_e300`) on the 5k val set before deciding.


---

## 7. Robustness round: ColorJitter vs. AugMix (judged on Kaggle)

**Why.** The public leaderboard score is ~10 points below val (86.70% vs. 96.60% for B + AA, 200 epochs; top team 91.4%). The test set looks like new natural photos ("a version of CIFAR-10" per the spec), measurably brighter, higher-contrast, with ~2× the clipped pixels. Val comes from the training distribution, so it can't show robustness gains; these runs are compared on Kaggle against `arch_B_autoaug_e300`.

| Config | Change vs. `arch_B_autoaug_e300` | Val | Kaggle (no TTA / flip TTA) |
|---|---|---|---|
| `arch_B_autoaug_e300` | — | 97.06% | 87.50% / **88.20%** |
| `arch_B_aa_cj_e300` | + ColorJitter (±20% brightness, contrast, saturation) | 96.98% | TBD / TBD |
| `arch_B_augmix_e300` | AugMix instead of AutoAugment | TBD | TBD / TBD |
| `arch_B_aa_pin_e300` | per-image normalization (each image by its own mean/std) | 96.66% | TBD / TBD |

**Code changes.** `normalization.py` (new): `data.normalization: per_image` normalizes each image by its own mean/std over all pixels and channels (removes global brightness/contrast, keeps color balance; std floored at 1/sqrt(N) as in TensorFlow); default `dataset` keeps the fixed CIFAR mean/std. Used by both `main.py` and `predict.py`, so test-time preprocessing always matches training (verified: default path reproduces the 87.50% submission byte for byte). `main.py`: `augmentation.augmix: true` adds torchvision's `AugMix` (defaults; without the paper's Jensen-Shannon consistency loss). `predict.py`: optional `--tta` (average softmax over image and mirror) and `--bn_adapt` (re-estimate BatchNorm statistics on the unlabeled test images; allowed by the spec? to be confirmed with the TA). Without flags, `predict.py` output is unchanged (verified: reproduces the 86.70% submission exactly).

**Test-time results for `arch_B_autoaug_e300`** (same model, three ways of predicting): plain 87.50%, flip TTA **88.20%** (+0.7), BN adaptation to the test images **84.90%** (−2.6). BN adaptation changed 151 predictions and made the score clearly worse, so the test set's difference from training is *not* mainly in low-level image statistics; the test images are better described as new photos than as a brightness/contrast-shifted copy. On the 5k val set, flip TTA gives only 97.06% → 97.10% (58 of 5,000 predictions change, net +2), much less than on Kaggle (+0.7), so TTA seems to help more on the shifted test images (with the caveat that one public score is noisy). Flip TTA is kept for the final submission; BN adaptation is dropped. This also lowers expectations for per-image normalization (same hypothesis).

Caveat on AugMix: it deliberately excludes brightness/contrast/color operations, which are the shift measured in the test set, so it is expected to be the weaker of the two here.

---

## 8. Depth experiment: deeper than B at the same budget

**Why.** Depth is the layout change that helped most so far (B, 34 convs, 96.00% vs. baseline 16 convs 95.56% and [3,3,3,2] 22 convs 95.74%), while width did not (A 95.68%; the B × 2 teacher only +0.12%). These push B's idea further: more blocks in stages 2-3, narrower channels to stay under 5M. Expected gain is modest (~+0.1-0.3% val).

| Config | Blocks | Channels | Params | Convs | Val | Kaggle |
|---|---|---|---|---|---|---|
| `arch_B_autoaug_e300` | [3,7,6,1] | [40,80,160,320] | 4,976,530 | 34 | 97.06% | 87.50% |
| `arch_D44_aa_e300` | [4,9,8,1] | [36,72,144,288] | 4,990,510 | 44 | 96.68% (96.86% TTA) | TBD |
| `arch_D56_aa_e300` | [5,12,10,1] | [32,64,128,256] | 4,775,850 | 56 | not run | — |

Same recipe as `arch_B_autoaug_e300` (AutoAugment, 300 epochs, fp32); only the layout differs. Compute per image ≈ B's, but wall time is likely somewhat longer (more sequential layers, narrower convs use the GPU less efficiently).

---

## 9. Paired comparison on the validation set (`compare_models.py`)

`compare_models.py` scores checkpoints on the 5k val split (same split as training), plain and with flip TTA, on the clean images and on shifted copies (brightness +0.03 / contrast ×1.1, Gaussian blur σ=0.7, JPEG quality 70), and runs exact McNemar tests between each pair (only images where exactly one model is right count; more sensitive than comparing accuracies).

| Model (plain / TTA) | clean | bright+contrast | blur | JPEG 70 |
|---|---|---|---|---|
| `arch_B_autoaug_e300` | 97.06 / 97.10 | 96.90 / 97.02 | 94.76 / 95.18 | 86.92 / 88.34 |
| `arch_B_aa_cj_e300` | 96.98 / 97.16 | 96.94 / 97.02 | 94.34 / 94.82 | 86.80 / 87.90 |
| `student_B_kd_e300` | 96.94 / **97.26** | 96.90 / **97.18** | **94.96** / 95.18 | **88.40 / 88.64** |

McNemar (TTA, clean): B vs. ColorJitter 49/52 images (p=0.84), B vs. student 37/45 (p=0.44), ColorJitter vs. student 43/48 (p=0.68): **no significant difference on clean images.** The student is best or tied in every column with TTA, which is consistent with distillation adding some robustness, but only the JPEG gap vs. ColorJitter approaches significance (p=0.05).

Findings: (1) the test-like brightness/contrast shift costs every model only ~0.1%, again ruling it out as the cause of the Kaggle gap (consistent with BN adaptation hurting). (2) JPEG compression hurts a lot (−9 points plain), so a JPEG-compressed test set would explain the gap, but the test images show *no* 8×8 JPEG block artifacts (block-boundary ratio 1.04 vs. 1.06 for train; JPEG q70 gives 1.18), so that explanation is ruled out too.

D44 added later (same script): 96.68 / 96.86 clean, 96.48 / 96.84 bright+contrast, 94.16 / 94.70 blur, 86.48 / 87.88 JPEG. It is below B and the student in every column. McNemar with TTA, clean: B vs. D44 59/47 (p=0.29); student vs. D44 58/38 (p=0.05). **More depth than B did not help** (D44 is, if anything, worse), so D56 (deeper still) was not run. Depth beyond B's 34 convs is past the useful range at this budget.


---

## 10. Optimizer experiment: SAM vs. AdamW (vs. our SGD)

| Config | Optimizer | lr / weight decay | Val | Kaggle (TTA) |
|---|---|---|---|---|
| `arch_B_autoaug_e300` | SGD, momentum 0.9 | 0.1 / 5e-4 | 97.06% | 88.20% |
| `arch_B_aa_sam_e300` | SAM on the same SGD, rho 0.05 | 0.1 / 5e-4 | TBD | TBD |
| `arch_B_aa_adamw_e300` | AdamW | 1e-3 / 0.05 (decoupled) | TBD | TBD |

Everything else is identical (B, AutoAugment, 300 epochs, cosine). Both new runs use bf16 AMP (the SGD baseline was fp32; AMP has not shown a measurable effect so far).

- **SAM** (Foret et al., 2021) keeps the SGD recipe; each step perturbs the weights by rho along the normalized gradient, recomputes the gradient there and applies it at the original weights, favoring flat minima. ~2× compute per step. BatchNorm running statistics are not updated by the second pass. rho = 0.05 is the paper's CIFAR default.
- **AdamW** needs its own hyperparameters; lr 1e-3 / wd 0.05 are common defaults, not tuned here (single run), so a result below SGD may partly reflect untuned settings.

**Code** (`main.py`): `training.optimizer: sgd | adamw | sam`, `training.sam_rho`. Default `sgd` (unchanged). Verified: unit test of the SAM step (weights restored exactly after the perturbed pass, BN running stats untouched by it, gradient differs from the first pass) and 1-epoch smoke runs of both configs.
