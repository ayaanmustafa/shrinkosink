# shrinkosink
# BYOL → KD ConvNeXt (STL-10) — 83% Accuracy

## Overview

This notebook implements a **self-supervised → distillation → supervised fine-tuning pipeline** to achieve strong performance on STL-10 with limited labeled data.

The key idea:

* Learn **rich representations without labels (BYOL)**
* Transfer them into a **smaller, efficient student (KD)**
* Finish with **supervised fine-tuning + regularization**

---

## Pipeline

### 1. Self-Supervised Pretraining (BYOL)

* Backbone: ConvNeXt (large teacher)
* Dataset: STL-10 *unlabeled split*
* Objective: learn invariant representations via two augmented views
* No negatives, no labels

Key components:

* Online + Target networks (EMA update)
* Projection + Prediction MLPs
* Cosine EMA momentum schedule

Outcome:

* Teacher learns **semantic features**, not just classification shortcuts

---

### 2. Feature Distillation (KD)

* Student: smaller ConvNeXt (with recurrence)
* Teacher: frozen BYOL encoder

Loss:

* Cosine similarity (feature alignment)
* Variance regularization (prevent collapse)

```text
Loss = alignment + 0.1 * variance penalty
```

Outcome:

* Student inherits **structure of representation space**
* Much faster + smaller model

---

### 3. Supervised Fine-Tuning

Dataset: STL-10 labeled (5k samples)

Enhancements:

* Label smoothing (0.1)
* Mixup (α = 0.2)
* RandAugment
* LR warmup + cosine decay
* KD loss (logits) combined with CE

```text
Final Loss = (1 - α) * CE + α * KD
```

Outcome:

* Improved generalization on small dataset
* Better calibration + robustness

---

### 4. Test-Time Augmentation (TTA)

* Multiple random crops per image
* Horizontal flip ensembling

Outcome:

* +1–2% accuracy boost at inference

---

## Architecture Choices

### Teacher

* Large ConvNeXt
* No recurrence
* High capacity → better feature learning

### Student

* Smaller ConvNeXt
* Recurrent refinement (2 passes)
* Projection layer for feature alignment

---

## Performance

| Stage                 | Role                       |
| --------------------- | -------------------------- |
| BYOL                  | Representation learning    |
| KD (features)         | Compress teacher knowledge |
| Fine-tune + KD logits | Task specialization        |
| TTA                   | Inference boost            |

**Final Accuracy: ~83% (STL-10 test set)**

---

## Why This Works

1. **BYOL removes label bottleneck**

   * Uses all unlabeled data
   * Learns transferable features

2. **KD bridges capacity gap**

   * Student mimics teacher geometry
   * Avoids training from scratch

3. **Aggressive regularization**

   * Mixup + label smoothing reduce overfitting
   * Critical for 5k samples

4. **Efficient training tricks**

   * `torch.compile`
   * AMP (mixed precision)
   * non_blocking transfers
   * persistent dataloaders

---

## Constraints Satisfied

* Single notebook pipeline
* Fast epochs (2–3 min BYOL feasible on GPU)
* Modular hyperparameters
* Teacher + student both included
* Easy prototyping via globals

---

## Key Takeaway

This is not just a model — it's a **training strategy**:

```text
Unlabeled Data → Representation → Compression → Supervised Refinement
```

That’s what pushes a small dataset setup to **~83% without external data or massive models**.
