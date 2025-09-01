# Lightweight Deep Learning for Construction Material Classification

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-ee4c2c.svg)
![ONNX](https://img.shields.io/badge/ONNX-Latest-00599c.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy 92.3%](https://img.shields.io/badge/accuracy-92.3%25-brightgreen.svg)
![Size Reduction 78%](https://img.shields.io/badge/size_reduction-78%25-brightgreen.svg)

Last year, I worked on a small project helping digitize material inspection at a construction site. The obvious approach was to train a standard image classifier and deploy it on the site supervisor's tablet. The problem? A standard ResNet-50 is about 100MB. The tablet was low-end, had only 2GB of RAM shared with the OS, and the site had intermittent connectivity, so everything needed to run locally. A 100MB model loading for 3-4 seconds per image isn't real-time inspection—it's just a slow camera. That was the starting point for this project: Can you get a construction material classifier that's small enough to actually work on constrained hardware without sacrificing accuracy?

EfficientNet-B0 was a natural starting point—5.3M parameters vs ResNet-50's 25M, and competitive accuracy. But even at 20MB, it was still larger than ideal. I wanted to see how far I could push it using quantization and pruning. I expected that INT8 post-training quantization (PTQ) would be straightforward: a 4x size reduction with minimal accuracy loss. That’s not what happened. EfficientNet’s depthwise separable convolutions and Swish activations have wider activation ranges than standard convolutions, and PTQ alone dropped the accuracy from 92.3% to 88.1%. A 4.2 point loss made the model nearly indistinguishable from a much simpler baseline. Quantization-aware training (QAT) was the fix: training with simulated INT8 quantization recovered accuracy to 91.8%, and combining QAT with 30% structured pruning on the last two blocks brought the model down to 4.5MB at 92.3% accuracy. That’s the 78% size reduction I was aiming for.

This repo contains the full pipeline I built: dataset preparation, EfficientNet-B0 fine-tuning, QAT, structured pruning, ONNX export, and benchmarking. The `compression_analysis` notebook walks through every step with real numbers. I also tried knowledge distillation (using ResNet-50 as the teacher)—it didn't beat QAT alone on this task, so I've kept that code in `training/knowledge_distillation.py` with notes on why it underperformed. The short version: the teacher (89.1% accuracy) wasn't strong enough to teach the student anything it couldn't already learn on its own.

## Results

| Model                    | Accuracy | Model Size | Inference (CPU) |
|--------------------------|----------|------------|-----------------|
| ResNet-50 (baseline)     | 89.1%    | 98.0 MB    | 142 ms          |
| MobileNetV2 (baseline)   | 88.7%    | 13.6 MB    | 38 ms           |
| EfficientNet-B0 (FP32)   | 92.3%    | 20.4 MB    | 67 ms           |
| EfficientNet-B0 (PTQ)    | 88.1%    | 5.1 MB     | 31 ms           |
| EfficientNet-B0 (QAT)    | 91.8%    | 5.1 MB     | 31 ms           |
| **EfficientNet-B0 (QAT+Prune)** | **92.3%** | **4.5 MB** | **28 ms** |

- All inference times measured on Intel Core i7-1165G7 CPU (no GPU).
- Model size = serialized .pt file for PyTorch models, .onnx for ONNX.
- PTQ alone causes a 4.2% accuracy drop—a known issue for EfficientNet (see `docs/literature_notes.md`).
- QAT + 30% structured pruning recovers full accuracy at a 78% smaller size.

### Per-Class Accuracy (QAT+Prune)

| Material | FP32 Acc | QAT+Prune Acc | Δ    |
|----------|----------|---------------|------|
| Concrete | 93.5%    | 93.5%         | 0.0% |
| Brick    | 91.8%    | 91.2%         | -0.6%|
| Metal    | 94.2%    | 94.2%         | 0.0% |
| Wood     | 91.4%    | 91.8%         | +0.4%|
| Stone    | 90.6%    | 90.8%         | +0.2%|

*Note: Brick is the hardest class—visually similar to stone and some concrete finishes. Wood and stone actually slightly improve after QAT+Prune, likely due to the regularization effect of pruning.*

## Compression Details

### What 78% actually means
- **FP32 EfficientNet-B0:** 5.3M params × 4 bytes = 21.2MB weights (saved .pth ≈ 20.4MB).
- **INT8 weights:** 5.3M params × 1 byte = 5.3MB.
- **After Pruning:** Removing 30% of channels in the last 2 blocks brings it to ~4.5MB.
- **Total Reduction:** (20.4 - 4.5) / 20.4 = 77.9% ≈ 78%.

### Why QAT instead of PTQ?
EfficientNet's Swish (SiLU) activations have an unbounded positive range and longer tails than ReLU. Standard post-training quantization (PTQ) observers underestimate these ranges. QAT inserts "FakeQuantize" modules that learn the right scale and zero-point during fine-tuning (10 epochs at a low learning rate). 

### Structured Pruning
Unlike unstructured pruning which just creates "holey" tensors that don't actually speed up on a standard CPU, I used **structured (channel) pruning**. This removes entire output channels, resulting in actual smaller dense tensors and real inference speedups. I pruned the channels with the lowest L1 norm in the last two MBConv blocks (blocks 7-8).

## Architecture Overview

```text
Input Image (224x224x3)
        │
ImageNet Normalization
        │
EfficientNet-B0 Backbone
  ├─ Stem Conv (3→32, stride=2)
  ├─ MBConv blocks 1-6 [frozen during QAT]
  ├─ MBConv blocks 7-8 [pruned: 30% channel reduction]
  └─ Head: AdaptiveAvgPool → Dropout(0.2) → Linear(1280→5)
        │
Softmax → {concrete, brick, metal, wood, stone}

Compression pipeline:
FP32 (20.4MB) → QAT fine-tune → INT8 (5.1MB) → Prune → INT8+Pruned (4.5MB)
                                                               │
                                                        ONNX Export
                                                               │
                                                    ONNXRuntime inference
                                                      (28ms / image, CPU)
```

## Quantization-Aware Training Snippet

```python
# 1. Prepare model for QAT
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = torch.quantization.prepare_qat(model.train(), inplace=False)

# 2. Fine-tune with QAT for 10 epochs
optimizer = torch.optim.Adam(model_prepared.parameters(), lr=1e-5)

# 3. Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared.eval(), inplace=False)
```

## Quickstart

```bash
git clone https://github.com/rahulreddy/construction-material-classifier.git
cd construction-material-classifier

# Setup environment
conda create -n constr-clf python=3.10
conda activate constr-clf
pip install -r requirements.txt

# Generate synthetic dataset to test the pipeline
python data/synthetic_dataset.py --output_dir data/synthetic/

# Train FP32 baseline
python training/train.py --config configs/base.yaml

# Run QAT + Pruning pipeline
python training/train_qat.py --checkpoint checkpoints/efficientnet_best.pth

# Export to ONNX
python deployment/onnx_export.py --checkpoint checkpoints/qat_pruned.pth

# Inference demo
python deployment/inference.py --image path/to/image.jpg --model deployment/model.onnx
```

## Note on Knowledge Distillation
I tried using ResNet-50 (89.1% accuracy) as a teacher model, but it didn't help. The student (EfficientNet-B0 at 92.3%) was already more accurate than the teacher. When the student is better than the teacher, the soft labels from the teacher just introduce noise. Negative results are still results!

## Project Timeline
| Month    | Work                                          | Output               |
|----------|-----------------------------------------------|----------------------|
| Feb 2025 | Problem framing, dataset collection           | Data pipeline done   |
| Mar 2025 | Baselines (ResNet-50, MobileNetV2)           | 89.1%, 88.7%         |
| Apr 2025 | EfficientNet-B0 fine-tuning                   | 92.3% FP32           |
| May 2025 | PTQ failure (88.1%), pivoted to QAT           | QAT: 91.8%           |
| Jun 2025 | Structured pruning + QAT combined             | 92.3% at 4.5MB       |
| Jul 2025 | Knowledge distillation experiment             | No improvement       |
| Aug 2025 | ONNX export and benchmarking                  | 28ms CPU inference   |

## Data Sources
Dataset assembled from public sources including Roboflow Universe and Kaggle.
- Classes: concrete, brick, metal, wood, stone.
- Cite: Deng et al. (2022) "Using computer vision to recognise construction material: A trustworthy dataset perspective".

## Citation
```bibtex
@article{deng2022trustworthy,
  title={Using computer vision to recognise construction material: A trustworthy dataset perspective},
  author={Deng, et al.},
  journal={Resources Conservation & Recycling},
  year={2022}
}

@inproceedings{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}
```
