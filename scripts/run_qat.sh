#!/bin/bash
echo "Starting Quantization-Aware Training..."
python training/train_qat.py --checkpoint checkpoints/efficientnet_best.pth --config configs/qat.yaml
echo "QAT and Pruning complete. Compressed model saved to checkpoints/qat_pruned_model.pth"
