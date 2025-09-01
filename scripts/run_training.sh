#!/bin/bash
echo "Starting FP32 Fine-tuning Pipeline..."
python training/train.py --config configs/base.yaml
echo "Fine-tuning complete. Best model saved to checkpoints/efficientnet_best.pth"
