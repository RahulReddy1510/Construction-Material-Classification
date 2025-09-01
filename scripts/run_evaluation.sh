#!/bin/bash
echo "Evaluating baseline and compressed models..."
python evaluation/evaluate.py --model checkpoints/efficientnet_best.pth --config configs/base.yaml
python evaluation/evaluate.py --model checkpoints/qat_pruned_model.pth --config configs/qat.yaml --quantized
echo "Benchmarking inference..."
python evaluation/inference_benchmark.py
echo "Evaluation suite finished."
