# Literature Notes

Technical foundations and references for the construction-material-classifier.

## 1. EfficientNet: Rethinking Model Scaling (Tan et al., 2019)
- Introduced Compound Scaling (Depth, Width, Resolution).
- EfficientNet-B0 as a highly efficient FLOP-to-Accuracy baseline.
- **Project Note**: High baseline accuracy (92.3%) confirmed its suitability as a backbone.

## 2. Quantization-Aware Training (QAT)
- **Problem**: Post-Training Quantization (PTQ) fails on Swish/SiLU due to unbounded dynamic range.
- **Solution**: Simulated quantization during fine-tuning allows the optimizer to adjust weights to minimize bit-width discretization error.
- **Reference**: PyTorch Quantization API Documentation.

## 3. Structured Pruning for CPU Acceleration
- **Method**: L1-norm based channel removal.
- **Result**: Significant latency reduction on standard x86/ARM CPUs compared to unstructured (sparse) pruning.
- **Reference**: "Pruning Filters for Efficient ConvNets" (Li et al., 2016).
