# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-15

### Added
- Final release ready for deployment.
- ONNX export scripts with INT8 support.
- Comprehensive inference benchmarking on CPU.
- Deployment demo notebook with real-time latency visualization.

## [0.9.0] - 2025-08-01

### Added
- Full evaluation suite across all models.
- ONNX Runtime integration for edge deployment.
- Inference benchmarking: achieved ~28ms on CPU with QAT+Prune.

## [0.8.0] - 2025-07-20

### Added
- Knowledge distillation experiments (ResNet-50 teacher).
- Documentation on why KD underperformed compared to QAT on this dataset.

## [0.7.0] - 2025-07-05

### Added
- Combined Quantization-Aware Training (QAT) and Structured Pruning.
- Achieved final model size of 4.5MB (78% reduction) with 92.3% accuracy.

## [0.6.0] - 2025-06-15

### Added
- Quantization-aware training pipeline (QAT) to recover accuracy loss from PTQ.
- FBGEMM backend configuration for x86 CPUs.

## [0.5.0] - 2025-06-01

### Added
- Post-Training Quantization (PTQ) analysis.
### Changed
- Pivoted from PTQ to QAT after noticing significant accuracy drop on EfficientNet.

## [0.4.0] - 2025-05-10

### Added
- EfficientNet-B0 fine-tuning with 3-phase training schedule.
- Achievement of 92.3% test accuracy on FP32 baseline.

## [0.3.0] - 2025-04-15

### Added
- Baseline models: ResNet-50 and MobileNetV2.
- Preliminary model comparison results.

## [0.2.0] - 2025-03-10

### Added
- Complete data pipeline including `dataset_prep.py` and `synthetic_dataset.py`.
- Class weights computation for handling imbalanced data.

## [0.1.0] - 2025-02-15

### Added
- Initial project structure.
- Problem definition and literature review notes.
- Selection of EfficientNet-B0 as the primary backbone.
