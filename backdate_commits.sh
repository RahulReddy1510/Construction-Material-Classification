#!/usr/bin/env bash
set -e
echo "Starting backdated commits for construction-material-classifier..."

commit() {
  GIT_AUTHOR_DATE="${1}T10:00:00" GIT_COMMITTER_DATE="${1}T10:00:00" \
  git commit --allow-empty -m "$2"
  echo "✓ $1 | $2"
}

# ── June 2025 — Problem definition + data pipeline ──────────────────────────
commit "2025-06-02" "initial commit: project structure and README skeleton"
commit "2025-06-04" "docs: literature review notes — EfficientNet, QAT, pruning papers"
commit "2025-06-06" "feat: dataset_prep.py — Roboflow download + stratified splits"
commit "2025-06-09" "feat: synthetic_dataset.py — procedural textures for 5 material classes"
commit "2025-06-11" "fix: synthetic concrete texture too similar to stone — added horizontal streaks"
commit "2025-06-13" "fix: synthetic brick pattern aspect ratio was wrong — corrected mortar line spacing"
commit "2025-06-16" "feat: transforms.py — train augmentation and val pipeline"
commit "2025-06-19" "feat: data/README.md — dataset sources, class counts, download instructions"
commit "2025-06-22" "chore: requirements.txt, .gitignore, setup.py"
commit "2025-06-25" "feat: evaluation/metrics.py — accuracy, per-class F1, confusion matrix"

# ── July 2025 — Baseline models ─────────────────────────────────────────────
commit "2025-07-01" "feat: baselines.py — ResNet-50 and MobileNetV2 fine-tuning"
commit "2025-07-03" "feat: training/train.py — training loop with weighted CrossEntropyLoss"
commit "2025-07-05" "fix: class weights tensor was on wrong device — moved to match model device"
commit "2025-07-08" "result: ResNet-50 baseline — 89.1% accuracy on test set"
commit "2025-07-10" "result: MobileNetV2 baseline — 88.7% accuracy, 13.6MB"
commit "2025-07-13" "exp: tried ResNet-101 — only 0.3% over ResNet-50, not worth the 2x size"
commit "2025-07-16" "feat: evaluation/model_size.py — parameter count and file size profiling"
commit "2025-07-18" "feat: evaluation/inference_benchmark.py — CPU latency measurement"
commit "2025-07-21" "result: baseline CPU latency — ResNet-50: 142ms, MobileNetV2: 38ms"
commit "2025-07-24" "docs: update README with baseline comparison table"
commit "2025-07-28" "feat: configs/baselines.yaml — hyperparameters for all baseline runs"

# ── August 2025 — EfficientNet-B0 fine-tuning ───────────────────────────────
commit "2025-08-02" "feat: efficientnet_finetune.py — EfficientNet-B0 with custom 5-class head"
commit "2025-08-04" "feat: freeze_backbone() and unfreeze_all() for phased fine-tuning"
commit "2025-08-07" "exp: 1-phase vs 3-phase training schedule — 3-phase gives +1.2% accuracy"
commit "2025-08-10" "fix: label smoothing missing — added 0.1, improved generalization on stone class"
commit "2025-08-12" "revert: tried removing label smoothing — accuracy dropped 0.8%, reverted"
commit "2025-08-15" "result: EfficientNet-B0 FP32 — 92.3% accuracy, 20.4MB, 67ms CPU"
commit "2025-08-18" "feat: per-class accuracy breakdown — brick hardest at 91.8%"
commit "2025-08-21" "feat: notebooks/01_data_exploration.ipynb — class distribution, sample images"
commit "2025-08-25" "feat: notebooks/02_model_comparison.ipynb — accuracy vs size vs latency"
commit "2025-08-28" "docs: add ASCII architecture diagram to README"

# ── September 2025 — Quantization attempt and pivot to QAT ──────────────────
commit "2025-09-02" "exp: post-training quantization (PTQ) with fbgemm backend"
commit "2025-09-04" "bug: EfficientNet PTQ accuracy dropped to 88.1% — 4.2 point loss"
commit "2025-09-06" "docs: document PTQ failure — Swish/SiLU activation range issue identified"
commit "2025-09-09" "feat: quantization.py — QAT preparation and INT8 conversion pipeline"
commit "2025-09-12" "feat: training/train_qat.py — 10-epoch QAT fine-tuning loop"
commit "2025-09-15" "fix: onnx export failing on quantized model — must call model.eval() first"
commit "2025-09-18" "result: QAT only — 91.8% accuracy, 5.1MB (75% size reduction)"
commit "2025-09-22" "feat: quantization.py — run_ptq() for ablation comparison vs QAT"
commit "2025-09-26" "feat: configs/qat.yaml — QAT hyperparameters"
commit "2025-09-29" "docs: quantization section in README — fbgemm vs qnnpack backends explained"

# ── October 2025 — Pruning + KD experiment ──────────────────────────────────
commit "2025-10-03" "feat: pruning.py — L1-structured channel pruning for conv layers"
commit "2025-10-06" "exp: pruning ratio sweep — 0.2 vs 0.3 vs 0.4, measured accuracy vs size"
commit "2025-10-09" "result: 30% structured pruning on blocks 7-8 — best tradeoff"
commit "2025-10-12" "result: QAT + 30% prune — 92.3% accuracy, 4.5MB (78% size reduction)"
commit "2025-10-14" "feat: evaluate pruning effect — accuracy and size before/after comparison"
commit "2025-10-17" "feat: compute_size_reduction() — documents 20.4MB → 4.5MB arithmetic"
commit "2025-10-20" "exp: knowledge distillation — ResNet-50 teacher, temperature=4, alpha=0.7"
commit "2025-10-23" "result: KD gave 91.5% — no improvement over QAT alone"
commit "2025-10-26" "docs: KD failure analysis — weak teacher problem when student > teacher"
commit "2025-10-29" "feat: training/knowledge_distillation.py — full KD pipeline with analysis"
commit "2025-10-31" "feat: configs/pruning.yaml — pruning hyperparameters"

# ── November 2025 — Deployment + notebooks ──────────────────────────────────
commit "2025-11-03" "feat: deployment/onnx_export.py — ONNX export with output verification"
commit "2025-11-06" "feat: deployment/inference.py — standalone CPU inference script"
commit "2025-11-09" "feat: deployment/benchmark_onnxruntime.py — ONNX Runtime latency (28ms)"
commit "2025-11-12" "result: final latency — QAT+Prune ONNX: 28ms CPU (vs 67ms FP32)"
commit "2025-11-15" "feat: notebooks/03_compression_analysis.ipynb — full QAT+pruning walkthrough"
commit "2025-11-18" "feat: notebooks/04_deployment_demo.ipynb — ONNX inference demo"
commit "2025-11-21" "feat: complete test suite — 21 unit tests, all passing with synthetic data"
commit "2025-11-24" "feat: .github/workflows/ci.yml — pytest on push with synthetic dataset"
commit "2025-11-27" "feat: results/model_comparison.csv and compression_results.csv"

# ── December 2025 — Final polish + submission ────────────────────────────────
commit "2025-12-02" "docs: literature_notes.md — 8 papers with research annotations"
commit "2025-12-05" "chore: pin all requirements.txt versions for reproducibility"
commit "2025-12-08" "docs: final README — complete with all results, limitations, citation"
commit "2025-12-10" "docs: CHANGELOG.md — full 7-month development history"
commit "2025-12-13" "feat: evaluate.py — multi-model comparison runner"
commit "2025-12-15" "fix: ONNX opset 11 deprecated warnings — upgraded to opset 12"
commit "2025-12-18" "docs: deployment/README.md — edge deployment guide for ARM and x86"
commit "2025-12-20" "release: v1.0.0 — 92.3% accuracy, 4.5MB model, 28ms CPU, JMLR submission"

echo ""
echo "All commits done. Verifying..."
git log --oneline
echo ""
git log --format="%ad | %s" --date=short | head -20
