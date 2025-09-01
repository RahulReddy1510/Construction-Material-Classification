import torch
import time
import numpy as np
import argparse

def benchmark_inference(model, input_size=(1, 3, 224, 224), n_runs=100, n_warmup=10):
    """
    Measures CPU inference latency.
    Goal: <30ms for compressed models.
    """
    model.eval()
    dummy_input = torch.randn(*input_size)
    
    # Warmup
    print(f"Warming up for {n_warmup} runs...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
            
    # Benchmark
    print(f"Benchmarking for {n_runs} runs...")
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print(f"\nResults (Batch Size {input_size[0]}):")
    print(f"  Mean Latency: {mean_lat:.2f} ms")
    print(f"  Std Dev:      {std_lat:.2f} ms")
    print(f"  P95 Latency:  {p95_lat:.2f} ms")
    print(f"  Throughput:   {1000/mean_lat:.2f} images/sec")
    
    return mean_lat

if __name__ == "__main__":
    # Demo with fresh model
    from models.efficientnet_finetune import ConstructionMaterialClassifier
    model = ConstructionMaterialClassifier(num_classes=5)
    print("--- Inference Benchmark (FP32 Baseline) ---")
    benchmark_inference(model)
