import onnxruntime as ort
import numpy as np
import time
import argparse
import torch

def benchmark_onnx(onnx_path, n_runs=100):
    """
    Benchmarks ONNX model using ONNX Runtime.
    This simulates the actual field deployment on edge CPUs.
    """
    # Create session
    # Select providers (CPUExecutionProvider for edge deployments without GPU)
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Random input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
        
    # Benchmark
    print(f"Benchmarking {onnx_path} for {n_runs} runs...")
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    print(f"\nONNX Runtime Inference Results:")
    print(f"  Average Latency: {np.mean(latencies):.2f} ms")
    print(f"  P95 Latency:     {np.percentile(latencies, 95):.2f} ms")
    print(f"  Throughput:      {1000/np.mean(latencies):.2f} images/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ONNX model")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    args = parser.parse_args()
    
    benchmark_onnx(args.onnx)
