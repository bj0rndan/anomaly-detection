import os
import torch
import torch.nn as nn
import torch.profiler
import time
from pathlib import Path
from detector import ResNetFeatureExtractor, FeatCAE

def log_gpu_usage():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        print("CUDA not available - running on CPU")

def run_inference_with_profiler(model, input_data, model_name, output_dir):
    """
    Run inference with profiling and save results.
    
    Args:
        model: The model to profile
        input_data: Input tensor for inference
        model_name: Name of the model for logging
        output_dir: Directory to save profiling results
    """
    print(f"\nRunning inference on {model_name}...")
    log_gpu_usage()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        profile_memory=True,
    ) as prof:
        with torch.profiler.record_function("model_inference"):
            with torch.no_grad():
                output = model(input_data)
            
            log_gpu_usage()
    
    # Print profiling results
    print("\nProfiling Results:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=10
    ))
    
    # Save results to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f'{model_name}_profiler_results.csv'
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Operation,CPU Time (ms),Device Time (ms),Called,CPU Memory (MB),Device Memory (MB)\n")
        for row in prof.key_averages():
            # Clean the key string to avoid formatting issues
            clean_key = str(row.key).replace(',', ';')
            
            # Format each field individually
            cpu_time = f"{row.cpu_time_total/1000:.2f}"
            device_time = f"{row.device_time_total/1000:.2f}" if torch.cuda.is_available() else "0.00"
            call_count = str(row.count)
            cpu_memory = f"{row.cpu_memory_usage/1024/1024:.2f}"
            device_memory = f"{row.device_memory_usage/1024/1024:.2f}" if torch.cuda.is_available() else "0.00"
            
            # Write the line with proper escaping
            f.write(f'"{clean_key}",{cpu_time},{device_time},{call_count},{cpu_memory},{device_memory}\n')
    
    print(f"\nProfiling results saved to: {csv_path}")
    return output, prof

def main():
    # Configuration
    input_size = (512, 512)
    model_path = Path('/home/jovyan/work/anomaly_detection_demo/modelsave/model.pth')
    output_dir = Path('/home/jovyan/work/anomaly_detection_demo/outputs')
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    fe_model = ResNetFeatureExtractor(layer2=True)
    fe_model.to(device)
    
    checkpoint = torch.load(model_path)
    cae_model = FeatCAE(in_channels=512, latent_dim=100)  # Adjust parameters as needed
    cae_model.load_state_dict(checkpoint['model_state_dict'])
    cae_model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512).to(device)
    
    # Profile feature extractor
    fe_output, fe_prof = run_inference_with_profiler(
        fe_model, 
        dummy_input,
        "FeatureExtractor_l2l3",
        output_dir
    )
    
    # Profile CAE
    cae_output, cae_prof = run_inference_with_profiler(
        cae_model,
        fe_output,
        "CAE_model",
        output_dir
    )

if __name__ == "__main__":
    main()
