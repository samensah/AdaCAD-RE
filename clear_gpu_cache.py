import torch
import gc

def clear_gpu_cache():
    """
    Clears the GPU cache by emptying the CUDA cache and running garbage collection.
    This can help free up memory on your GPU.
    """
    # Empty the cache
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Empty CUDA cache
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        print(f"After clearing - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"After clearing - GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("GPU cache cleared successfully!")
    else:
        print("No GPU detected. CUDA is not available.")

if __name__ == "__main__":
    clear_gpu_cache()