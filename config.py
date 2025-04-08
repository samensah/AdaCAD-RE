import argparse
import os
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Relation Extraction Model Configuration")
    
    # Model arguments
    parser.add_argument("--base_model_name", type=str, default="microsoft/Phi-4-mini-instruct",
                        help="Path or name of the base language model: microsoft/Phi-4-mini-instruct, ../Llama-3.2-3B-Instruct")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature for sampling (lower is more deterministic)")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="data/refind/test.json",
                        help="Path to the data file")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=23,
                        help="Random seed for reproducibility")
    
    # Experiment tracking
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging level")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    return args

# For importing in other modules
config = get_args()

if __name__ == "__main__":
    # Print all arguments when run directly
    args = get_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")